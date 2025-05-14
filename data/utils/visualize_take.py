import h5py
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import subprocess
from typing import Tuple
from utils.constants import CAMERA_TYPE_MAPPING, EGOCENTRIC_SOURCES, EXOCENTRIC_SOURCES, GAZE_FIXATION, GAZE_FIXATION_TO_TAKE
from utils.visualize_timepoint import draw_camera_label, _needs_fixation, apply_lut, _draw_hand_points

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize EgoExOR Take")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./", 
        required=True,
        help="Directory where data files are stored"
    )
    parser.add_argument(
        "--h5_file", 
        type=str,  
        required=True,
        help=" Name of the input HDF5 file to visualize take from."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Name of the output file. it will be saved in data_dir/visualization folder."
    )

    parser.add_argument(
        "--surgery_type", 
        type=str, 
        required=True,
        help="Name of the surgery type. [Ultrasound, MISS]."
    )

    parser.add_argument(
        "--procedure_id", 
        type=int, 
        required=True,
        help="ID of the procedure from the desired surgery type."
    )

    parser.add_argument(
        "--take_id", 
        type=int, 
        required=True,
        help="ID of the TAKE from the desired surgery_type/procedure."
    )

    parser.add_argument(
        "--debug_limit", 
        type=int, 
        default=None,
        help="Set the number of frames you want to include in video. Debug purposes."
    )

    return parser.parse_args()

def load_data(h5_file, surgery_type, procedure_id, take_id):
    """Load all required data from the HDF5 file for a specific subclip."""
    with h5py.File(h5_file, 'r') as f:
        base_path = f'/data/{surgery_type}/{procedure_id}/take/{take_id}'
        
        # Check if the specified subclip exists
        if base_path not in f:
            raise ValueError(f"Take not found: {base_path}")
        
        # Load RGB frames
        rgb = f[f'{base_path}/frames/rgb'][:]
        
        # Load sources metadata
        sources_group = f[f'{base_path}/sources']
        source_count = sources_group.attrs['source_count']
        sources = {}
        for i in range(source_count):
            source_name = sources_group.attrs[f'source_{i}']
            sources[i] = source_name.decode('utf-8') if isinstance(source_name, bytes) else source_name
        
        # Load eye gaze data if available
        eye_gaze = None
        eye_gaze_depth = None
        if f'{base_path}/eye_gaze/coordinates' in f:
            eye_gaze = f[f'{base_path}/eye_gaze/coordinates'][:]
            eye_gaze_depth = f[f'{base_path}/eye_gaze_depth/values'][:]
        
        # Load hand tracking data if available
        hand_tracking = None
        if f'{base_path}/hand_tracking/positions' in f:
            hand_tracking = f[f'{base_path}/hand_tracking/positions'][:]
        
        # For annotations, we'll check if the annotations directory exists, but we'll load them per frame
        annotations_dir_exists = f'{base_path}/annotations' in f
        
        # Load vocabulary for annotations
        entity_vocab = {}
        relation_vocab = {}
        if '/metadata/vocabulary/entity' in f:
            for name, id_ in f['/metadata/vocabulary/entity']:
                name = name.decode('utf-8') if isinstance(name, bytes) else name
                entity_vocab[id_] = name
        if '/metadata/vocabulary/relation' in f:
            for name, id_ in f['/metadata/vocabulary/relation']:
                name = name.decode('utf-8') if isinstance(name, bytes) else name
                relation_vocab[id_] = name
        
        # Load audio waveform
        audio = None
        if f'{base_path}/audio/waveform' in f:
            audio = f[f'{base_path}/audio/waveform'][:]
        
        return {
            'rgb': rgb,
            'sources': sources,
            'eye_gaze': eye_gaze,
            'eye_gaze_depth': eye_gaze_depth,
            'hand_tracking': hand_tracking,
            'h5_file': h5_file,
            "take_path": base_path,
            'annotations_path': f'{base_path}/annotations' if annotations_dir_exists else None,
            'entity_vocab': entity_vocab,
            'relation_vocab': relation_vocab,
            'audio': audio
        }
    
def get_frame_annotations(h5_file, annotations_path, frame_id):
    """Get annotations for a specific frame."""
    if annotations_path is None:
        return None
    
    with h5py.File(h5_file, 'r') as f:
        ann_path = f'{annotations_path}/frame_{frame_id}/rel_annotations'
        if ann_path in f:
            return f[ann_path][:]
        return None
    

def _choose_grid_layout(num_cameras: int) -> Tuple[int, int]:
    """Choose grid layout for mosaic visualization."""
    if num_cameras <= 0:
        return (1, 1)
    import math
    rows = int(math.floor(math.sqrt(num_cameras)))
    cols = int(math.ceil(num_cameras / rows))
    return (rows, cols)


def _write_stereo_wav(wav_path: str, stereo_data: np.ndarray, sample_rate: int = 48000):
    """Write stereo audio data to WAV file."""
    import soundfile as sf
    max_value = np.max(np.abs(stereo_data))
    if max_value > 0:
        stereo_data /= max_value
    sf.write(wav_path, stereo_data, samplerate=sample_rate)
    print(f"Mixed audio saved to '{wav_path}'.")


def visualize_take(h5_file, 
                   surgery_type, procedure_id, take_id, 
                   output_path, 
                   fps=15,
                   include_audio= True,
                   debug_limit=None):
    take_data = load_data(h5_file, surgery_type, procedure_id, take_id)
    rgb_video = take_data['rgb']
    gaze_data = take_data['eye_gaze']
    gaze_depth_data = take_data['eye_gaze_depth']
    hand_data = take_data['hand_tracking']
    sources = take_data['sources']
    take_path = take_data['take_path']
    global_audio = take_data['audio']
    h5_file_path = take_data['h5_file']
    annotations_path = take_data['annotations_path']
    
    num_frames, num_cameras, frame_h, frame_w, _ = rgb_video.shape

    mosaic_rows = 3
    mosaic_cols = (num_cameras + mosaic_rows - 1) // mosaic_rows

    frames_mosaic_h = mosaic_rows * frame_h
    frames_mosaic_w = mosaic_cols * frame_w

    text_annotations_height = frame_h // 2 

    mosaic_w = frames_mosaic_w
    mosaic_h = frames_mosaic_h + text_annotations_height

    # Create temporary video without audio
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_noaudio_path = f"temp_noaudio_{surgery_type}_{procedure_id}_{take_id}.mp4"
    writer = cv2.VideoWriter(temp_noaudio_path, fourcc, fps, (mosaic_w, mosaic_h))
    if debug_limit is not None:
        num_frames = debug_limit
        
    for f_idx in tqdm(range(num_frames), desc="Processing frames"):
        # Create a blank image for the mosaic
        mosaic_img = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        # Track which positions in the grid are filled
        occupied_positions = set()

        # Process camera frames
        for cam_idx, cam_name in sources.items():
            frame = rgb_video[f_idx, cam_idx].copy()
            if cam_name not in EGOCENTRIC_SOURCES and cam_name not in ["or_light", "microscope", "simstation", "ultrasound"]:
                # rgbd video from azure kinect sources neeeded to be applied LUT in order to make them more similar to the other cameras
                frame = apply_lut(frame)

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            expected_type = CAMERA_TYPE_MAPPING[cam_name]
            if cam_name in EGOCENTRIC_SOURCES and gaze_data is not None:
                gaze_points = gaze_data[f_idx]
                for g in gaze_points:
                    if g[0] == expected_type and not np.allclose(g[1:], [-1, -1]):
                        # apply offset iff this (role, take) needs fixation
                        gx, gy = int(g[1]), int(g[2])
                        if _needs_fixation(cam_name, take_path):
                            # print(f"Applying fixation offset for {cam_name} at frame {f_idx}")
                            gx += GAZE_FIXATION["x"]
                            gy += GAZE_FIXATION["y"]
                        
                        if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
                            cv2.circle(frame, (gx, gy), 6, (255, 0, 0), thickness=-1)


            if cam_name in EGOCENTRIC_SOURCES and hand_data is not None:
                hands = hand_data[f_idx]
                for h in hands:
                    if h[0] == expected_type:
                        pts = h[1:].reshape(8, 2)
                        _draw_hand_points(frame, pts.flatten())

            # Add depth info to label if available
            depth_str = ""
            if gaze_depth_data is not None:
                for i, g in enumerate(gaze_data[f_idx]):
                   if g[0] == expected_type: 
                       depth = gaze_depth_data[f_idx, i]
                       if not np.isnan(depth):
                           depth_str = f" | Depth: {depth:.2f}m"
                           break
                       
            draw_camera_label(frame, f"{cam_name}{depth_str}")



            # Insert frame into 3-row mosaic layout
            r = cam_idx // mosaic_cols
            c = cam_idx % mosaic_cols
            y_start = r * frame_h
            x_start = c * frame_w
            mosaic_img[y_start:y_start + frame_h, x_start:x_start + frame_w] = frame
            occupied_positions.add((r, c))

        # Create and add scene graph visualization (double-wide now)
        current_annotations = None
        if annotations_path is not None:
            current_annotations = get_frame_annotations(h5_file_path, annotations_path, f_idx)
        # Create text annotation section (below camera frames)
        
        text_annotation_section = np.ones((text_annotations_height, 
                                           mosaic_w, 3), dtype=np.uint8) * 240  # Light gray background
        

        # Add gradient header for text annotations
        header_height = 30
        header = np.zeros((header_height, mosaic_w, 3), dtype=np.uint8)
        for i in range(header_height):
            header[i] = [180, 130, 70]  # Steel blue color

        text_annotation_section[0:header_height] = header
        cv2.putText(text_annotation_section, "Scene Graph", (10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        if current_annotations is not None and len(current_annotations) > 0:
            max_annotations_per_col = (text_annotations_height - 40) // 20
            col_width = mosaic_w // 3

            for idx, (subj, pred, obj) in enumerate(current_annotations):
                if isinstance(subj, bytes):
                    subj = subj.decode('utf-8')
                if isinstance(pred, bytes):
                    pred = pred.decode('utf-8')
                if isinstance(obj, bytes):
                    obj = obj.decode('utf-8')
                
                col = idx // max_annotations_per_col
                row = idx % max_annotations_per_col
                
                if col >= 3:  # Limit to 3 columns
                    break
                    
                x_pos = 10 + col * col_width
                y_pos = 50 + row * 20
                
                # Draw text with nice formatting
                annotation_text = f"{subj},{obj};{pred}"
                cv2.putText(text_annotation_section, annotation_text, (x_pos, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)               
        else:
            # No annotations message
            cv2.putText(text_annotation_section, "No annotations available for this frame", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)       
        # Add text annotation row
        y_start = frames_mosaic_h
        mosaic_img[y_start:y_start + text_annotations_height, 0:mosaic_w] = text_annotation_section
        
        writer.write(mosaic_img)
    
    writer.release()
    print(f"Video saved to '{temp_noaudio_path}'.")

    if include_audio:
        if global_audio is not None and global_audio.size > 0:
            sample_rate = 48000
            temp_wav_path = "temp_audio.wav"
            _write_stereo_wav(temp_wav_path, global_audio, sample_rate=sample_rate)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", temp_noaudio_path,
                "-i", temp_wav_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                "-shortest", output_path
            ]
            print("[Visualizer] Muxing audio + video with ffmpeg...")
            proc = subprocess.run(ffmpeg_cmd, capture_output=True)
            if proc.returncode != 0:
                print("FFmpeg error:", proc.stderr.decode("utf-8"))
            else:
                print(f"[Visualizer] Final video saved to {output_path}")

            # Clean up temporary files
            if os.path.exists(temp_noaudio_path):
                os.remove(temp_noaudio_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
    else:
        os.rename(temp_noaudio_path, output_path)


def main():
    """Main function to execute the script."""
    args = parse_args()

    try:
        from pathlib import Path
        print(f"Loading data for surgery: {args.surgery_type}, procedure: {args.procedure_id}, take: {args.take_id}")
        data_dir=Path(args.data_dir)
        h5_file = data_dir / args.h5_file 
        output_dir = data_dir / "visualization"
        os.makedirs(output_dir, exist_ok=True)  # Ensure visualization directory exists
        output_path = output_dir / args.output_file
        # Call the visualize_subclip function with the parsed arguments
        visualize_take(
            h5_file=h5_file,
            surgery_type=args.surgery_type,
            procedure_id=args.procedure_id,
            take_id=args.take_id,
            output_path=output_path,
            debug_limit = args.debug_limit,
            fps=15,
            width=1920,
            height=1080
        )
        
        print(f"Visualization complete. Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0
    
if __name__ == "__main__":
    exit(main()) 