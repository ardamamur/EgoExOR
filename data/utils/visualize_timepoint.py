import os
import h5py
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from utils.constants import CAMERA_TYPE_MAPPING, GAZE_FIXATION, GAZE_FIXATION_TO_TAKE

def get_frame_annotations(h5_file, annotations_path, frame_id):
    """Get annotations for a specific frame."""
    if annotations_path is None:
        return None
    
    with h5py.File(h5_file, 'r') as f:
        ann_path = f'{annotations_path}/frame_{frame_id}/rel_annotations'
        if ann_path in f:
            return f[ann_path][:]
        return None

def _draw_hand_points(img_bgr: np.ndarray, hand_vals: np.ndarray):
    """Draw hand tracking points and connecting lines.
    
    Args:
        img_bgr: BGR image to draw on
        hand_vals: 16-element vector [LWx, LWy, LPx, LPy, RWx, RWy, RPx, RPy, 
                                    LWnx, LWny, LPnx, LPny, RWnx, RWny, RPnx, RPny]
    """
    colors = [
        (255, 0, 0),    # LW => Blue
        (255, 0, 0),    # LP => Blue
        (0, 255, 0),    # RW => Green
        (0, 255, 0),    # RP => Green
        (255, 128, 0),  # LW normal => Orange
        (255, 128, 0),  # LP normal => Orange
        (0, 255, 255),  # RW normal => Yellow
        (0, 255, 255)   # RP normal => Yellow
    ]
    
    # Draw circles for each hand point
    for i in range(8):
        x = hand_vals[2 * i]
        y = hand_vals[2 * i + 1]
        if not np.isnan(x) and not np.isnan(y):
            xi, yi = int(x), int(y)
            if 0 <= xi < img_bgr.shape[1] and 0 <= yi < img_bgr.shape[0]:
                cv2.circle(img_bgr, (xi, yi), 4, colors[i], thickness=-1)

    # Draw lines connecting base points to corresponding normal tip points
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]  # Base to normal point connections
    for base_idx, tip_idx in pairs:
        bx = hand_vals[2 * base_idx]
        by = hand_vals[2 * base_idx + 1]
        tx = hand_vals[2 * tip_idx]
        ty = hand_vals[2 * tip_idx + 1]
        if (not np.isnan(bx) and not np.isnan(by) and
            not np.isnan(tx) and not np.isnan(ty)):
            xi_b, yi_b = int(bx), int(by)
            xi_t, yi_t = int(tx), int(ty)
            if (0 <= xi_b < img_bgr.shape[1] and 0 <= yi_b < img_bgr.shape[0] and
                0 <= xi_t < img_bgr.shape[1] and 0 <= yi_t < img_bgr.shape[0]):
                cv2.line(img_bgr, (xi_b, yi_b), (xi_t, yi_t), colors[tip_idx], thickness=2)    

def apply_lut(frame):
    """Apply a LUT (Look-Up Table) to the BGR frame to adjust colors."""
    try:
        from pathlib import Path
        from pillow_lut import load_cube_file
        from PIL import Image
        import cv2
        import numpy as np
        
        # Define paths to check for LUT files
        possible_lut_paths = [
            Path("utils/egoexor.cube"),  # Main project location
            Path("egoexor.cube"),        # Alternative location
        ]
        
        # Try to find an existing LUT file
        lut_path = None
        for path in possible_lut_paths:
            if path.exists():
                lut_path = str(path)
                break
                
        if lut_path is None:
            return frame
            
        # Apply LUT
        lut = load_cube_file(lut_path)
        pil_image = Image.fromarray(frame)  # Convert BGR NumPy array to PIL Image
        lut_image = pil_image.filter(lut)  # Apply LUT
        # Convert back to NumPy array and ensure BGR format
        frame = np.array(lut_image)  # PIL Image to NumPy array (RGB)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr
        
    except (ImportError, FileNotFoundError, Exception) as e:
        print(f"Warning: Failed to apply LUT: {e}")
        return frame
    
def _needs_fixation(role: str, take_path: str) -> bool:
    """
    Return True if the given (role, take_path) combination requires
    the extra gaze-fixation offset.
    """
    takes_for_role = GAZE_FIXATION_TO_TAKE.get(role, [])
    return take_path in takes_for_role

def draw_camera_label(frame_bgr, text, font=cv2.FONT_HERSHEY_SIMPLEX,
                      scale=0.5, color=(255, 255, 255), thickness=1):
    """Draw semi-transparent label box on the image."""
    x, y = 5, 25
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (text_w + 10, text_h + 10), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def create_mosaic(image_dict, title=None, ncols=4, figsize=(18, 10)):
    """Display multiple images in a mosaic view using matplotlib."""
    n_images = len(image_dict)
    nrows = (n_images + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16)

    axes = axes.flatten()
    for ax, (label, img) in zip(axes, image_dict.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    for ax in axes[n_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def visualize_frame_group(
    h5_path: str = None,
    h5_file: h5py.File = None,
    surgery_type: str = None,
    procedure_id: int = None,
    take_id: int = None,
    frame_idx: int = 0,
    save_frames: bool = False,
    figures_dir: str | os.PathLike = "figures",
):
    """
    Load RGB, gaze, and hand data from a specific frame and show in a mosaic.

    You must provide either `h5_path` or an open `h5_file` handle.
    Any cam whose frame isn’t a valid H×W×3 array will be skipped.
    """
    # --- open file if needed ---
    if h5_file is None:
        if h5_path is None:
            raise ValueError("Must provide either h5_path or h5_file")
        f = h5py.File(h5_path, "r")
        close_when_done = True
    else:
        f = h5_file
        close_when_done = False

    cam_overlay = {}
    aria_roles = {"head_surgeon", "assistant", "circulator", "anesthetist", "or_light", "microscope"}
    take_path = f"/data/{surgery_type}/{procedure_id}/take/{take_id}"

    try:
        rgb = f[f"{take_path}/frames/rgb"][frame_idx]  # (n_cams, H, W, 3)
        sources_group = f[f"{take_path}/sources"]
        source_map = {
            i: sources_group.attrs[f"source_{i}"]
            for i in range(sources_group.attrs["source_count"])
        }

        gaze_data = f.get(f"{take_path}/eye_gaze/coordinates")
        hand_data = f.get(f"{take_path}/hand_tracking/positions")
        annotations_path = f"{take_path}/annotations"

        for cam_idx, cam_name in source_map.items():
            frame = rgb[cam_idx].copy()
            if isinstance(cam_name, (bytes, bytearray)):
                cam_name = cam_name.decode("utf-8")
            if cam_name not in aria_roles and cam_name not in ["simstation", "ultrasound"]:
                frame = apply_lut(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            draw_camera_label(frame, cam_name)

            expected_type = CAMERA_TYPE_MAPPING[cam_name]

            if cam_name in aria_roles and gaze_data is not None:
                gaze_points = gaze_data[frame_idx]
                for g in gaze_points:
                    if g[0] == expected_type and not np.allclose(g[1:], [-1, -1]):
                        # apply offset iff this (role, take) needs fixation
                        gx, gy = int(g[1]), int(g[2])
                        if _needs_fixation(cam_name, take_path):
                            # print(f"Applying fixation offset for {cam_name} at frame {frame_idx}")
                            gx += GAZE_FIXATION["x"]
                            gy += GAZE_FIXATION["y"]

                        if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
                            cv2.circle(frame, (gx, gy), 6, (255, 0, 0), -1)

            if cam_name in aria_roles and hand_data is not None:
                hands = hand_data[frame_idx]
                for h in hands:
                    if h[0] == expected_type:
                        pts = h[1:].reshape(8, 2)
                        _draw_hand_points(frame, pts.flatten())

            
            # Save ready‑for‑display BGR copy
            frame_bgr_ready = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cam_overlay[cam_name] = frame_bgr_ready

            # ---- optional per‑camera export ----
            if save_frames:
                fig_path = Path(figures_dir)
                fig_path.mkdir(parents=True, exist_ok=True)
                fn = f"{take_path.strip('/').replace('/', '_')}_frame{frame_idx}_{cam_name}.png"
                cv2.imwrite(str(fig_path / fn), frame_bgr_ready)


        frame_annotation = f[f'{annotations_path}/frame_{frame_idx}/rel_annotations'][:]
        if frame_annotation is not None:
            # Choose an icon (e.g. 📊, 🖼️, 🔍) and build your title
            icon = "📊"
            title_text = f"{icon} Scene Graph Annotation {icon}"
            border = "=" * len(title_text)

            # Print a “pretty” header
            print(f"\n{border}\n{title_text}\n{border}\n")

            # Now your original loop…
            for row in frame_annotation:
                print("· " + " ".join(x.decode() for x in row))


    finally:
        if close_when_done:
            f.close()

    create_mosaic(cam_overlay, title=f"{surgery_type} Procedure {procedure_id} Take {take_id} @ Frame {frame_idx}")


# Example usage
# visualize_frame_group(hf_file, "Ultrasound", 5, 1, frame_idx=1000)
