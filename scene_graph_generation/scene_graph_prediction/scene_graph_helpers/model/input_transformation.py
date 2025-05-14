import torch
import torchaudio
import transformers
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import ClapModel, ClapProcessor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scene_graph_generation.scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import reversed_sources, SOURCES, GAZE_FIXATION, GAZE_FIXATION_TO_TAKE


class SpeechProcessor:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialize the speech processor with Whisper for speech-to-text transcription.
        
        Args:
            model_name (str): Name or path of the pretrained Whisper model (e.g., 'openai/whisper-small').
                              Compatible with transformers==4.31.0.
        """
        # Load Whisper model and processor
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper_model.eval()
        # Freeze model parameters to save memory and prevent training
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        self.processor = WhisperProcessor.from_pretrained(model_name)
    
    def __call__(self, audio: torch.Tensor, orig_sr=48000) -> dict:
        """
        Process audio input to extract speech transcriptions using Whisper.
        
        Args:
            audio (torch.Tensor): Input audio tensor of shape [batch_size, sequence_length, num_channels]
                                 or [batch_size, sequence_length].
            orig_sr (int): Original sampling rate of the audio (default: 48000, as used in AudioProcessor).
        
        Returns:
            dict: Dictionary containing transcriptions for each audio sample in the batch.
        
        Note:
            - Audio is resampled to 16kHz (Whisper requirement) if orig_sr is different.
            - Requires transformers==4.31.0, torch==2.0.1, torchaudio==2.0.2.
        """
        audio = audio.clone()
        device = audio.device
        
        # Handle audio dimensions and convert to mono if stereo
        if audio.dim() == 3:  # [batch_size, sequence_length, num_channels]
            if audio.shape[-1] == 2:  # Stereo
                audio = audio.mean(dim=-1)  # Average across stereo channels to mono
        elif audio.dim() == 2:  # [batch_size, sequence_length]
            pass
        else:
            raise ValueError("Expected audio tensor of shape [batch_size, sequence_length, num_channels] or [batch_size, sequence_length]")
        
        audio = audio.to(dtype=torch.float32)
        
        # Resample audio to 16kHz if needed
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
            audio = resampler(audio)
        
        # Process audio with Whisper processor
        inputs = self.processor(
            audio=audio.cpu().numpy(),  # Convert to numpy for processor
            sampling_rate=16000,        # Whisper expects 16kHz
            return_tensors="pt"         # Return PyTorch tensors
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device, dtype=torch.float32) for k, v in inputs.items()}
        
        # Perform inference with no gradients
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                inputs["input_features"],
                max_length=448,  # Reasonable max length for transcriptions
                num_beams=5      # Beam search for better accuracy
            )
        
        # Decode the predicted token IDs to text
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return {"transcriptions": transcriptions}
    
class AudioProcessor:
    def __init__(self, model_name="laion/larger_clap_general", d_model=1024, clap_hidden_size=512):
        """
        Initialize the audio encoder with CLAP feature extraction.
        
        Args:
            d_model (int): The target embedding dimension for the Transformer.
            dropout (float): Dropout rate for the projector.
            pretrained_clap (str): Path or name of the pretrained CLAP model checkpoint.
        """
        super().__init__()
        self.clap_model = ClapModel.from_pretrained(model_name)
        self.clap_model.eval()
        for param in self.clap_model.parameters():
            param.requires_grad = False
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.clap_feature_dim = clap_hidden_size
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        # Input: [sequence_length, snippet_length, num_channels]
        audio = audio.clone()
        device = audio.device
        # Normalize by max absolute value per snippet
        B, sample_rate, stereo_channel = audio.shape  # e.g., [8, 1, 48000, 2]
        if stereo_channel == 2:
            #audio = self.to_mono(audio.permute(0, 2, 1)).permute(0, 2, 1)  # [8, 48000, 1]
            audio = audio.mean(dim=-1)  # [8, 48000], average across stereo channels
            #audio = audio.squeeze(-1)  # [8, 48000]
        audio = audio.to(dtype=torch.float32)
        inputs = self.processor(
            audios=audio.cpu().numpy(),            # Convert to numpy for processor
            return_tensors="pt",                   # Return PyTorch tensors
            sampling_rate=48000                    # Specify sample rate
        )
        inputs = {k: v.to(device, dtype=torch.float32) for k, v in inputs.items()}
        with torch.no_grad():  # No gradients for pretrained model
            audio_features = self.clap_model.get_audio_features(**inputs)
        return audio_features
    

class AudioTransform:
    def __init__(self):
        # Simple normalization; could add augmentation like noise later
        pass
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Input: [sequence_length, snippet_length, num_channels]
        data = data.clone()
        # Normalize by max absolute value per snippet
        max_vals = data.abs().max(dim=1, keepdim=True)[0]  # Max per snippet
        data /= (max_vals + 1e-6)  # Avoid division by zero
        return data

def _needs_fixation(role: str, take_path: str) -> bool:
    """
    Return True if the given (role, take_path) combination requires
    the extra gaze-fixation offset.
    """
    takes_for_role = GAZE_FIXATION_TO_TAKE.get(role, [])
    #print(f"Checking fixation for {role} in {take_path}")
    return take_path in takes_for_role
class GazeNormalize:
    def __init__(self, img_width: int = 336, img_height: int = 336):
        self.img_width = img_width
        self.img_height = img_height
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone()
        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        data[..., 0] = torch.clamp(data[..., 0] / self.img_width, 0, 1)
        data[..., 1] = torch.clamp(data[..., 1] / self.img_height, 0, 1)
        return data

class GazeDepthNormalize:
    def __init__(self, max_depth: float = 1.0):
        self.max_depth = max_depth
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone()
        data = torch.nan_to_num(data, nan=0.0, posinf=self.max_depth, neginf=0.0)
        data = torch.clamp(data / self.max_depth, 0, 1)
        return data

class HandTrackingNormalize:
    def __init__(self, img_width: int = 336, img_height: int = 336):
        self.img_width = img_width
        self.img_height = img_height
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone()
        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        data[..., 0::2] = torch.clamp(data[..., 0::2] / self.img_width, 0, 1)
        data[..., 1::2] = torch.clamp(data[..., 1::2] / self.img_height, 0, 1)
        return data

# Modified FrameTransform class
class FrameTransform:
    def __init__(self, processor, augment=None, pad_to_square=True):
        """
        Initialize FrameTransform with an image processor, optional augmentation, and padding option.
        
        Args:
            processor: Image processor (e.g., CLIPProcessor) for preprocessing.
            augment: Optional augmentation function (e.g., torchvision transforms).
            pad_to_square: Whether to pad images to square using processor.image_mean.
        """
        self.processor = processor
        self.augment = augment
        self.pad_to_square = pad_to_square

    def expand2square(self, pil_img, background_color):
        """
        Pad a PIL image to a square by adding borders with the specified background color.
        
        Args:
            pil_img: PIL.Image object.
            background_color: Tuple of RGB values (0-255) for padding.
        
        Returns:
            PIL.Image: Square image with padding.
        """
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __call__(self, data):
        """
        Process a single PIL Image or a list/tensor of images.
        
        Args:
            data: PIL.Image, torch.Tensor [H, W, 3], or [cameras, H, W, 3].
        
        Returns:
            torch.Tensor: Processed image(s) in [C, H, W] or [N, C, H, W] format, or None if invalid.
        """
        if isinstance(data, Image.Image):
            # Process single PIL Image
            if self.augment is not None:
                data = self.augment(data)

            # Pad to square if enabled
            if self.pad_to_square:
                # Use processor.image_mean for background color (convert to 0-255 range)
                background_color = tuple(int(x * 255) for x in self.processor.image_mean)
                data = self.expand2square(data, background_color)

            # Preprocess with processor
            processed = self.processor.preprocess(data, return_tensors='pt')['pixel_values']
            return processed.squeeze(0).to(dtype=torch.bfloat16)

        elif isinstance(data, torch.Tensor):
            shape = data.shape
            assert shape[-1] == 3, f"Expected 3 channels, got {shape[-1]}"

            if len(shape) == 3:  # [H, W, 3]
                # Convert tensor to PIL Image
                frame = data.cpu().numpy()
                if frame.max() == 0.0:  # Skip zero frames
                    return None
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                frame_pil = Image.fromarray(frame).convert('RGB')

                # Process as single image
                return self.__call__(frame_pil)

            elif len(shape) == 4:  # [cameras, H, W, 3]
                processed_frames = []
                for i in range(shape[0]):
                    frame = data[i]
                    if frame.max() == 0.0:  # Skip zero frames
                        continue
                    # Convert to PIL Image
                    frame_np = frame.cpu().numpy()
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    else:
                        frame_np = frame_np.astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np).convert('RGB')

                    # Process single frame
                    proc = self.__call__(frame_pil)
                    if proc is not None:
                        processed_frames.append(proc)

                return torch.stack(processed_frames, dim=0) if processed_frames else None

            else:
                raise ValueError(f"Unexpected input shape for frames: {shape}")

        else:
            raise ValueError(f"Unsupported input type: {type(data)}")