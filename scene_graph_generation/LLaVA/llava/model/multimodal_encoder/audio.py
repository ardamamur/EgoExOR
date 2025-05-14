"""
Audio encoder for GazeOR models.

This module defines the encoder for audio data.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import ClapModel, ClapProcessor

class AudioEncoder(nn.Module):
    """
    Audio encoder that uses CLAP to extract features from audio snippets.
    Input shape: [B, Sample_rate, stereo_channel] (e.g., [8, 48000, 2])
    Output shape: [B, N_Token, d_model] (e.g., [8, 1, 256]) --> each sample represented by a single token. 
    """
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
        # Depreceated  : self.to_mono = torchaudio.transforms.DownmixMono()  # Convert stereo to mono
        self.projector = nn.Sequential(
            nn.Linear(clap_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio encoding.
        
        Args:
            audio_snippets (torch.Tensor): Input audio snippets of shape [B, S, Sample_rate, stereo_channel]
        """
        # Store original dtype for output conversion
        original_dtype = audio.dtype
        device = audio.device
        
        B, sample_rate, stereo_channel = audio.shape  # e.g., [8, 1, 48000, 2]

        if stereo_channel == 2:
            #audio = self.to_mono(audio.permute(0, 2, 1)).permute(0, 2, 1)  # [8, 48000, 1]
            audio = audio.mean(dim=-1)  # [8, 48000], average across stereo channels
            #audio = audio.squeeze(-1)  # [8, 48000]
        
        # Force audio to float32 for CLAP processing
        audio = audio.to(dtype=torch.float32)

        inputs = self.processor(
            audios=audio.cpu().numpy(),            # Convert to numpy for processor
            return_tensors="pt",                   # Return PyTorch tensors
            sampling_rate=48000                    # Specify sample rate
        )

        # Move inputs to the correct device and ensure float32
        inputs = {k: v.to(device, dtype=torch.float32) for k, v in inputs.items()}
        
        # Extract features using CLAP model (which requires float32)
        with torch.no_grad():  # No gradients for pretrained model
            audio_features = self.clap_model.get_audio_features(**inputs)
            
        # Apply projector (still in float32)
        audio_embed = self.projector(audio_features)
        
        # Convert back to original dtype before returning
        audio_embed = audio_embed.to(dtype=original_dtype)
        
        return audio_embed