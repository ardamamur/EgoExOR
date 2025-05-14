"""
Heatmap Generator for GazeOR models.

This module defines the HeatmapGenerator class which converts gaze/hand coordinates
to heatmaps using Gaussian blur.
"""

import torch
import torch.nn as nn


class HeatmapEncoder(nn.Module):
    def __init__(self, height=336, width=336, sigma=10.0, use_gaze: bool = True, use_hand: bool = True):
        super().__init__()
        self.height = height
        self.width = width
        self.sigma = sigma / max(height, width)  # Scale sigma to normalized space
        # Precompute normalized grid (0-1)
        x = torch.arange(width).float() / (width - 1)
        y = torch.arange(height).float() / (height - 1)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='xy')
        self.grid_x = nn.Parameter(self.grid_x.contiguous().clone(), requires_grad=False)
        self.grid_y = nn.Parameter(self.grid_y.contiguous().clone(), requires_grad=False)
        self.use_gaze = use_gaze
        self.use_hand = use_hand

    def _generate_gaussian_heatmap(self, coords, batch_size, seq_len, num_cameras):
        heatmaps = torch.zeros(batch_size, seq_len, num_cameras, self.height, self.width,
                              device=coords.device)
        for b in range(batch_size):
            for s in range(seq_len):
                for c in range(num_cameras):
                    if coords[b, s, c].sum() > 0:
                        x, y = coords[b, s, c, 0], coords[b, s, c, 1]
                        # Assume coords are already 0-1, no clamping needed
                        gauss = torch.exp(-((self.grid_x - x) ** 2 + (self.grid_y - y) ** 2) /
                                        (2 * self.sigma ** 2))
                        heatmaps[b, s, c] = gauss / (gauss.sum() + 1e-6)
        return heatmaps

    def forward(self, gaze_coords: torch.Tensor = None, hand_coords: torch.Tensor = None):
        batch_size, seq_len, num_cameras = gaze_coords.shape[:3]
        if self.use_gaze:
            gaze_heatmap = self._generate_gaussian_heatmap(gaze_coords, batch_size, seq_len, num_cameras)
        else:
            gaze_heatmap = torch.zeros_like(gaze_coords)
        if self.use_hand:
            hand_heatmap = self._generate_gaussian_heatmap(hand_coords, batch_size, seq_len, num_cameras)
        else:
            hand_heatmap = torch.zeros_like(hand_coords)
        if not self.use_gaze and not self.use_hand:
            raise ValueError("At least one of use_gaze or use_hand must be True")
        
        unified_heatmap = gaze_heatmap + hand_heatmap
        unified_heatmap = unified_heatmap / (unified_heatmap.max(dim=-1, keepdim=True)[0]
                                            .max(dim=-2, keepdim=True)[0] + 1e-6)
        return unified_heatmap