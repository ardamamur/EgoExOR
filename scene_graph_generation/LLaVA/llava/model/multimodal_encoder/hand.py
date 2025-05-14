import torch
import torch.nn as nn
import torch.nn.init as init
class HandEncoder(nn.Module):
    """
    MLP to encode hand tracking data (16-dimensional features).
    """
    def __init__(self, input_dim=16, output_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
        # Initialize weights
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, hand_tracking):
        """
        Args:
            hand_tracking: [batch_size*num_cameras, 16]
        Returns:
            hand_features: [batch_size*num_cameras, output_dim]
        """
        return self.mlp(hand_tracking)