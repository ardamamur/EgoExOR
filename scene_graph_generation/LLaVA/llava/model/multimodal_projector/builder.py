import re
import sys
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from llava.model.multimodal_encoder.gaze import GazeEncoder
from llava.model.multimodal_encoder.hand import HandEncoder
from transformers import BertConfig, BertModel
from llava.model.multimodal_projector.pointtransformerv3 import Point
from llava.model.multimodal_projector.pointtransformerv3 import PointTransformerV3
from torch.cuda import amp

from typing import List, Tuple

# Add the project root to the path to access helpers
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../"))
from scene_graph_generation.helpers.config_utils import ConfigManager

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class ImageEmbeddingPooler(nn.Module):
    """
    Encode non-image egocentric and exocentric features separately.
    Then pool them to a fixed size by using the branch pooler separately.
    Concatenate and return as image_features for further projection into the LLM embedding space.
    """
    def __init__(self):
        super().__init__()
        config_manager = ConfigManager()
        self.config = config_manager.config
        self.embedding_dim = 1024
        self.num_output_tokens = self.config.num_output_tokens  # Fixed length output from the branch pooler
        self.num_layers = self.config.num_layers  
        self.num_heads = self.config.nhead  
        self.projection_dim = self.config.projection_dim  # 2048
        self.batch_size = self.config.batch_size  # 2
        self.dataset_name = self.config.dataset_name  # 'egoexor'

        if self.dataset_name == 'egoexor':
            self.gaze_encoder = GazeEncoder(input_dim=3, output_dim=self.embedding_dim)  # 2 for gaze (x, y) + 1 for depth
            self.hand_encoder = HandEncoder(input_dim=16, output_dim=self.embedding_dim)
        
        if self.dataset_name != '4dor':
            # used for processing point clouds
            self.point_transformer = PointTransformerV3(
                cls_mode=True,
                project_pc_dim=1024
            )
            self.point_pooling = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling to reduce to 1xD

            self.audio_projector = nn.Linear(512, 1024)
            self.point_cloud_projector = None

        max_position_embeddings = 576 * 9 # 7 max images, 1 audio, 1 point cloud

        # Branch poolers for egocentric and exocentric features
        bert_config = BertConfig(
            hidden_size=self.embedding_dim,
            num_hidden_layers=3,
            num_attention_heads=self.num_heads,
            intermediate_size=self.embedding_dim * 4,
            use_positional_embeddings=True,
            max_position_embeddings=max_position_embeddings,
            use_bfloat16=True,
            vocab_size=1,
        )
        self.bert = BertModel(bert_config)

    def pad_embeddings(self, embeddings, padding_value=0):
        """
        Pad the embeddings to have the same number in each batch.

        Args:
            embeddings (List[Tensor]): List of embedding tensors, each with shape (num_images, embedding_dim).
            padding_value (float): Value to use for padding.

        Returns:
            Tensor: Padded embeddings with shape (batch_size, max_num_images, embedding_dim).
            Tensor: Mask indicating real data (1) and padding (0) with shape (batch_size, max_num_images).
        """
        batch_size = len(embeddings)
        max_num_images = max(emb.shape[0] for emb in embeddings)
        embedding_dim = embeddings[0].shape[-1] 

        # Initialize padded tensor and mask
        padded_embeddings = torch.full(
            (batch_size, max_num_images, embedding_dim),
            padding_value,
            dtype=embeddings[0].dtype,
            device=embeddings[0].device
        )
        mask = torch.zeros(batch_size, max_num_images, dtype=torch.bool, device=embeddings[0].device)

        # Copy each embedding into the padded tensor and set mask
        for i, emb in enumerate(embeddings):
            num_images = emb.shape[0]
            padded_embeddings[i, :num_images] = emb
            mask[i, :num_images] = 1

        return padded_embeddings, mask
    
    def _encode_pc(self, point_clouds):
        device = torch.device('cuda')
        self.point_transformer.float()
        real_batch_size = len(point_clouds)
        pc_feats = torch.zeros((real_batch_size, 512), dtype=torch.float, device=device)
        
        
        batch = []
        all_coords = []
        all_feats = []
        valid_batch_idx = 0
        
        # Validate and process input point clouds
        for i in range(real_batch_size):
            point_cloud = point_clouds[i]
            if point_cloud is None:
                continue
            if point_cloud.shape[0] == 0:
                continue
            point_cloud = point_cloud.float().to(device)
            num_points = point_cloud.shape[0]
            assert not torch.isnan(point_cloud).any(), f"NaN in point_cloud[{i}]"
            assert not torch.isinf(point_cloud).any(), f"Inf in point_cloud[{i}]"
            batch.append(torch.full((num_points,), valid_batch_idx, dtype=torch.long, device=device))
            valid_batch_idx += 1
            all_coords.append(point_cloud[:, :3])  # xyz coordinates
            all_feats.append(point_cloud)  # xyzrgb features
        
        if len(batch) == 0:
            return torch.zeros((real_batch_size, 1, 1024), dtype=torch.float, device=device)
        
        # Concatenate and validate
        batch = torch.cat(batch, dim=0)
        all_coords = torch.cat(all_coords, dim=0)
        all_feats = torch.cat(all_feats, dim=0)
        assert not torch.isnan(all_coords).any(), "NaN in all_coords"
        assert not torch.isnan(all_feats).any(), "NaN in all_feats"
        assert not torch.isinf(all_coords).any(), "Inf in all_coords"
        assert not torch.isinf(all_feats).any(), "Inf in all_feats"
        
        # Create Point object
        point_data = Point(
            coord=all_coords,
            feat=all_feats,
            grid_size=torch.tensor(0.01, dtype=torch.float, device=device),
            batch=batch
        )
        
        # Process through PointTransformerV3
        with torch.no_grad():  # Ensure no gradients for debugging
            point_data = self.point_transformer(point_data)
        feat = point_data.feat
        assert not torch.isnan(feat).any(), "NaN in point_transformer feat"
        assert not torch.isinf(feat).any(), "Inf in point_transformer feat"
        
        # Pool features for each point cloud
        valid_batch_idx = 0
        for i in range(real_batch_size):
            point_cloud = point_clouds[i]
            if point_cloud is None or point_cloud.shape[0] == 0:
                continue
            mask = point_data['batch'] == valid_batch_idx
            feat_mask = feat[mask]
            assert feat_mask.shape[0] > 0, f"Empty feat_mask for point_cloud[{i}]"
            assert not torch.isnan(feat_mask).any(), f"NaN in feat_mask[{i}]"
            pooled_feat = self.point_pooling(feat_mask.unsqueeze(0).permute(0, 2, 1)).squeeze(-1)
            assert not torch.isnan(pooled_feat).any(), f"NaN in pooled_feat[{i}]"
            pc_feats[i] = pooled_feat
            valid_batch_idx += 1
        
        # Project features
        assert not torch.isnan(pc_feats).any(), "NaN in pc_feats before project_pc"
        pc_feats = self.point_transformer.project_pc(pc_feats.float())
        assert not torch.isnan(pc_feats).any(), "NaN in pc_feats after project_pc"
        
        # Prepare final output
        pc_features = pc_feats.unsqueeze(1)
        pc_features = pc_features.to(torch.bfloat16)
        assert not torch.isnan(pc_features).any(), "NaN in pc_features after bfloat16"
        
        return pc_features


    def forward(self, embeddings, mask, 
                gaze=None, gaze_depth=None, 
                hand_tracking=None, 
                audio=None,
                point_cloud=None,
                split_sizes=None
                ):
        """
        Args:
            embeddings: [batch_size, num_cameras*num_tokens, embed_dim] (e.g. [4, 2304, 1024])
            mask: [batch_size, num_cameras*num_tokens]
            gaze: [batch_size, num_cameras, 2] or [batch_size*num_cameras, 2]
            gaze_depth: [batch_size, num_cameras] or [batch_size*num_cameras]
            hand_tracking: [batch_size, num_cameras, 16] or [batch_size*num_cameras, 16]
            point_cloud: List : [num_points, point_cloud_dim]
            audio: [batch_size, audio_dim, 2]
        Returns:
            pooled_features: [batch_size, num_output_tokens, embed_dim]
        """
        batch_size, total_tokens, embed_dim = embeddings.shape

        # Process gaze and gaze depth together
        gaze_features = None
        if gaze is not None and gaze_depth is not None:
            if gaze.dim() == 3:  # [batch_size, num_cameras, 2]
                gaze = gaze.view(batch_size * gaze.shape[1], 2)
                gaze_depth = gaze_depth.view(batch_size * gaze_depth.shape[1], 1)
            # Combine gaze and gaze depth
            gaze_input = torch.cat([gaze, gaze_depth], dim=-1)  # [batch_size*num_cameras, 3]
            gaze_features = self.gaze_encoder(gaze_input)
            gaze_features = torch.split(gaze_features, split_sizes, dim=0)
            gaze_features, gaze_mask = self.pad_embeddings(gaze_features)

        # Process hand tracking
        hand_features = None
        if hand_tracking is not None:
            if hand_tracking.dim() == 3:  # [batch_size, num_cameras, 16]
                hand_tracking = hand_tracking.view(batch_size * hand_tracking.shape[1], 16)
            hand_features = self.hand_encoder(hand_tracking)
            hand_features = torch.split(hand_features, split_sizes, dim=0)
            hand_features, hand_mask = self.pad_embeddings(hand_features)

        # Process audio
        audio_features = None
        if audio is not None and self.audio_projector is not None:
            audio_embeds = torch.zeros(batch_size, 1, 512, dtype=torch.bfloat16, device=torch.device('cuda'))
            for i, audio_snippet in enumerate(audio):
                if audio_snippet is not None:
                    audio_embeds[i] = audio_snippet.to(torch.bfloat16)
            audio_features = self.audio_projector(audio_embeds)
            assert not torch.isnan(audio_features).any(), "NaN in audio_features"


        # Process point cloud
        pc_features = None
        if point_cloud is not None and self.point_transformer is not None:
            with amp.autocast(enabled=False):
                pc_features = self._encode_pc(point_cloud) if point_cloud is not None else None

        if pc_features is not None:
            pc_features = pc_features.to(embeddings.dtype) 


        # Combine additional features
        extra_features = []
        if gaze_features is not None: extra_features.append(gaze_features)
        if hand_features is not None: extra_features.append(hand_features)
        if audio_features is not None: extra_features.append(audio_features)
        if pc_features is not None: extra_features.append(pc_features)
        
        if extra_features:
            extra_features = torch.cat(extra_features, dim=1)  # [batch_size, num_extra_tokens, embed_dim]
            embeddings = torch.cat([embeddings, extra_features], dim=1)  # [batch_size, total_tokens + num_extra_tokens, embed_dim]
            extra_mask = torch.ones(batch_size, extra_features.shape[1], device=mask.device, dtype=mask.dtype)
            mask = torch.cat([mask, extra_mask], dim=1)

        bert_output = self.bert(inputs_embeds=embeddings, attention_mask=mask, return_dict=True)
        bert_output = bert_output['last_hidden_state'].to(embeddings.dtype)[:, :576]
        return bert_output



def build_image_pooler(config):
    return ImageEmbeddingPooler()