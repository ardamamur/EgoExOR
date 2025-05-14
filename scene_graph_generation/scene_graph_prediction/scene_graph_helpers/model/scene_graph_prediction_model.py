# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adopted from https://github.com/egeozsoy/MM-OR/scene_graph_prediction/scene_graph_helpers/model/scene_graph_prediction_model.py
import json
import re
from collections import defaultdict
from copy import deepcopy

import h5py
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm
from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN
from LLaVA.llava.conversation import SeparatorStyle, default_conversation
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model, load_pretrained_model_cpu

from .input_transformation import (
    FrameTransform, GazeNormalize, GazeDepthNormalize, 
    HandTrackingNormalize, AudioTransform, AudioProcessor
)

from ..dataset.dataset_utils import (
    reversed_sources, reversed_relation_synonyms, reversed_entity_synonyms,
    map_vocab_idx_to_scene_graph_name, map_scene_graph_name_to_vocab_idx,
    scene_graph_name_to_vocab_idx,
    GAZE_FIXATION, SOURCES
)
from ..dataset.or_dataset import _needs_fixation
from typing import Dict, Optional, Sequence, List, Tuple, Any


from collections.abc import Mapping, Sequence

def to_device(obj, device):
    """Recursively move **all** tensors in *obj* to *device*."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    # Dict  ────────────────────────────────────────────────
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            obj[k] = to_device(v, device)
        return obj                      # keep the same container

    # List / tuple  ────────────────────────────────────────
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(to_device(v, device) for v in obj)

    # Anything else (ints, strings, None…)
    return obj



class ModelWrapper:
    def __init__(self, hdf5_path, dataset_name, relationNames, classNames, model_path, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False, load_4bit=False, temporality=None, mv_type="learned", device="cuda", device_map="auto"):
        self.hdf5_path = hdf5_path
        self.n_object_types = 6
        self.relationNames = relationNames
        self.classNames = classNames
        self.relation_names_lower_case = [relation.lower() for relation in self.relationNames]
        self.lr = 0.00003
        # evaluation metrics
        self.train_take_rel_preds = defaultdict(list)
        self.train_take_rel_gts = defaultdict(list)
        self.val_take_rel_preds = defaultdict(list)
        self.val_take_rel_gts = defaultdict(list)

        self.train_take_rel_binary_interaction_preds = defaultdict(list)
        self.train_take_rel_binary_interaction_gts = defaultdict(list)
        self.val_take_rel_binary_interaction_preds = defaultdict(list)
        self.val_take_rel_binary_interaction_gts = defaultdict(list)

        self.train_take_entity_preds = defaultdict(list)
        self.train_take_entity_gts = defaultdict(list)
        self.val_take_entity_preds = defaultdict(list)
        self.val_take_entity_gts = defaultdict(list)

        self.reset_metrics()

        self.model_name = get_model_name_from_path(model_path)
        if device == "cpu":
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model_cpu(model_path, model_base, 
                                                                                            self.model_name, 
                                                                                            load_8bit, load_4bit,
                                                                                            device=device,
                                                                                            device_map=torch.device("cpu"))
        else:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, 
                                                                                                    self.model_name, 
                                                                                                    load_8bit, load_4bit,
                                                                                                    device_map=device_map)
        self.model.config.mv_type = mv_type
        self.model.config.tokenizer_padding_side = "left"
        self.temporal_online_prediction = False
        if temporality is not None and temporality == "PRED":
            print('Preparing temporality PRED')
            self.take_to_history = defaultdict(list)
            self.temporal_online_prediction = True


        self.frame_transform = FrameTransform(self.image_processor)
        self.gaze_normalize = GazeNormalize(img_width=336, img_height=336)
        self.depth_normalize = GazeDepthNormalize(max_depth=1.0)
        self.hand_normalize = HandTrackingNormalize(img_width=336, img_height=336)
        self.audio_normalize = AudioTransform()
        # since we used frozen audio processor we dont need to get it from pretrained egoexor mode
        self.audio_processor = AudioProcessor(model_name="laion/larger_clap_general", d_model=1024, clap_hidden_size=512)

        self.dataset_name = dataset_name

        self.is_egoexor = True if self.dataset_name == "egoexor" else False
        self.is_4dor = True if self.dataset_name == "4dor" else False
        self.is_mmor = True if self.dataset_name == "mmor" else False

    def load_and_process_image(self, img_tensor):
        img_np = img_tensor.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        img_np = img_np[..., ::-1]  # Convert RGB to BGR
        img_pil = Image.fromarray(img_np).convert('RGB')
        return self.frame_transform(img_pil)
    

    def forward(self, batch):
        batch_size = len(batch["sample"])
        outputs = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for batch_idx in range(batch_size):
                metadata = batch["sample"][batch_idx]["hdf5_indices"]
                available_modalities = metadata["available_modalities"]
                ego_source_names = batch["ego_source_names"][batch_idx]
                exo_source_names = batch["exo_source_names"][batch_idx]
                ego_source_ids = batch["ego_source_ids"][batch_idx]
                exo_source_ids = batch["exo_source_ids"][batch_idx]

                path = f"data/{metadata['surgery_type']}/{metadata['procedure_id']}/take/{metadata['take_id']}"
                frame_idx = metadata["frame_idx"]
                frame_rgb = torch.from_numpy(f[f'{path}/frames/rgb'][frame_idx]).float()

                # --- Ego & Exo Image Processing ---
                ego_images, exo_images = [], []
                for source_img_id in ego_source_ids:
                    processed = self.load_and_process_image(frame_rgb[source_img_id])
                    if processed is not None:
                        ego_images.append(processed)

                for source_img_id in exo_source_ids:
                    processed = self.load_and_process_image(frame_rgb[source_img_id])
                    if processed is not None:
                        exo_images.append(processed)

                # --- Modalities ---
                modality_data = {}

                # Eye Gaze
                if 'eye_gaze' in available_modalities:
                    gaze_key = f'{path}/eye_gaze/coordinates'
                    if gaze_key in f:
                        raw = f[gaze_key][frame_idx]
                        for i in range(raw.shape[0]):
                            cam_id = int(raw[i, 0])
                            role = reversed_sources.get(cam_id)
                            if role and _needs_fixation(role, path):
                                raw[i, 1] += GAZE_FIXATION["x"]
                                raw[i, 2] += GAZE_FIXATION["y"]

                        coords = torch.from_numpy(raw[:, 1:3]).float()
                        coords = self.gaze_normalize(coords).to(dtype=torch.bfloat16)
                        camera_ids = torch.from_numpy(raw[:, 0]).long()
                        ego_source_ids_mapped = [SOURCES[name] for name in ego_source_names]

                        valid_indices = [idx for idx, cam_id in enumerate(camera_ids) if cam_id.item() in ego_source_ids_mapped]
                        valid_coords = coords[valid_indices]
                        valid_camera_ids = camera_ids[valid_indices]

                        ordered_indices = []
                        for source_id in ego_source_ids_mapped:
                            for idx, cam_id in enumerate(valid_camera_ids):
                                if cam_id.item() == source_id:
                                    ordered_indices.append(idx)
                                    break

                        modality_data['eye_gaze'] = {
                            'data': valid_coords[ordered_indices] if ordered_indices else torch.zeros((0, 2), dtype=torch.bfloat16),
                            'camera_ids': valid_camera_ids[ordered_indices] if ordered_indices else torch.zeros((0,), dtype=torch.long),
                        }

                # Eye Gaze Depth
                if 'eye_gaze_depth' in available_modalities:
                    depth_key = f'{path}/eye_gaze_depth/values'
                    if depth_key in f:
                        raw_d = torch.from_numpy(f[depth_key][frame_idx]).float()
                        raw_d = self.depth_normalize(raw_d).to(dtype=torch.bfloat16)

                        valid_indices = [i for i in range(len(raw_d)) if i + min(ego_source_ids) in ego_source_ids]
                        valid_d = raw_d[valid_indices]

                        ordered_indices = []
                        for idx in valid_indices:
                            source_idx = idx + min(ego_source_ids)
                            if source_idx in ego_source_ids:
                                ordered_indices.append(valid_indices.index(idx))

                        modality_data['eye_gaze_depth'] = {
                            'data': valid_d[ordered_indices] if ordered_indices else torch.zeros((0,), dtype=torch.bfloat16)
                        }

                # Hand Tracking
                if 'hand_tracking' in available_modalities:
                    hand_key = f'{path}/hand_tracking/positions'
                    if hand_key in f:
                        raw_h = torch.from_numpy(f[hand_key][frame_idx][:, 1:]).float()
                        mask = torch.isnan(raw_h).any(dim=-1)
                        raw_h = torch.nan_to_num(raw_h, nan=0.0)
                        raw_h = self.hand_normalize(raw_h).to(dtype=torch.bfloat16)
                        camera_ids = torch.arange(raw_h.shape[0]).long()

                        valid_indices = [idx for idx in range(raw_h.shape[0]) if idx + min(ego_source_ids) in ego_source_ids]
                        valid_h = raw_h[valid_indices]
                        valid_mask = mask[valid_indices]
                        valid_camera_ids = camera_ids[valid_indices]

                        ordered_indices = []
                        for name in ego_source_names:
                            target_id = ego_source_ids[ego_source_names.index(name)]
                            for idx, cam_idx in enumerate(valid_camera_ids):
                                if cam_idx + min(ego_source_ids) == target_id:
                                    ordered_indices.append(idx)
                                    break

                        modality_data['hand_tracking'] = {
                            'data': valid_h[ordered_indices] if ordered_indices else torch.zeros((0, raw_h.shape[1]), dtype=torch.bfloat16),
                            'mask': valid_mask[ordered_indices] if ordered_indices else torch.zeros((0,), dtype=torch.bool),
                            'camera_ids': valid_camera_ids[ordered_indices] if ordered_indices else torch.zeros((0,), dtype=torch.long),
                        }

                # Point Cloud
                if 'point_cloud' in available_modalities:
                    points_key = f'{path}/point_cloud/coordinates'
                    colors_key = f'{path}/point_cloud/colors'
                    if points_key in f and colors_key in f:
                        coords = np.asarray(f[points_key][frame_idx])
                        colors = np.asarray(f[colors_key][frame_idx])
                        pts6 = np.concatenate([coords, colors], axis=1)
                        modality_data['point_cloud'] = {
                            'data': torch.from_numpy(pts6).float()
                        }

                # Audio
                if 'audio' in available_modalities:
                    audio_key = f'{path}/audio/snippets'
                    if audio_key in f:
                        raw_a = torch.from_numpy(f[audio_key][frame_idx]).float()
                        raw_a = self.audio_normalize(raw_a).to(dtype=torch.bfloat16)
                        raw_a = self.audio_processor(raw_a.unsqueeze(0))  # Add batch dim
                        modality_data['audio'] = {'data': raw_a}

                # Conversations (LLM input)
                conv = deepcopy(default_conversation)
                convo = batch["sample"][batch_idx]["conversations"]
                conv.append_message(convo[0]["from"], convo[0]["value"])
                conv.append_message(convo[1]["from"], None)
                prompt = conv.get_prompt()

                data_dict = {
                    "prompt" : prompt,
                }

                if not self.is_egoexor:
                    combined_exo_images = ego_images + exo_images
                    combined_exo_source_ids = ego_source_ids + exo_source_ids
                    combined_exo_source_names = ego_source_names + exo_source_names
                    
                    # For evaluation: Select up to 7 images deterministically
                    # - Up to 3 egocentric images (prioritizing assistant, head_surgeon, anesthetist if all 4 exist)
                    # - Up to 3 exocentric images (prioritizing external_1, external_3, external_5 if available)
                    # - 1 ultrasound image (if available)
                    max_images = min(7, len(combined_exo_images))
                    kept_indices = []

                    # Step 1: Select up to 3 egocentric images
                    ego_indices = []
                    for i, source_names in enumerate(combined_exo_source_names):
                        if any(name in ['assistant', 'head_surgeon', 'circulator', 'anesthetist'] for name in source_names):
                            ego_indices.append(i)

                    # Check if all 4 egocentric sources exist
                    ego_source_set = set()
                    for i in ego_indices:
                        ego_source_set.update(combined_exo_source_names[i])
                    all_ego_present = all(name in ego_source_set for name in ['assistant', 'head_surgeon', 'circulator', 'anesthetist'])

                    if all_ego_present:
                        # Keep assistant, head_surgeon, anesthetist
                        for i in ego_indices:
                            if any(name in ['assistant', 'head_surgeon', 'anesthetist'] for name in combined_exo_source_names[i]):
                                kept_indices.append(i)
                    else:
                        # Keep up to 3 egocentric images in order
                        ego_count = 0
                        for i in ego_indices:
                            if ego_count < 3:
                                kept_indices.append(i)
                                ego_count += 1

                    if len(kept_indices) >= max_images:
                        kept_indices = kept_indices[:max_images]

                    # Step 2: Select up to 3 exocentric images (non-ultrasound)
                    exo_indices = []
                    for i, source_names in enumerate(combined_exo_source_names):
                        if i not in kept_indices and 'ultrasound' not in source_names:
                            exo_indices.append(i)

                    # Check for external_[1 to 5]
                    external_indices = []
                    for i in exo_indices:
                        if any(name in ['external_1', 'external_2', 'external_3', 'external_4', 'external_5'] for name in combined_exo_source_names[i]):
                            external_indices.append(i)

                    if external_indices:
                        # Prioritize external_1, external_3, external_5
                        priority_exo = []
                        for i in external_indices:
                            if any(name in ['external_1', 'external_3', 'external_5'] for name in combined_exo_source_names[i]):
                                priority_exo.append(i)
                        # Add up to 3 prioritized external indices
                        for i in priority_exo:
                            if len(kept_indices) < max_images:
                                kept_indices.append(i)
                    else:
                        # Add up to 3 exocentric images in order
                        exo_count = 0
                        for i in exo_indices:
                            if exo_count < 3 and len(kept_indices) < max_images:
                                kept_indices.append(i)
                                exo_count += 1

                    if len(kept_indices) >= max_images:
                        kept_indices = kept_indices[:max_images]

                    # Step 3: Select 1 ultrasound image (if available)
                    for i, source_names in enumerate(combined_exo_source_names):
                        if 'ultrasound' in source_names and i not in kept_indices:
                            if len(kept_indices) < max_images:
                                kept_indices.append(i)
                            break

                    # Sort indices to maintain order
                    kept_indices = sorted(kept_indices)

                    # Update the combined lists to only include selected indices
                    combined_exo_images = [combined_exo_images[i] for i in kept_indices]
                    combined_exo_source_names = [combined_exo_source_names[i] for i in kept_indices]
                    combined_exo_source_ids = [combined_exo_source_ids[i] for i in kept_indices]



                else:
                    combined_exo_images = exo_images
                    combined_exo_source_ids = exo_source_ids
                    combined_exo_source_names = exo_source_names

                if combined_exo_images:
                    data_dict['exo_frames'] = torch.stack(combined_exo_images)
                    data_dict['exo_source_ids'] = combined_exo_source_ids
                    data_dict['exo_source_names'] = combined_exo_source_names

                # Only populate ego frames when egoexor
                if self.is_egoexor and ego_images:
                    data_dict['ego_frames']       = torch.stack(ego_images)
                    data_dict['ego_source_ids']   = ego_source_ids
                    data_dict['ego_source_names'] = ego_source_names

                # Fallback zeros when truly no images and multimodal
                if not ego_images and not exo_images:
                    zeros = torch.zeros(1, 3, 336, 336)
                    data_dict['exo_frames']       = zeros
                    data_dict['ego_frames']       = zeros if self.is_egoexor else None
                    data_dict['exo_source_ids']   = [0]
                    data_dict['exo_source_names'] = []

                    if self.is_egoexor:
                        data_dict['ego_source_ids']   = [0]
                        data_dict['ego_source_names'] = []

                data_dict.update(modality_data)
                outputs.append(data_dict)


        # at the and batch should have the same sturcture as before but with added new data
        # === Build final batch dictionary ===
        final_batch = {}

        # 1. Extract prompts
        all_prompts = [x["prompt"] for x in outputs]
        
        # 2. Tokenize prompts with image tokens
        if batch_size == 1:
            input_ids = tokenizer_image_token(
                all_prompts[0],
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(self.model.device)
        else:
            input_ids = [
                tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
                for prompt in all_prompts
            ]
            # Apply left-padding manually
            inverted = [torch.flip(ids, dims=[0]) for ids in input_ids]
            padded = torch.nn.utils.rnn.pad_sequence(inverted, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            input_ids = torch.flip(padded, dims=[1]).to(self.model.device)

        final_batch["input_ids"] = input_ids

        # 4. Collect all non-text fields
        def collect(key):
            vals = [x[key] for x in outputs if key in x]
            if all(isinstance(v, torch.Tensor) for v in vals):
                # if shapes mismatch, just return the list of tensors
                try:
                    return torch.stack(vals)
                except RuntimeError:
                    return vals
            return vals


        for key in ["ego_frames", "exo_frames", "exo_source_ids", "ego_source_ids", "exo_source_names", "ego_source_names"]:
            if any(key in x for x in outputs):
                final_batch[key] = collect(key)


        # 5. Multimodal modality dicts (can be nested or flattened)
        for modality in ["eye_gaze", "eye_gaze_depth", "hand_tracking", "point_cloud", "audio"]:
            if any(modality in x for x in outputs):
                final_batch[modality] = [x.get(modality) for x in outputs]


        if self.is_4dor:
            # keep only text + ego + exo
            allowed = {
                "input_ids",
                "labels",
                "attention_mask",
                "ego_frames",
                "ego_source_names",
                "ego_source_ids",
                "exo_frames",
                "exo_source_names",
                "exo_source_ids",
            }
            final_batch = {k: v for k, v in final_batch.items() if k in allowed}

        if self.is_mmor:
            # drop any gaze / hand‐tracking
            for mod in ("eye_gaze", "eye_gaze_depth", "hand_tracking"):
                final_batch.pop(mod, None)
            # and ensure ego is not present (we merged it into exo in __getitem__)
            for key in ("ego_frames", "ego_source_names", "ego_source_ids"):
                final_batch.pop(key, None)

        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)


        forward_kwargs = {
            "input_ids": final_batch["input_ids"],
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": 300,
            "stopping_criteria": [stopping_criteria]
        }

        # List all optional modalities and their arg‐names
        optional_mods = {
            "ego_frames":       "ego_frames",
            "exo_frames":       "exo_frames",
            "eye_gaze":         "eye_gaze",
            "eye_gaze_depth":   "eye_gaze_depth",
            "hand_tracking":    "hand_tracking",
            "ego_source_ids":   "ego_source_ids",
            "exo_source_ids":   "exo_source_ids",
            "audio":            "audio",
            "point_cloud":      "point_cloud",
        }

        # Add only those that are present (and non-None)
        for batch_key, model_arg in optional_mods.items():
            if batch_key in final_batch and final_batch[batch_key] is not None:
                forward_kwargs[model_arg] = final_batch[batch_key]

        # Grab your model’s device
        device = next(self.model.parameters()).device  # e.g. 'cuda:0' or 'cpu'

        # Move everything to the same device
        forward_kwargs = to_device(forward_kwargs, device)

        for k, v in forward_kwargs.items():
            if torch.is_tensor(v):
                assert v.device == device, f"{k} still on {v.device}"
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        assert vv.device == device, f"{k}[{kk}] still on {vv.device}"
            elif isinstance(v, Sequence):
                for i, vv in enumerate(v):
                    if torch.is_tensor(vv):
                        assert vv.device == device, f"{k}[{i}] still on {vv.device}"



        with torch.inference_mode():
            output_ids = self.model.generate(**forward_kwargs)

        if batch_size == 1:
            outputs = [
                self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:]
                ).strip()
            ]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:].tolist(),
                skip_special_tokens=True
            )

        return outputs


    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)

            self.train_take_rel_binary_interaction_preds = defaultdict(list)
            self.train_take_rel_binary_interaction_gts = defaultdict(list)

            self.train_take_entity_preds = defaultdict(list)
            self.train_take_entity_gts = defaultdict(list)
        elif split == 'val':
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

            self.val_take_rel_binary_interaction_preds = defaultdict(list)
            self.val_take_rel_binary_interaction_gts = defaultdict(list)

            self.val_take_entity_preds = defaultdict(list)
            self.val_take_entity_gts = defaultdict(list)
        else:
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

            self.train_take_rel_binary_interaction_preds = defaultdict(list)
            self.train_take_rel_binary_interaction_gts = defaultdict(list)
            self.val_take_rel_binary_interaction_preds = defaultdict(list)
            self.val_take_rel_binary_interaction_gts = defaultdict(list)

            self.train_take_entity_preds = defaultdict(list)
            self.train_take_entity_gts = defaultdict(list)
            self.val_take_entity_preds = defaultdict(list)
            self.val_take_entity_gts = defaultdict(list)

    def infer(self, dataloader):
        return self.validate(dataloader, return_raw_predictions=True)
    
    def validate(self, dataloader, limit_val_batches=None, logging_information=None, return_raw_predictions=False):
        take_rel_preds = defaultdict(list)
        take_rel_gts = defaultdict(list)
        take_rel_binary_interaction_preds = defaultdict(list)
        take_rel_binary_interaction_gts = defaultdict(list)
        take_entity_preds = defaultdict(list)
        take_entity_gts = defaultdict(list)

        sample_id_to_raw_predictions = {}  # dictionary to store predicted scene graphs
        limit_counter = None
        if isinstance(limit_val_batches, int):
            limit_counter = limit_val_batches
        elif isinstance(limit_val_batches, float):
            limit_counter = int(limit_val_batches * len(dataloader))

        for batch in tqdm(dataloader):
            if limit_counter is not None:
                if limit_counter <= 0:
                    break
                limit_counter -= 1

            assert len(batch) == 1 or not self.temporal_online_prediction

            outputs = self.forward(batch)
            for idx, output in enumerate(outputs):
                sample = batch["sample"][idx]
                gt_scene_graph = sample["conversations"][1]["value"]
                timepoint = int(sample["timepoint"])
                surgery_type = sample["hdf5_indices"]["surgery_type"]
                procedure_id = sample["hdf5_indices"]["procedure_id"]
                take_id = sample["hdf5_indices"]["take_id"]
                take_name  = f"{surgery_type}_{procedure_id}_{take_id}"

                # --- Prediction parding ----
                triplets = []
                raw_triplets = []
                # remove everything between the first """ and the last """ using regex. This is used for chain of thought
                output = re.sub(r'""".*?"""', '', output, flags=re.DOTALL)
                if '<SG>' in output and '</SG>' in output and output.index('<SG>') < output.index('</SG>'):
                    triplet_str = output.split('<SG>')[1].split('</SG>')[0].strip().split(';')
                else:
                    triplet_str = output.split(';')

                for triplet in triplet_str:
                    triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
                    if triplet == '':
                        continue
                    triplet = triplet.split(',')
                    triplet = [elem.strip() for elem in triplet]
                    if len(triplet) != 3:
                        continue
                    sub, obj, pred = triplet
                    raw_triplets.append((sub, pred, obj))
                    if sub in reversed_entity_synonyms:
                        sub = reversed_entity_synonyms[sub]
                    if obj in reversed_entity_synonyms:
                        obj = reversed_entity_synonyms[obj]
                    triplets.append((sub, pred, obj))
                # these have to be mapped. First to human names, also the predicates
                sample_id_to_raw_predictions[sample['id']] = raw_triplets
                if self.temporal_online_prediction:
                    self.take_to_history[take_name].append({'timepoint_idx': timepoint, 'scene_graph': raw_triplets})
                rel_preds = []
                for (sub, pred, obj) in triplets:
                    try:
                        sub = map_scene_graph_name_to_vocab_idx(sub.replace(' ', '_'))
                        obj = map_scene_graph_name_to_vocab_idx(obj.replace(' ', '_'))
                        pred = map_scene_graph_name_to_vocab_idx(pred)
                        rel_preds.append((sub, pred, obj))
                    except Exception as e:
                        print(e)
                        continue
                
                # --- Ground Truth parsing (same logic) ---
                gt_triplets = []
                raw_gt_triplets = []

                gt_scene_graph = re.sub(r'""".*?"""', '', gt_scene_graph, flags=re.DOTALL)
                if '<SG>' in gt_scene_graph and '</SG>' in gt_scene_graph and gt_scene_graph.index('<SG>') < gt_scene_graph.index('</SG>'):
                    gt_triplet_str = gt_scene_graph.split('<SG>')[1].split('</SG>')[0].strip().split(';')
                else:
                    gt_triplet_str = gt_scene_graph.split(';')

                for triplet in gt_triplet_str:
                    triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
                    if not triplet:
                        continue
                    triplet = [elem.strip() for elem in triplet.split(',')]
                    if len(triplet) != 3:
                        continue
                    sub, obj, pred = triplet
                    raw_gt_triplets.append((sub, pred, obj))
                    if sub in reversed_entity_synonyms:
                        sub = reversed_entity_synonyms[sub]
                    if obj in reversed_entity_synonyms:
                        obj = reversed_entity_synonyms[obj]
                    if pred in reversed_relation_synonyms:
                        pred = reversed_relation_synonyms[pred]
                    gt_triplets.append((sub, pred, obj))

                rel_labels = []
                for (sub, pred, obj) in gt_triplets:
                    try:
                        sub = map_scene_graph_name_to_vocab_idx(sub.replace(' ', '_'))
                        obj = map_scene_graph_name_to_vocab_idx(obj.replace(' ', '_'))
                        pred = map_scene_graph_name_to_vocab_idx(pred)
                        rel_labels.append((sub, pred, obj))
                    except Exception as e:
                        print(f"[GT Mapping Error] {e}")
                        continue

                if len(rel_labels) == 0:
                    all_gt_objects = []
                else:
                    all_gt_objects = sorted(set([sub for sub, _, _ in rel_preds] + [obj for _, _, obj in rel_preds]))
                
                # Search for all possible relationships between objects, those that don't have any should be labeled 'none', otherwise the correct relation is asked for
                all_pred_objects = sorted(set([sub for sub, _, _ in rel_preds] + [obj for _, _, obj in rel_preds]))

                for gt_obj1 in all_gt_objects:
                    # add this object to ground truth entities
                    take_entity_gts[take_name].append(self.classNames.index(map_vocab_idx_to_scene_graph_name(gt_obj1)))
                    # if this object is part of the predicted entities, add it to the predicted entities
                    if gt_obj1 in all_pred_objects:
                        take_entity_preds[take_name].append(self.classNames.index(map_vocab_idx_to_scene_graph_name(gt_obj1)))
                    else:
                        take_entity_preds[take_name].append(-1)
                    for gt_obj2 in all_gt_objects:
                        if gt_obj1 == gt_obj2:
                            continue
                        for gt_sub, gt_rel, gt_obj in rel_labels:
                            if gt_sub == gt_obj1 and gt_obj == gt_obj2:
                                take_rel_gts[take_name].append(self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(gt_rel)))
                                take_rel_binary_interaction_gts[take_name].append(1)
                                break
                        else:
                            take_rel_gts[take_name].append(self.relation_names_lower_case.index('none'))
                            take_rel_binary_interaction_gts[take_name].append(0)
                        for pred_sub, pred_rel, pred_obj in rel_preds:
                            if pred_sub == gt_obj1 and pred_obj == gt_obj2:
                                try:
                                    pred_rel_id = self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(pred_rel))
                                    take_rel_binary_interaction_preds[take_name].append(1)
                                except Exception as e:  # if a   none sense relation was predicted ignore
                                    pred_rel_id = self.relation_names_lower_case.index('none')
                                    take_rel_binary_interaction_preds[take_name].append(0)
                                take_rel_preds[take_name].append(pred_rel_id)
                                break
                        else:
                            take_rel_preds[take_name].append(self.relation_names_lower_case.index('none'))
                            take_rel_binary_interaction_preds[take_name].append(0)
        
        self.val_take_rel_preds, self.val_take_rel_gts = take_rel_preds, take_rel_gts
        self.val_take_rel_binary_interaction_preds, self.val_take_rel_binary_interaction_gts = take_rel_binary_interaction_preds, take_rel_binary_interaction_gts
        self.val_take_entity_preds, self.val_take_entity_gts = take_entity_preds, take_entity_gts
        self.evaluate_predictions(None, 'val', logging_information=logging_information)
        self.reset_metrics(split='val')

        if return_raw_predictions:
            return sample_id_to_raw_predictions
        

    
    def evaluate_predictions(self, epoch_loss, split, logging_information=None):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds
            take_rel_gts = self.train_take_rel_gts
            take_rel_binary_interaction_preds = self.train_take_rel_binary_interaction_preds
            take_rel_binary_interaction_gts = self.train_take_rel_binary_interaction_gts
            take_entity_preds = self.train_take_entity_preds
            take_entity_gts = self.train_take_entity_gts
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds
            take_rel_gts = self.val_take_rel_gts
            take_rel_binary_interaction_preds = self.val_take_rel_binary_interaction_preds
            take_rel_binary_interaction_gts = self.val_take_rel_binary_interaction_gts
            take_entity_preds = self.val_take_entity_preds
            take_entity_gts = self.val_take_entity_gts
        else:
            raise NotImplementedError()

        all_rel_gts = []
        all_rel_preds = []
        all_rel_binary_interaction_gts = []
        all_rel_binary_interaction_preds = []
        all_entity_gts = []
        all_entity_preds = []
        data_type_rel_preds = defaultdict(list)
        data_type_rel_gts = defaultdict(list)
        data_type_rel_binary_interaction_preds = defaultdict(list)
        data_type_rel_binary_interaction_gts = defaultdict(list)
        data_type_entity_preds = defaultdict(list)
        data_type_entity_gts = defaultdict(list)

        for take_name in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_name]
            rel_gts = take_rel_gts[take_name]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)

            rel_binary_interaction_preds = take_rel_binary_interaction_preds[take_name]
            rel_binary_interaction_gts = take_rel_binary_interaction_gts[take_name]
            all_rel_binary_interaction_gts.extend(rel_binary_interaction_gts)
            all_rel_binary_interaction_preds.extend(rel_binary_interaction_preds)

            entity_preds = take_entity_preds[take_name]
            entity_gts = take_entity_gts[take_name]
            all_entity_gts.extend(entity_gts)
            all_entity_preds.extend(entity_preds)

            # Determine data type based on take_idx or take name
            if 'MISS' in take_name:
                data_type = 'MISS'
            else:
                data_type = 'Ultrasound'

            data_type_rel_preds[data_type].extend(rel_preds)
            data_type_rel_gts[data_type].extend(rel_gts)
            data_type_rel_binary_interaction_preds[data_type].extend(rel_binary_interaction_preds)
            data_type_rel_binary_interaction_gts[data_type].extend(rel_binary_interaction_gts)
            data_type_entity_preds[data_type].extend(entity_preds)
            data_type_entity_gts[data_type].extend(entity_gts)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, output_dict=True, digits=4)  # non existing relations will be counted as True
            real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
            for rel_name in self.relationNames:
                if cls_report[rel_name]['support'] == 0:
                    continue
                for score_type in ['precision', 'recall', 'f1-score']:
                    real_macro_values[score_type].append(cls_report[rel_name][score_type])
                    # self.log(f'{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type], rank_zero_only=True)
                    if logging_information is not None:
                        logging_information['logger'].log_metrics({f'{rel_name}/{take_name}_{score_type[:2].upper()}': cls_report[rel_name][score_type]}, step=logging_information['checkpoint_id'])
            real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, digits=4)  # non existing relations will be counted as True
            

            print(f'\nTake {take_name}\n')
            print(cls_report)
            print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

            # a less granular report for binary_interaction and entity detection
            binary_interaction_cls_report = classification_report(rel_binary_interaction_gts, rel_binary_interaction_preds, labels=[0, 1],
                                                                  target_names=['no_interaction', 'interaction'], output_dict=False, digits=4)
            entity_cls_report = classification_report(entity_gts, entity_preds, labels=list(range(len(self.classNames))),
                                                      target_names=self.classNames, output_dict=False, digits=4)
            print(f'\nBinary Interaction Classification Report for Take {take_name}\n')
            print(binary_interaction_cls_report)
            print(f'\nEntity Classification Report for Take {take_name}\n')
            print(entity_cls_report)

            # Compute and print classification reports per data type
        for data_type in data_type_rel_preds.keys():
            rel_preds = data_type_rel_preds[data_type]
            rel_gts = data_type_rel_gts[data_type]
            rel_binary_interaction_preds = data_type_rel_binary_interaction_preds[data_type]
            rel_binary_interaction_gts = data_type_rel_binary_interaction_gts[data_type]
            entity_preds = data_type_entity_preds[data_type]
            entity_gts = data_type_entity_gts[data_type]

            cls_report = classification_report(
                rel_gts, rel_preds,
                labels=list(range(len(self.relationNames))),
                target_names=self.relationNames,
                output_dict=True,
                digits=4  # non existing relations will be counted as True
            )
            real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
            for rel_name in self.relationNames:
                if cls_report[rel_name]['support'] == 0:
                    continue
                for score_type in ['precision', 'recall', 'f1-score']:
                    real_macro_values[score_type].append(cls_report[rel_name][score_type])
            real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}

            # Log per-data-type metrics if needed
            if logging_information is not None:
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_precision': cls_report['macro avg']['precision']}, step=logging_information['checkpoint_id'])
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_recall': cls_report['macro avg']['recall']}, step=logging_information['checkpoint_id'])
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_macro_f1': cls_report['macro avg']['f1-score']}, step=logging_information['checkpoint_id'])

            # Print per-data-type classification report
            print(f'\nData Type: {data_type}\n')
            cls_report_str = classification_report(
                rel_gts, rel_preds,
                labels=list(range(len(self.relationNames))),
                target_names=self.relationNames,
                digits=4  # non existing relations will be counted as True
            )
            print(cls_report_str)
            print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

            # Print per-data-type binary interaction and entity classification reports
            binary_interaction_cls_report = classification_report(
                rel_binary_interaction_gts, rel_binary_interaction_preds,
                labels=[0, 1],
                target_names=['no_interaction', 'interaction'],
                output_dict=False,
                digits=4
            )
            entity_cls_report = classification_report(
                entity_gts, entity_preds,
                labels=list(range(len(self.classNames))),
                target_names=self.classNames,
                output_dict=False,
                digits=4
            )
            print(f'\nBinary Interaction Classification Report for Data Type {data_type}\n')
            print(binary_interaction_cls_report)
            print(f'\nEntity Classification Report for Data Type {data_type}\n')
            print(entity_cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True, digits=4)
        old_macro_f1 = results['macro avg']['f1-score']
        real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
        for rel_name in self.relationNames:
            if results[rel_name]['support'] == 0:
                continue
            for score_type in ['precision', 'recall', 'f1-score']:
                real_macro_values[score_type].append(results[rel_name][score_type])
        real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}
        macro_f1 = real_macro_values['f1-score']
        if logging_information is not None:
            # logging_information will have a key use it to log to wandb. It will also have a checkpoint int, which we also want to log (similar to epoch). Also we want to use the split to log as train or val
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_precision': results['macro avg']['precision']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_recall': results['macro avg']['recall']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_macro_f1': results['macro avg']['f1-score']}, step=logging_information['checkpoint_id'])

        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, digits=4)  # non existing relations will be counted as True
        print(cls_report)
        print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

        # a less granular report for binary_interaction and entity detection
        binary_interaction_cls_report = classification_report(all_rel_binary_interaction_gts, all_rel_binary_interaction_preds, labels=[0, 1],
                                                            target_names=['no_interaction', 'interaction'], output_dict=False, digits=4)
        entity_cls_report = classification_report(all_entity_gts, all_entity_preds, labels=list(range(len(self.classNames))), target_names=self.classNames, output_dict=False, digits=4)
        print(f'\nBinary Interaction Classification Report for {split}\n')
        print(binary_interaction_cls_report)

        print(f'\nEntity Classification Report for {split}\n')
        print(entity_cls_report)

        return macro_f1




    