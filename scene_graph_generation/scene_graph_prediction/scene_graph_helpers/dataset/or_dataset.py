import json
from pathlib import Path
import random
import os
import sys
import torch
import torchaudio
import numpy as np
import h5py
from typing import Dict, Sequence, Any
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from .dataset_utils import GAZE_FIXATION_TO_TAKE

def _needs_fixation(role: str, take_path: str) -> bool:
    """
    Return True if the given (role, take_path) combination requires
    the extra gaze-fixation offset.
    """
    takes_for_role = GAZE_FIXATION_TO_TAKE.get(role, [])
    #print(f"Checking fixation for {role} in {take_path}")
    return take_path in takes_for_role

class ORDataset(Dataset):
    """Dataset for evaluation/inference with EgoExOR HDF5 data."""
    def __init__(self, 
                 data_path: str, 
                 hdf5_path: str, 
                 data_args: Any, 
                 split: str = 'val'):
        assert split in ['val', 'test'], f"Split must be 'val' or 'test', got {split}"
        self.split = split
        self.data_path = Path(data_path)
        self.hdf5_path = hdf5_path
        self.data_args = data_args

        # Load JSON data
        with self.data_path.open() as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _load_multimodal_data(self, sample: Dict) -> Dict:
        """Load multimodal data from HDF5 for a given sample."""
        hdf5_indices = sample['hdf5_indices']
        surgery_type = hdf5_indices['surgery_type']
        procedure_id = hdf5_indices['procedure_id']
        take_id = hdf5_indices['take_id']
        frame_idx = hdf5_indices['frame_idx']
        available_modalities = set(hdf5_indices['available_modalities'])

        is_egoexor = True if self.data_args.dataset_name == "egoexor" else False
        is_4dor = True if self.data_args.dataset_name == "4dor" else False
        is_mmor = True if self.data_args.dataset_name == "mmor" else False


        # Filter modalities based on dataset type
        if is_4dor:
            available_modalities &= {"ego_frames", "exo_frames"}
        if is_mmor:
            available_modalities -= {"eye_gaze", "eye_gaze_depth", "hand_tracking"}
        
        # stack the available modalities -> they are list of list so instead make it single list

        path = f'data/{surgery_type}/{procedure_id}/take/{take_id}'

        ego_source_names, exo_source_names = [], []
        ego_source_ids,   exo_source_ids   = [], []


        with h5py.File(self.hdf5_path, 'r') as f:
            # -- get the source name map -- #
            sources_path = f"{path}/sources"
            camera_names = {}
            ego_indices, exo_indices = [], []

            if sources_path in f:
                src_grp = f[sources_path]
                camera_count = src_grp.attrs.get('source_count', 0)

                # read each source_i --> name . RGB images sorted in the same order
                for i in range(camera_count):
                    key = f'source_{i}'
                    if key in src_grp.attrs:
                        name = src_grp.attrs[key]
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        camera_names[i] = name

                        # classify ego/exo cameras
                        if name in self.data_args.ego_sources:
                            ego_indices.append(i)
                        elif name in self.data_args.exo_sources:
                            if name == "ultrasound":
                                # check if ultrasound listed in available modalities
                                if 'ultrasound' in available_modalities:
                                    exo_indices.append(i)
                            else:
                                exo_indices.append(i)
            
            if ego_indices:
                # print the ego camera names
                #print(f"Ego cameras in {path}: {[camera_names[i] for i in ego_indices]}")
                ego_range = (min(ego_indices), max(ego_indices) + 1)
            else:
                print(f"Warning: No ego cameras found in {path}. Using default range (0, 4).")
                ego_range = (0, 4)

            if exo_indices:
                # print the exo camera names
                #print(f"Exo cameras in {path}: {[camera_names[i] for i in exo_indices]}")
                exo_range = (min(exo_indices), max(exo_indices) + 1)
            else:
                print(f"Warning: No exo cameras found in {path}. Using default range (4, 9).")
                exo_range = (4, 9)


            # --- load RGB frames for this timestep ---
            frame_ds = f[f'{path}/frames/rgb']
            frame_rgb = torch.from_numpy(frame_ds[frame_idx]).float()
            # frame_rgb shape = (n_cams, H, W, 3)

            # --- Ego frames --- #
            if 'ego_frames' in available_modalities:
                ego_slice = frame_rgb[ego_range[0]:ego_range[1]]
                ego_list = [(ego_slice[j], j + ego_range[0]) for j in range(ego_slice.shape[0])]
                for img, cam_idx in ego_list:
                    if isinstance(img, torch.Tensor):
                        # Assuming img is [H, W, 3], convert to numpy and then PIL
                        if isinstance(img, torch.Tensor):
                            if not img.any():          # all pixels zero?
                                # print(f"Warning: All pixels are zero in {path} for camera {cam_idx}. Skipping this frame.")
                                continue
                        ego_source_names.append(camera_names.get(cam_idx, f"source_{cam_idx}"))
                        ego_source_ids.append(cam_idx)
            
            # --- Exo frames ---
            if 'exo_frames' in available_modalities:
                exo_slice = frame_rgb[exo_range[0]:exo_range[1]]
                exo_list = [(exo_slice[j], j + exo_range[0]) for j in range(exo_slice.shape[0])]
                exo_source_names = []
                exo_source_ids = []
                for img, cam_idx in exo_list:
                    exo_source_names.append(camera_names.get(cam_idx, f"source_{cam_idx}"))
                    exo_source_ids.append(cam_idx)

        
        data_dict = {}
        data_dict['ego_source_ids']   = ego_source_ids
        data_dict['ego_source_names'] = ego_source_names
        data_dict['exo_source_ids']   = exo_source_ids
        data_dict['exo_source_names'] = exo_source_names
            
        # Fallback zeros when truly no images and multimodal
        if not ego_source_ids and not exo_indices and self.data_args.is_multimodal:
            data_dict['exo_source_ids']   = [0]
            data_dict['exo_source_names'] = []
            data_dict['ego_source_ids']   = [0]
            data_dict['ego_source_names'] = []

        return data_dict


    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        sample['sample_id'] = f"{sample['hdf5_indices']['surgery_type']}_{sample['hdf5_indices']['procedure_id']}_{sample['hdf5_indices']['take_id']}_{sample['hdf5_indices']['frame_idx']}"
        
        # Load multimodal data
        multimodal_data = self._load_multimodal_data(sample)
        data_dict = {
            "sample" : sample,
            **multimodal_data
        }

        # Fallback for multimodal model with no images
        if not (multimodal_data.get('ego_source_ids') or multimodal_data.get('exo_source_idsframes')) and self.data_args.is_multimodal:
            data_dict['exo_source_ids'] = [0]
            data_dict['exo_source_names'] = []
            if self.data_args.dataset_name == "egoexor":
                data_dict['ego_source_ids'] = [0]
                data_dict['ego_source_names'] = []

        return data_dict
    


@dataclass
class DataCollatorForORDataset(object):
    """Collate examples for supervised fine-tuning."""

    data_args: Any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        if any ('sample' in instance for instance in instances):
            batch["sample"] = [instance.get('sample', []) for instance in instances]
        if any('ego_source_ids' in instance for instance in instances):
            batch['ego_source_names'] = [instance.get('ego_source_names', []) for instance in instances]
            batch['ego_source_ids'] = [instance.get('ego_source_ids', []) for instance in instances]
        if any('exo_source_ids' in instance for instance in instances):
            batch['exo_source_names'] = [instance.get('exo_source_names', []) for instance in instances]
            batch['exo_source_ids'] = [instance.get('exo_source_ids', []) for instance in instances]

        return batch