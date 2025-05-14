# Adopted from https://github.com/egeozsoy/MM-OR/scene_graph_prediction/llava_helpers/generate_dataset_for_llava.py
import argparse
import random
import warnings
import os
from collections import Counter
from pathlib import Path
from random import shuffle

import h5py
import json_tricks as json
import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from tqdm import tqdm
import sys

from scene_graph_prediction.llava_helpers.scene_graph_converters import parse_llava_sg, llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from scene_graph_prediction.llava_helpers.scene_graph_templates import SCENE_GRAPH_PROMPT
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import (
    ENTITY_VOCAB, RELATION_VOCAB,
    EGO_SOURCES, EXTERNAL_PATTERN, EXO_SOURCES, ROBOT_SOURCES,
    reversed_entity_synonyms, reversed_relation_synonyms
)

warnings.filterwarnings('ignore')

def config_loader(dataset_name: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / f'{dataset_name}.json' 
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config

def scene_graph_to_string(scene_graph):
    '''
    Scene graph is a list of relations in the form of (subject, object, predicate)
    '''
    out = '<SG> '
    for (subject, object, predicate) in scene_graph:
        subject = subject.replace('_', ' ').lower()
        object = object.replace('_', ' ').lower()
        predicate = predicate.replace('_', ' ').lower()
        out += f'{subject},{object},{predicate}; '
    out = out.rstrip('; ') + ' </SG>'
    return out

def apply_template(scene_graph, timepoint, sample_id, hdf5_indices):
    human_prompt = SCENE_GRAPH_PROMPT
    sample = {
        'id': sample_id,
        'timepoint': timepoint,
        'hdf5_indices': hdf5_indices,
        'conversations': [
            {'from': 'human', 'value': f"<image>\n{human_prompt}"},
            {'from': 'gpt', 'value': scene_graph}
        ]
    }
    return sample

def generate_finetuning_samples_from_hdf5(
    hdf5_path,
    split,
    config,
    entity_vocab,
    predicate_vocab,
    n_permutations=1,
    modality_dropout_prob=0.5,
    reduce_ratio=15,
):
    samples = []
    available_modalities = set()
    # Determine which modalities are globally enabled
    enabled_modalities = {
        mod for mod, settings in config['modalities'].items() if settings.get('enabled', False)
    }

    with h5py.File(hdf5_path, 'r') as f:
        # Load split indices
        split_data = f[f'splits/{split}'][()]
        indices = [
            (
                item['surgery_type'].decode('utf-8'),
                item['procedure_id'],
                item['take_id'],
                item['frame_id'],
            )
            for item in split_data
        ]
        if split in ["validation", "test"]:
            num_samples = len(indices) // reduce_ratio   # integer division
            # (optional) for reproducibility
            random.seed(42)
            # draw that many *unique* items at random
            print("reduce the val size by %15")
            selected_indices = random.sample(indices, num_samples)
            indices = selected_indices

        missing_triplets = []

        for surgery_type, procedure_id, take_id, frame_idx in tqdm(
            indices, desc='Generating samples'
        ):
            path = f"data/{surgery_type}/{procedure_id}/take/{take_id}"

            # Read sources via attributes
            sources = []
            sg_path = f"{path}/sources"
            if sg_path in f:
                sg = f[sg_path]
                count = sg.attrs.get('source_count', 0)
                for i in range(count):
                    raw = sg.attrs.get(f"source_{i}")
                    name = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
                    sources.append(name)

            # Determine source flags
            has_ego = any(src in EGO_SOURCES for src in sources)
            has_robot = any(src in ROBOT_SOURCES for src in sources)
            has_external = any(EXTERNAL_PATTERN.match(src) for src in sources)
            has_exo = any(src in EXO_SOURCES or EXTERNAL_PATTERN.match(src) for src in sources)

            # Reset available modalities per frame
            available_modalities = set()

            # Ego-based modalities
            if has_ego:
                if 'ego_frames' in enabled_modalities and 'frames' in f[f'{path}']:
                    # always include ego images later apply image dropout 
                    available_modalities.add('ego_frames')
                for mod in ('eye_gaze', 'eye_gaze_depth', 'hand_tracking', 'audio'):
                    if mod in enabled_modalities and mod in f[f'{path}']:
                        if random.random() >= modality_dropout_prob:
                            available_modalities.add(mod)

            # Exo frames for OR light, microscope, simstation, or external
            if has_exo and 'exo_frames' in enabled_modalities and 'frames' in f[f'{path}']:
                # always include exo images later apply image dropout
                available_modalities.add('exo_frames')

            # Ultrasoun screen recordings 
            if has_robot and "ultrasound" in enabled_modalities and "frames" in f[f'{path}']:
                if random.random() >= modality_dropout_prob:
                    available_modalities.add('ultrasound')

            # Point cloud for external cameras
            if has_external and 'point_cloud' in enabled_modalities and 'point_cloud' in f[f'{path}']:
                if random.random() >= modality_dropout_prob:
                    available_modalities.add('point_cloud')

            # Load annotations
            annotation_path = f"{path}/annotations/frame_{frame_idx}/rel_annotations"
            if annotation_path not in f:
                missing_triplets.append(annotation_path)
                continue
            
            triplets = []
            for triplet in f[annotation_path]:
                # Parse byte-string like b'head_surgeon holding scalpel'
                parts = [x.decode('utf-8') for x in triplet]
                if len(parts) >= 3:  # Ensure we have at least subject, predciate, object
                    # If there are more than 3 parts, assume the middle is the predicate
                    if len(parts) > 3:
                        raw_sub = parts[0]
                        raw_obj = parts[-1]
                        raw_pred = " ".join(parts[1:-1])
                    else:
                        raw_sub, raw_pred, raw_obj = parts

                    # normalize entity and relation names
                    sub = reversed_entity_synonyms.get(raw_sub, raw_sub)
                    obj = reversed_entity_synonyms.get(raw_obj, raw_obj)
                    pred = reversed_relation_synonyms.get(raw_pred, raw_pred)
                    triplets.append((sub, obj, pred))

                else:
                    print(f"Warning: Malformed triplet '{triplet}' in {annotation_path}") 

            # Prepare sample metadata
            sample_prefix = f"{surgery_type}_{procedure_id}_{take_id}_{frame_idx}"
            hdf5_indices = {
                'surgery_type': surgery_type,
                'procedure_id': procedure_id,
                'take_id': take_id,
                'frame_idx': frame_idx,
                'available_modalities': list(available_modalities),
            }

            # Generate permutations
            for pi in range(n_permutations):
                random.shuffle(triplets)
                sg_str = scene_graph_to_string(triplets)
                sample = apply_template(
                    sg_str,
                    timepoint=frame_idx,
                    sample_id=f"{sample_prefix}_{pi}",
                    hdf5_indices=hdf5_indices,
                )
                samples.append(sample)
    return samples, missing_triplets

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hdf5_path', type=str, default=None, help='Path to EgoExOR HDF5 file')
    parser.add_argument('--dataset_name', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.dataset_name)
    # Use config values for parameters
    hdf5_path = args.hdf5_path if args.hdf5_path else config['hdf5_path']
    N_PERM = config['preprocessing']['n_permutations']
    ADD_TEMPORAL = config['preprocessing']['temporal']['add_temporal']
    WITH_TEMPORAL_AUG = config['preprocessing']['temporal']['with_temporal_aug']
    DROP_HISTORY = config['preprocessing']['temporal']['drop_history']
    MODALITY_DROPOUT_PROB = config['preprocessing']['modality_dropout_prob']
    SPLIT = config['split']
    NAME = config['output']['json_filename_template'].format(
        split=SPLIT, n_perm=N_PERM, add_temp=ADD_TEMPORAL, with_temp_aug=WITH_TEMPORAL_AUG
    )
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    if MODALITY_DROPOUT_PROB > 0:
        NAME += f'_modalitydrop{MODALITY_DROPOUT_PROB}'

    print(f'Creating samples for LLaVA dataset with name {NAME}')

    # Create output directory
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'liuhaotian/llava-v1.5-7b',
        model_max_length=2048,
        padding_side='right',
        use_fast=False,
    )

    entity_vocab = {v: k for k, v in ENTITY_VOCAB.items()}
    predicate_vocab = {v: k for k, v in RELATION_VOCAB.items()}

    samples, missing_triplets = generate_finetuning_samples_from_hdf5(
        hdf5_path, SPLIT, config, entity_vocab, predicate_vocab,
        n_permutations=N_PERM, modality_dropout_prob=MODALITY_DROPOUT_PROB,
        reduce_ratio=config['preprocessing']['reduce_ratio'],
    )

    token_freq = Counter()
    longest_sample = -1
    for sample in tqdm(samples, desc='Calculating token frequencies'):
        for conversation in sample['conversations']:
            if conversation['from'] == 'gpt':
                tokenized = tokenizer.tokenize(conversation['value'])
                token_freq.update(tokenized)
                longest_sample = max(longest_sample, len(tokenized))

    shuffle(samples)

    with open(f'{output_dir}/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)

    with open(f'{output_dir}/{config["output"]["token_freq_filename"].format(n_perm=N_PERM)}', 'w') as f:
        json.dump(token_freq, f, indent=4)

    # Save missing triplets
    # with open(f'{output_dir}/{NAME}_missing_triplets.txt', 'w') as f:
    #     for item in missing_triplets:
    #         f.write(f"{item}\n")

if __name__ == '__main__':
    import subprocess
    subprocess.call(['nvidia-smi', '-L'])
    main()