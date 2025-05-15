# Adopted from https://github.com/egeozsoy/MM-OR/scene_graph_generation/scene_graph_prediction/main.py
import os
import sys

os.environ["WANDB_DIR"] = os.path.abspath("wandb")
os.environ["TMPDIR"] = os.path.abspath("wandb")

import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
from types import SimpleNamespace

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Adjust path relative to the current working directory
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)
sys.path.append("../scene_graph_generation/LLaVA")

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset, DataCollatorForORDataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model import ModelWrapper
from scene_graph_generation.helpers.config_utils import ConfigManager
from scene_graph_prediction.utils.util import read_classes, read_relationships

def update_config(config_dict, eval_config):
    egocentric_features = ["images"] # default
    exocentric_features = ["images"] # default
    if "gaze" in eval_config.egocentric_features:
        egocentric_features.append("gaze")
    if "hand" in eval_config.egocentric_features:
        egocentric_features.append("hand")
    if "audio" in eval_config.exocentric_features:
        exocentric_features.append("audio")
    if "point_cloud" in eval_config.exocentric_features:
        exocentric_features.append("point_cloud")
    if "speech" in eval_config.exocentric_features:
        exocentric_features.append("speech")
    
    config_dict["egocentric_features"] = egocentric_features
    config_dict["exocentric_features"] = exocentric_features
    config_dict["batch_size"] = eval_config.batch_size
    config_dict["dataset_name"] = eval_config.dataset_name

    return config_dict

def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def load_checkpoint_data(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def update_checkpoint_data(file_path, model_name, checkpoint_id, wandb_run_id=None):
    data = load_checkpoint_data(file_path)
    if model_name not in data:
        data[model_name] = {"checkpoints": [], "wandb_run_id": wandb_run_id}
    if checkpoint_id not in data[model_name]["checkpoints"]:
        data[model_name]["checkpoints"].append(checkpoint_id)
    if wandb_run_id:
        data[model_name]["wandb_run_id"] = wandb_run_id
    with open(file_path, 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='path to eval samples')
    parser.add_argument('--benchmark_on', type=str, default="egoexor", help='name of the method for the benchmarking')
    parser.add_argument('--mode', type=str, default="egoexor", help='Evaluation Mode')

    args = parser.parse_args()
    pl.seed_everything(42, workers=True)

    device = "cuda"
    device_map = "auto"

    config = config_loader(args.config)
    config = SimpleNamespace(**config)
    mode = args.mode 
    shuffle = True
    batch_size = 16

    name = args.benchmark_on
    config.dataset_name = name
    config.data_path = args.data_path
    config.batch_size = batch_size
    if args.mode == "infer":
        config.split = "test"
    else:
        config.split = "validation"

    print("Evaluation Mode : ", mode, " on ", name)

    config_manager = ConfigManager()
    config_dict = config_manager.load_config()
    config_dict = update_config(config_dict, config)
    config_manager.update_config(config_dict)

    print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n--- Device {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

        if torch.cuda.memory_allocated(i) == 0:
            device_to_use = f"cuda:{i}"
            break

    relationNames = read_relationships("./data/relationships.txt")
    classNames = read_classes("./data/classes.txt")

    if mode == 'evaluate':
        print(f'Model path: {args.model_path}')
        eval_dataset = ORDataset(
            data_path=config.data_path,
            hdf5_path=Path(config.data_dir) / config.hdf5_path,
            data_args=config,
            split="test"
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataCollatorForORDataset(config)
        )
        # Init Model
        model = ModelWrapper(hdf5_path=Path(config.data_dir) / config.hdf5_path,
                    dataset_name = config.dataset_name,
                    relationNames=relationNames, 
                    classNames=classNames, 
                    model_path=args.model_path,
                    temporality=config.temporality,
                    mv_type = "learned",
                    device = device,
                    device_map = device_map
                )
        model.validate(eval_loader, limit_val_batches=None)

    elif mode == "eval_all":
        print(f'Model path: {args.model_path}')
        evaluated_file = 'evaluated_checkpoints.json'
        checkpoint_data = load_checkpoint_data(evaluated_file)
        model_path = Path(args.model_path) # checkpoint saved folder not subfolder
        model_name = model_path.name
        eval_every_n_checkpoints = 5 
        wandb_run_id = checkpoint_data.get(model_name, {}).get("wandb_run_id", None)
        logger = pl.loggers.WandbLogger(project='egoexor_eval', name=model_name, save_dir='logs', offline=False, id=wandb_run_id)
        
        eval_dataset = ORDataset(
            data_path=config.data_path,
            hdf5_path=Path(config.data_dir) / config.hdf5_path,
            data_args=config,
            split="test"
        )

        # always eval last checkpoint
        checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))
        print(checkpoints)
        checkpoint_idx = 0
        while checkpoint_idx < len(checkpoints):
            checkpoint = checkpoints[checkpoint_idx]
            if checkpoint_idx % eval_every_n_checkpoints != 0 and checkpoint_idx != len(checkpoints) - 1:
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            if checkpoint_idx == 0 and 'continue' not in model_name:
                checkpoint_idx += 1
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            checkpoint_id = int(checkpoint.name.split('-')[-1])
            if model_name in checkpoint_data and checkpoint_id in checkpoint_data[model_name]["checkpoints"]:
                print(f'Checkpoint {checkpoint_id} for model {model_name} already evaluated. Skipping.')
                checkpoint_idx += 1
                continue
            print(f'Evaluating checkpoint: {checkpoint}...')
            torch.cuda.empty_cache()

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=DataCollatorForORDataset(config)
            )
            model = ModelWrapper(hdf5_path=Path(config.data_dir) / config.hdf5_path,
                    dataset_name = config.dataset_name,
                    relationNames=relationNames, 
                    classNames=classNames, 
                    model_path=str(checkpoint),
                    temporality=config.temporality,
                    mv_type = "learned",
                    device = device,
                    device_map = device_map
                )
            
            model.validate(eval_loader, logging_information={'split': 'val', "logger": logger, 
                                                             "checkpoint_id": checkpoint_id})

            # cleanup before next run
            del model
            update_checkpoint_data(evaluated_file, model_name, checkpoint_id, logger.experiment.id)
            checkpoint_idx += 1
            checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))  # update checkpoints in case new ones were added

    elif mode == 'infer':
        print('INFER')
        print(f'Model path: {args.model_path}')
        infer_split = 'test'
        eval_dataset = ORDataset(
            data_path=config.data_path,
            hdf5_path=Path(config.data_dir) / config.hdf5_path,
            data_args=config,
            split=infer_split
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataCollatorForORDataset(config)
        )
        # Init Model
        model = ModelWrapper(hdf5_path=Path(config.data_dir) / config.hdf5_path,
                    dataset_name = config.dataset_name,
                    relationNames=relationNames, 
                    classNames=classNames, 
                    model_path=args.model_path,
                    temporality=config.temporality,
                    mv_type = "learned",
                    device = device,
                    device_map = device_map
                )
        results = model.infer(eval_loader)
        # results should be batch scan id -> list of relations
        output_name = f'scan_relations_{name}_{infer_split}.json'
        with open(output_name, 'w') as f:
            json.dump(results, f)

    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    main()
