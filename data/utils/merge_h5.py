#!/usr/bin/env python
"""
Script to merge multiple EgoExOR HDF5 dataset files into a single file.

Supports both Legacy (ardamamur/EgoExOR) and HQ (TUM/EgoExOR) schemas.
Version is auto-detected from the first input file.

Example usage:
    python merge_h5.py --data_dir ./ --input_files file1.h5 file2.h5 --output_file merged.h5
    python merge_h5.py --data_dir ./ --input_files hq_*.h5 --output_file merged_hq.h5  # HQ (no splits)
"""
import os
import sys
import h5py
import random
import logging
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge multiple EgoExOR HDF5 datasets.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./", 
        required=True,
        help="Directory where data files are stored"
    )
    parser.add_argument(
        "--input_files", 
        type=str, 
        nargs="+", 
        required=True,
        help="List of input HDF5 files to merge."
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default=None,
        required=False,
        help="Path of splits file (Legacy only; optional for HQ).",
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Path to save the merged HDF5 dataset."
    )
    return parser.parse_args()


def detect_dataset_version(h5_path):
    """Detect Legacy vs HQ schema. Returns 'legacy' or 'hq'."""
    with h5py.File(h5_path, "r") as f:
        if "procedures" in f:
            return "hq"
        if "data" in f:
            return "legacy"
    raise ValueError("Unknown HDF5 schema: neither 'procedures' nor 'data' found at root")


def copy_group(src_file, dst_file, src_path, dst_path=None):
    """
    Copy a group and all its contents from src_file to dst_file.
    
    Args:
        src_file: Source HDF5 file object
        dst_file: Destination HDF5 file object
        src_path: Path to the source group
        dst_path: Path to the destination group (if different from src_path)
    """
    if dst_path is None:
        dst_path = src_path
    
    # Create intermediate groups if needed
    dst_parent = os.path.dirname(dst_path)
    if dst_parent and dst_parent not in dst_file:
        dst_file.create_group(dst_parent)
    
    # Copy the group
    if src_path in src_file:
        # Create the group if it doesn't exist
        if dst_path not in dst_file:
            dst_file.create_group(dst_path)
            
        # Copy attributes
        for attr_name, attr_value in src_file[src_path].attrs.items():
            dst_file[dst_path].attrs[attr_name] = attr_value
        
        # Copy datasets in this group
        for name, item in src_file[src_path].items():
            src_item_path = f"{src_path}/{name}"
            dst_item_path = f"{dst_path}/{name}"
            
            if isinstance(item, h5py.Group):
                # Recursively copy subgroups
                copy_group(src_file, dst_file, src_item_path, dst_item_path)
            else:
                # Copy dataset if it doesn't exist
                if dst_item_path not in dst_file:
                    src_file.copy(src_item_path, dst_file[dst_path])

def get_take_frame_entries(h5_file, version):
    """
    Get all TAKE frame entries in the format (procedure, phase, take_id, frame_id).

    Args:
        h5_file: HDF5 file object
        version: 'legacy' or 'hq'

    Returns:
        List of tuples (procedure, phase, take_id, frame_id)
    """
    entries = []

    if version == "legacy":
        if "data" not in h5_file:
            return entries
        for surgery_type in h5_file["data"]:
            for procedure_id in h5_file[f"data/{surgery_type}"]:
                take_path = f"data/{surgery_type}/{procedure_id}/take"
                if take_path in h5_file:
                    for take_id in h5_file[take_path]:
                        frames_path = f"{take_path}/{take_id}/frames/rgb"
                        if frames_path in h5_file:
                            try:
                                num_frames = h5_file[frames_path].shape[0]
                                for frame_id in range(num_frames):
                                    entries.append((surgery_type, int(procedure_id), int(take_id), frame_id))
                            except Exception as e:
                                logger.warning(f"Error getting frame count for {frames_path}: {e}")
    else:  # hq
        if "procedures" not in h5_file:
            return entries
        for procedure in h5_file["procedures"]:
            phases_grp = h5_file["procedures"][procedure].get("phases")
            if phases_grp is None:
                continue
            for phase in phases_grp:
                takes_grp = phases_grp[phase].get("takes")
                if takes_grp is None:
                    continue
                for take_id in takes_grp:
                    frames_path = f"procedures/{procedure}/phases/{phase}/takes/{take_id}/frames/rgb"
                    if frames_path in h5_file:
                        try:
                            num_frames = h5_file[frames_path].shape[0]
                            for frame_id in range(num_frames):
                                entries.append((procedure, int(phase), int(take_id), frame_id))
                        except Exception as e:
                            logger.warning(f"Error getting frame count for {frames_path}: {e}")

    return entries



def _populate_split(h5_file, split_name, entries):
    """Populate a dataset split with entries.
        
    Args:
        split_name: Name of the split (train, validation, test)
        entries: List of tuples (surgery_type, clip_id, subclip_id, frame_idx)
    """
    # Create structured array for split
    split_dtype = [
        ('surgery_type', h5py.string_dtype()),
        ('procedure_id', np.int32),
        ('take_id', np.int32),
        ('frame_id', np.int32)
    ]
    
    # Convert entries to structured array
    split_data = np.array(
        [(entry[0], int(entry[1]), int(entry[2]), int(entry[3])) for entry in entries],
        dtype=split_dtype
    )

    # Create group if it doesn't exist
    if "splits" not in h5_file:
        h5_file.create_group("splits")
    
        # Create or overwrite dataset
    if split_name in h5_file["splits"]:
        del h5_file[f"splits/{split_name}"]
        
    h5_file.create_dataset(
        f"splits/{split_name}",
        data=split_data,
        compression="gzip"
    )

def create_splits(h5_file, version, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42):
    """
    Create train/validation/test splits ensuring per-procedure constraints:
      - Procedures with 1 take: all to train
      - Procedures with 2 takes: one to train, one randomly to val or test
      - Procedures with >=3 takes: at least one take in val and one in test, rest split by ratio

    Args:
        h5_file: HDF5 file object
        version: 'legacy' or 'hq'
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    proc_takes = defaultdict(list)
    if version == "legacy":
        for surgery_type in h5_file["data"]:
            for procedure_id in h5_file[f"data/{surgery_type}"]:
                take_path = f"data/{surgery_type}/{procedure_id}/take"
                if take_path in h5_file:
                    for take_id in h5_file[take_path]:
                        proc_takes[(surgery_type, procedure_id)].append(take_id)
    else:  # hq
        for procedure in h5_file["procedures"]:
            phases_grp = h5_file["procedures"][procedure].get("phases")
            if phases_grp is None:
                continue
            for phase in phases_grp:
                takes_grp = phases_grp[phase].get("takes")
                if takes_grp is None:
                    continue
                for take_id in takes_grp:
                    proc_takes[(procedure, int(phase))].append(int(take_id))

    train_subclips, val_subclips, test_subclips = [], [], []

    # Assign takes according to rules
    for (surgery_type, procedure_id), takes in proc_takes.items():
        n = len(takes)
        if n == 1:
            train_ids, val_ids, test_ids = takes, [], []

        elif n == 2:
            t = takes.copy()
            random.shuffle(t)
            train_ids = [t[0]]
            # Randomly assign second to val or test
            if random.random() < val_size / (val_size + test_size):
                val_ids, test_ids = [t[1]], []
            else:
                val_ids, test_ids = [], [t[1]]

        else:
            t = takes.copy()
            random.shuffle(t)
            # Compute per-proc counts enforcing at least one in val and test
            n_val = max(1, round(val_size * n))
            n_test = max(1, round(test_size * n))
            n_train = n - n_val - n_test
            # Ensure at least one train
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            # Adjust sums
            total = n_train + n_val + n_test
            if total < n:
                n_train += (n - total)
            elif total > n:
                n_train -= (total - n)

            train_ids = t[:n_train]
            val_ids   = t[n_train:n_train + n_val]
            test_ids  = t[n_train + n_val:n_train + n_val + n_test]

        # Collect subclip entries
        for tid in train_ids:
            train_subclips.append((surgery_type, procedure_id, tid))
        for tid in val_ids:
            val_subclips.append((surgery_type, procedure_id, tid))
        for tid in test_ids:
            test_subclips.append((surgery_type, procedure_id, tid))

    def _frames_path(st, pid, tid):
        if version == "legacy":
            return f"data/{st}/{pid}/take/{tid}/frames/rgb"
        return f"procedures/{st}/phases/{pid}/takes/{tid}/frames/rgb"

    def _populate_frames(subclips, out_list):
        for st, pid, tid in subclips:
            frames_key = _frames_path(st, pid, tid)
            if frames_key in h5_file:
                for fid in range(len(h5_file[frames_key])):
                    out_list.append((st, pid, tid, fid))

    # Build frame-level splits
    train_entries, val_entries, test_entries = [], [], []
    _populate_frames(train_subclips, train_entries)
    _populate_frames(val_subclips,   val_entries)
    _populate_frames(test_subclips,  test_entries)

    # Shuffle for randomness
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)

    # Populate HDF5 groups
    _populate_split(h5_file, "train",      train_entries)
    _populate_split(h5_file, "validation", val_entries)
    _populate_split(h5_file, "test",       test_entries)

    # Optionally, print stats
    total = len(train_entries) + len(val_entries) + len(test_entries)
    print(f"Train: {len(train_entries)} frames ({len(train_entries)/total:.1%})")
    print(f"Val:   {len(val_entries)} frames ({len(val_entries)/total:.1%})")
    print(f"Test:  {len(test_entries)} frames ({len(test_entries)/total:.1%})")


def merge_files(input_files, splits_file, output_file):
    """
    Merge multiple HDF5 files into a single file.

    Supports both Legacy (data/...) and HQ (procedures/...) schemas.
    Version is auto-detected from the first input file.

    Args:
        input_files: List of input file paths
        splits_file: Path to splits HDF5 file (Legacy only; pass None for HQ)
        output_file: Output file path
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Detect version from first file and ensure all inputs match
    version = detect_dataset_version(input_files[0])
    logger.info(f"Detected dataset version: {version}")
    for f in input_files[1:]:
        v = detect_dataset_version(f)
        if v != version:
            raise ValueError(
                f"All input files must use the same schema. {input_files[0]} is {version}, "
                f"but {f} is {v}."
            )

    root_key = "data" if version == "legacy" else "procedures"

    with h5py.File(output_file, "w") as out_file:
        logger.info(f"Initializing base structure from {input_files[0]}")
        with h5py.File(input_files[0], "r") as first_file:
            copy_group(first_file, out_file, "metadata")
            if root_key not in out_file:
                out_file.create_group(root_key)

        all_entries = []
        for input_file in input_files:
            logger.info(f"Processing file: {input_file}")
            with h5py.File(input_file, "r") as in_file:
                file_entries = get_take_frame_entries(in_file, version)
                all_entries.extend(file_entries)
                logger.info(f"Found {len(file_entries)} frame entries in {input_file}")

                if root_key not in in_file:
                    continue

                if version == "legacy":
                    _merge_legacy(in_file, out_file)
                else:
                    _merge_hq(in_file, out_file)

        if splits_file is not None:
            logger.info(f"Loading splits from {splits_file}")
            with h5py.File(splits_file, "r") as split_h5:
                if "splits" not in split_h5:
                    logger.error("No 'splits' group found in your splits_file")
                    sys.exit(1)
                if "splits" in out_file:
                    del out_file["splits"]
                split_h5.copy("splits", out_file, name="splits")
        elif version == "hq":
            logger.info("HQ mode: no splits file provided (splits are optional for HQ)")

        logger.info(f"Dataset merge complete! Saved to {output_file}")


def _merge_legacy(in_file, out_file):
    """Merge Legacy schema: data/{surgery_type}/{procedure_id}/take/{take_id}"""
    for surgery_type in in_file["data"]:
        if surgery_type not in out_file["data"]:
            out_file["data"].create_group(surgery_type)
        for procedure_id in in_file[f"data/{surgery_type}"]:
            procedure_path = f"data/{surgery_type}/{procedure_id}"
            if procedure_id not in out_file[f"data/{surgery_type}"]:
                logger.info(f"Copying {procedure_path}")
                copy_group(in_file, out_file, procedure_path)
            else:
                takes_path = f"{procedure_path}/take"
                if takes_path in in_file and takes_path in out_file:
                    for take_id in in_file[takes_path]:
                        take_path = f"{takes_path}/{take_id}"
                        if take_id not in out_file[takes_path]:
                            logger.info(f"Copying new subclip {take_path}")
                            copy_group(in_file, out_file, take_path)
                        else:
                            logger.warning(f"Skipping {take_path} - already exists in output file")
                elif takes_path in in_file:
                    copy_group(in_file, out_file, takes_path)
                else:
                    logger.warning(f"No subclips found in {procedure_path}")


def _merge_hq(in_file, out_file):
    """Merge HQ schema: procedures/{procedure}/phases/{phase}/takes/{take}"""
    for procedure in in_file["procedures"]:
        if procedure not in out_file["procedures"]:
            out_file["procedures"].create_group(procedure)
        proc_path = f"procedures/{procedure}"
        phases_grp = in_file[proc_path].get("phases")
        if phases_grp is None:
            continue
        if "phases" not in out_file[proc_path]:
            out_file[proc_path].create_group("phases")
        for phase in phases_grp:
            phase_path = f"{proc_path}/phases/{phase}"
            if phase not in out_file[proc_path]["phases"]:
                out_file[proc_path]["phases"].create_group(phase)
            takes_grp = in_file[phase_path].get("takes")
            if takes_grp is None:
                continue
            if "takes" not in out_file[phase_path]:
                out_file[phase_path].create_group("takes")
            for take_id in takes_grp:
                take_path = f"{phase_path}/takes/{take_id}"
                if take_id not in out_file[phase_path]["takes"]:
                    logger.info(f"Copying {take_path}")
                    copy_group(in_file, out_file, take_path)
                else:
                    logger.warning(f"Skipping {take_path} - already exists in output file")

def main():
    args = parse_args()

    input_files = []
    data_root = Path(args.data_dir)
    for f in args.input_files:
        f = data_root / f
        input_files.append(f)

    if args.splits_file is not None:
        splits_file = data_root / args.splits_file
    else:
        splits_file = None

    # Verify input files exist
    for file_path in input_files:
        if not os.path.exists(file_path):
            logger.error(f"Input file does not exist: {file_path}")
            return 1
        
    output_file = data_root / args.output_file

        # Merge files
    try:
        merge_files(
            input_files,
            splits_file,
            output_file
        )
        return 0
    except Exception as e:
        logger.error(f"Error merging files: {e}", exc_info=True)
        return 1
    

if __name__ == "__main__":
    sys.exit(main()) 
