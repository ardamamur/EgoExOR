import h5py
from huggingface_hub import hf_hub_download
from pathlib import Path

def load_egoexor_h5(
    repo_id: str = "ardamamur/EgoExOR",
    filename: str = "miss_4.h5",
    cache_dir: str = None,
    verbose: bool = True
) -> h5py.File:
    """
    Load a specific HDF5 file from a Hugging Face dataset repo.

    Args:
        repo_id (str): Hugging Face dataset repo ID
        filename (str): HDF5 filename (e.g. 'miss_4.h5', 'ultrasound_5_14.h5')
        cache_dir (str): Optional custom cache directory
        verbose (bool): Whether to print download status

    Returns:
        h5py.File: Read-only file handle to the requested HDF5 dataset
    """
    if verbose:
        print(f"📥 Downloading {filename} from {repo_id}...")

    h5_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
        local_dir=cache_dir,
        local_dir_use_symlinks=False
    )

    if verbose:
        print(f"✅ Loaded {filename} → {h5_path}")

    return h5_path
