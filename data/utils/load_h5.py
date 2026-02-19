import h5py
from pathlib import Path
from typing import Literal, Tuple, Union

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


DatasetVersion = Literal["legacy", "hq"]


def detect_dataset_version(h5_path: Union[str, Path]) -> DatasetVersion:
    """
    Detect whether the HDF5 file uses legacy (EgoExOR) or HQ (EgoExOR-HDF5) schema.

    Legacy: root key /data, path /data/{surgery_type}/{procedure_id}/take/{take_id}
    HQ: root key procedures, path procedures/{procedure}/phases/{phase}/takes/{take}

    Returns:
        "legacy" or "hq"
    """
    with h5py.File(h5_path, "r") as f:
        if "procedures" in f:
            return "hq"
        if "data" in f:
            return "legacy"
    raise ValueError("Unknown HDF5 schema: neither 'procedures' nor 'data' found at root")


def list_available_takes(h5_path: Union[str, Path]) -> list:
    """
    List all take paths in the HDF5 file. Use to discover procedure/phase/take.

    Returns:
        List of (procedure, phase, take) tuples for both formats.
    """
    version = detect_dataset_version(h5_path)
    result = []
    with h5py.File(h5_path, "r") as f:
        if version == "legacy":
            if "data" not in f:
                return result
            for st in f["data"].keys():
                for proc_id in f[f"data/{st}"].keys():
                    take_grp = f["data"].get(f"{st}/{proc_id}/take")
                    if take_grp is None:
                        continue
                    for tid in take_grp.keys():
                        result.append((st, int(proc_id), int(tid)))
        else:
            if "procedures" not in f:
                return result
            for proc in f["procedures"].keys():
                phases = f["procedures"][proc].get("phases")
                if phases is None:
                    continue
                for ph in phases.keys():
                    takes_grp = f["procedures"][proc]["phases"][ph].get("takes")
                    if takes_grp is None:
                        continue
                    for t in takes_grp.keys():
                        result.append((proc, int(ph), int(t)))
    return sorted(result)


def get_take_path(
    version: DatasetVersion,
    *,
    procedure: str = None,
    phase: int = None,
    take: int = None,
) -> str:
    """
    Build the HDF5 group path for a take.

    Both formats use procedure, phase, take:
    - procedure: e.g. "MISS", "Ultrasound"
    - phase: procedure/phase index (legacy: procedure_id, HQ: phase)
    - take: take index
    """
    if procedure is None or phase is None or take is None:
        raise ValueError("All of procedure, phase, take are required")
    if version == "legacy":
        return f"/data/{procedure}/{phase}/take/{take}"
    return f"procedures/{procedure}/phases/{phase}/takes/{take}"


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
        str: Path to the downloaded HDF5 file
    """
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required for load_egoexor_h5. Install with: pip install huggingface_hub")
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
