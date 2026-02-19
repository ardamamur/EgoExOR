# EgoExOR: An Ego-Exo-Centric Operating Room Dataset for Surgical Activity Understanding

The EgoExOR dataset provides a comprehensive, multimodal view of simulated surgical procedures, capturing both egocentric (ARIA glasses worn by participants) and exocentric (room cameras) perspectives in an operating room. It includes synchronized RGB video, eye gaze, hand tracking, audio, 3D point clouds, and scene graph annotations, all stored in an efficient HDF5 format. This dataset is designed to advance AI-driven surgical analysis, supporting applications like AI assistants, skill assessment, and multimodal modeling in medical and augmented reality domains.

## 🌟 Key Features

* **Multiple Modalities**: Each take includes RGB video, audio, eye gaze tracking, hand tracking, 3D point cloud data, and annotations, all captured simultaneously.

* **Time-Synchronized Streams**: All modalities are aligned on a common timeline, enabling precise cross-modal correlation (e.g. each video frame has corresponding gaze coordinates, hand positions, etc.).

* **Research Applicability**: EgoExOR aims to fill the gap in both egocentric and exocentric surgical datasets, supporting development of AI assistants, skill assessment tools, and multimodal models in medical and augmented reality domains.

## 📦 Dataset Versions

| Version | Hugging Face | Resolution | Notes |
|---------|--------------|------------|-------|
| **Legacy** (v1) | [ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR) | 336×336 | Original release; LUT applied for Azure Kinect sources |
| **HQ** (EgoExOR-HQ) | [TUM/EgoExOR](https://huggingface.co/datasets/TUM/EgoExOR) | 1344×1344 | Pre-corrected RGB, raw depth, per-device audio |

Both versions use the same unified API: **`procedure`**, **`phase`**, **`take`** (see Quick Start below).


## 🚀 Quick Start

Get started with the dataset using the provided Python utilities. Refer to [`tutorial.ipynb`](tutorial.ipynb) for a full walkthrough.

### 1. Load an HDF5 File

```python
# Legacy (from Hugging Face)
from data.utils.load_h5 import load_egoexor_h5
f_path = load_egoexor_h5("ardamamur/EgoExOR", "miss_4.h5")

# HQ: use local path or load_egoexor_h5("TUM/EgoExOR")
# from pathlib import Path
# f_path = Path("/path/to/EgoExOR_HQ.h5")

procedure, phase, take = "MISS", 4, 2
kw = {"procedure": procedure, "phase": phase, "take": take}
```

### 2. Discover Available Takes

```python
from data.utils.load_h5 import list_available_takes
takes = list_available_takes(f_path)
print(takes)  # [(procedure, phase, take), ...]
```

### 3. Visualize a Frame

```python
from data.utils.visualize_timepoint import visualize_frame_group
visualize_frame_group(h5_path=f_path, frame_idx=195, save_frames=False, **kw)
```

### 4. Visualize a Take (Video)

```python
from data.utils.visualize_take import visualize_take
visualize_take(
    h5_file=f_path,
    output_path="take_preview.mp4",
    include_audio=False,
    debug_limit=45,  # Set to None for full take
    **kw,
)
```

### 4. Merge into single HDF5
`merge_files` supports both **Legacy** and **HQ** schemas (auto-detected). For HQ, `splits_file` is optional.
```python
from data.utils.merge_h5 import merge_files
merge_files(
    input_files,   # List of paths of individual HDF5 files
    splits_file,   # Path to splits.h5 (Legacy only; use None for HQ)
    output_file="EgoExOR.h5"
)
```

## 📐 Camera Calibration and Depth Metadata

EgoExOR provides camera calibration and depth-related metadata for the exocentric camera setup, enabling geometry-aware analysis and spatial reasoning.  
The calibration information can be found in [`data/calibration/exocentric/`](calibration/exocentric/)

For each exocentric camera, the dataset includes:

### Intrinsic Parameters
- Image resolution (`width`, `height`)
- Focal lengths and principal points
- Full intrinsic camera matrices for both **color** and **depth** sensors
- Radial and tangential distortion coefficients

### Depth and Sensor Parameters
- Depth field-of-view parameters
- Metric depth scaling information
- Depth-to-color sensor alignment (`color2depth_transform`)

### Camera Pose (Extrinsics)
- Camera pose represented as:
  - Translation (in meters)
  - Rotation (quaternion: `x y z w`)
- The pose is defined in a common reference frame, allowing spatial alignment across cameras and modalities

### Sensor-to-Sensor Transforms
- Depth-to-accelerometer transform
- Depth-to-gyroscope transform  
  *(provided for completeness and synchronization with auxiliary sensor streams)*




## 🔎 Visualization (Optional)
For interactive browsing, qualitative inspection, and debugging, the dataset can optionally be explored using external visualization frameworks such as [FiftyOne](https://docs.voxel51.com/).

```bash
pip install fiftyone h5py huggingface-hub
```

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load dataset (downloads miss_4.h5 by default)
dataset = foz.load_zoo_dataset(
    "https://github.com/AdonaiVera/EgoExOR",
    max_samples=100,
)

# Launch the FiftyOne App
fo.launch_app(dataset)
```

The FiftyOne loader supports multiple options:
- **Specific files**: `h5_files=["miss_1.h5", "ultrasound_1.h5"]`
- **Local files**: `h5_path="/path/to/egoexor.h5"`
- **Full dataset**: `download_full=True` (requires ~100GB)

For full FiftyOne integration details, see [AdonaiVera/EgoExOR](https://github.com/AdonaiVera/EgoExOR).


## 📂 Dataset Structure

The dataset is available in two formats:

- **Individual Files**: Hosted on Hugging Face ([ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR) for Legacy, [TUM/EgoExOR](https://huggingface.co/datasets/TUM/EgoExOR) for HQ) for efficient storage and access.
- **Merged HDF5 File**: Consolidates all data, including splits, into a single file. Done locally via `merge_h5`.

### Path Structure

| Version | Take path |
|---------|-----------|
| **Legacy** | `/data/{procedure}/{phase}/take/{take}` |
| **HQ** | `procedures/{procedure}/phases/{phase}/takes/{take}` |

Use `detect_dataset_version(f_path)` and `list_available_takes(f_path)` from `data.utils.load_h5` to auto-detect and discover takes.

### Individual Files (Hugging Face Repository)

Individual files are organized hierarchically by procedure, phase, and take. The splits.h5 file (Legacy) defines train, validation, and test splits.

- **`metadata/`**
  - `vocabulary/`
    - `entity` (Dataset: `name`, `id`)
      - Lists entities (e.g., objects, people) with their names and unique IDs.
    - `relation` (Dataset: `name`, `id`)
      - Lists relationships (e.g., "holding") with their names and unique IDs.
  - `sources/`
    - `sources` (Dataset: `name`, `id`)
      - Lists data sources (e.g., cameras like 'assistant', 'ultrasound', 'external') with their names and unique IDs.
      - **Note**: Camera IDs in `eye_gaze/coordinates` are mapped to this `sources` dataset for accurate source names. Do not use `takes/<take_id>/sources/` for mapping camera IDs to get the source names, though the source names are listed in the same order.
  - `dataset/`
    - Attributes: `version`, `creation_date`, `title`
      - Provides dataset-level information, such as version number, creation date, and title.
- **`data/`** (Legacy) or **`procedures/`** (HQ)
  - **Legacy**: `data/<procedure>/<phase>/take/<take>/`
  - **HQ**: `procedures/<procedure>/phases/<phase>/takes/<take>/`
  - **`<take_id>/`** (Legacy: under `take/`) – directory for a specific recording (subclip)
    - **`sources/`**
      - Attributes: `source_count` (int), `source_0` (e.g., 'head_surgeon'), `source_1`, ...
      - Metadata for take cameras/sources, mapping array indices to camera/source IDs.
      - **Note**: Source names are in the same order as in `metadata/sources`, but for camera/source ID mapping (in gaze), use `metadata/sources` to get accurate source names.
    - **`frames/`**
      - `rgb` (Dataset: `[num_frames, num_cameras, height, width, 3]`, `uint8`)
      - Synchronized video frames with dimensions: number of frames, number of cameras, height, width, and 3 color channels.
    - **`eye_gaze/`**
      - `coordinates` (Dataset: `[num_frames, num_ego_cameras, 3]`, `float32`)
      - Eye gaze data from egocentric devices with dimensions: number of frames, number of ego cameras, and 3 values (camera/source ID, x-coordinate, y-coordinate).
      - Invalid gaze points are marked as `[-1., -1.]`.
      - **Note**: The `camera_id` in the last dimension must be mapped to `metadata/sources` for the correct source name, not to `takes/<take_id>/sources/`.
    - **`eye_gaze_depth/`**
      - `values` (Dataset: `[num_frames, num_ego_cameras]`, `float32`)
      - Depth values for eye gaze in meters, synchronized with `eye_gaze/coordinates` (can use camera/source ID from `coordinates`).
      - Defaults to 1.0 if depth data is unavailable.
    - **`hand_tracking/`**
      - `positions` (Dataset: `[num_frames, num_ego_cameras, 17]`, `float32`)
      - Hand tracking data from egocentric devices with dimensions: number of frames, number of ego cameras, and 17 values (camera ID + 8 keypoints for left hand + 8 keypoints for right hand, including wrist, palm, and normals).
      - Invalid points are marked with `NaN`.
    - **`audio/`** (Optional)
      - `waveform` (Dataset: `[num_samples, 2]`, `float32`) – full stereo audio
      - `snippets` (Dataset: `[num_frames, samples_per_snippet, 2]`, `float32`) – 1-second stereo snippets aligned with frames
    - **`point_cloud/`**
      - `coordinates` (Dataset: `[num_frames, num_points, 3]`, `float32`)
      - `colors` (Dataset: `[num_frames, num_points, 3]`, `float32`) – RGB, 0–1 range
    - **`annotations/`**
      - `frame_idx/rel_annotations` – text-based scene graph (e.g., "head_surgeon holding scalpel")
      - `frame_idx/scene_graph` – tokenized annotations using `metadata/vocabulary`
- **`splits.h5`** (Legacy)
  - Standalone file defining the dataset splits (`train`, `validation`, `test`).
  - Contains columns: `surgery_type`, `procedure_id`, `take_id`, `frame_id`
    - `surgery_type`: Type of surgical procedure (e.g., "appendectomy").
    - `procedure_id`: Unique identifier for a specific procedure.
    - `take_id`: Identifier for a specific recording (subclip) of a procedure.
    - `frame_id`: Identifier for individual frames within a take.

### Merged Dataset File (Locally)
The merged HDF5 file consolidates all data from the individual files into a single file. It follows the same root structure as above (`data/` for Legacy, `procedures/` for HQ). Legacy merges include `splits/` from `splits.h5`; HQ uses `procedures/` without splits.

- **`splits/`** (Legacy merged only)
  - **`train`**, **`validation`**, **`test`**
    - Each split is a dataset with columns: `surgery_type`, `procedure_id`, `take_id`, `frame_id`
    - Links to the corresponding data in the `data/` directory for easy access during machine learning tasks.


## ⚙️ Efficiency and Usability

- **Efficiency**:
  - **HDF5 Format**: Ideal for large, complex datasets with hierarchical organization and partial loading.
  - **Compression**: `gzip` reduces file size, critical for video and point cloud data.
  - **Chunking**: Enables efficient access to specific frame ranges, supporting sequence-based model training.
- **Usability**:
  - **Logical Structure**: Hierarchical organization (`data/<procedure>/<phase>/take/<take>/modality` or `procedures/.../takes/<take>/modality` for HQ) simplifies navigation.
  - **Embedded Metadata**: Source mappings and vocabularies enhance self-containment.
- **Scalability**: Easily accommodates new surgeries or clips by adding groups to the existing hierarchy.


## 📜 License

Released under the **Apache 2.0 License**, permitting free academic and commercial use with attribution.

---

## 📚 Citation

```bibtex
@article{ozsoy2025egoexor,
  title={Egoexor: An ego-exo-centric operating room dataset for surgical activity understanding},
  author={{\"O}zsoy, Ege and Mamur, Arda and Tristram, Felix and Pellegrini, Chantal and Wysocki, Magdalena and Busam, Benjamin and Navab, Nassir},
  journal={arXiv preprint arXiv:2505.24287},
  year={2025}
}
```
---

## 🤝 Contributing

Contributions are welcome! Submit pull requests to improve loaders, add visualizers, or share benchmark results.

---

**Dataset URLs**  
- Legacy: [ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR)  
- HQ: [TUM/EgoExOR](https://huggingface.co/datasets/TUM/EgoExOR)  
*Last Updated: February 2025*

