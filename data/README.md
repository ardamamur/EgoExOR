# EgoExOR : An Egocentric–Exocentric Operating Room Dataset for Comprehensive Understanding of Surgical Activities

The EgoExOR dataset provides a comprehensive, multimodal view of simulated surgical procedures, capturing both egocentric (ARIA glasses worn by participants) and exocentric (room cameras) perspectives in an operating room. It includes synchronized RGB video, eye gaze, hand tracking, audio, 3D point clouds, and scene graph annotations, all stored in an efficient HDF5 format. This dataset is designed to advance AI-driven surgical analysis, supporting applications like AI assistants, skill assessment, and multimodal modeling in medical and augmented reality domains.

## 🌟 Key Features

* **Multiple Modalities**: Each take includes RGB video, audio, eye gaze tracking, hand tracking, 3D point cloud data, and annotations, all captured simultaneously.

* **Time-Synchronized Streams**: All modalities are aligned on a common timeline, enabling precise cross-modal correlation (e.g. each video frame has corresponding gaze coordinates, hand positions, etc.).

* **Research Applicability**: EgoExOR aims to fill the gap in both egocentric and exocentric surgrical datasets, supporting development of AI assistants, skill assessment tools, and multimodal models in medical and augmented reality domains.

## 🚀 Quick Start with FiftyOne

[explore_with_fiftyone.webm](https://github.com/user-attachments/assets/3dbdd5be-5fa3-4fbf-a945-f65806bb0136)


```python
import fiftyone as fo
import fiftyone.zoo as foz

# Option 1: Download specific files from HuggingFace (default: miss_4.h5)
dataset = foz.load_zoo_dataset(
    "https://github.com/ardamamur/EgoExOR",
    max_samples=100,
    h5_files=["miss_4.h5"],  # Options: miss_1-4.h5, ultrasound_1-4.h5, ultrasound_5_14.h5, ultrasound_5_58.h5
)

# Option 2: Use local h5 file or directory
dataset = foz.load_zoo_dataset(
    "https://github.com/ardamamur/EgoExOR",
    max_samples=100,
    h5_path="/path/to/miss_4.h5",  # Single file or directory with .h5 files
)

# Option 3: Download full dataset and merge (~100GB required)
dataset = foz.load_zoo_dataset(
    "https://github.com/ardamamur/EgoExOR",
    max_samples=100,
    download_full=True,
)

fo.launch_app(dataset)
```

## 🚀 Quick Start
Get started with the dataset using the provided Python utilities. Refer to [´tutorial.ipynb´] for detailed examples.

### 1. Load an HDF5 File
```python
from data.utils.load_h5 import load_egoexor_h5
f_path = load_egoexor_h5("ardamamur/EgoExOR", "miss_4.h5")
```

### 2. Visualize a Frame
```python
from data.utils.visualize_timepoint import visualize_frame_group
visualize_frame_group(
    h5_path=f_path,
    surgery_type="MISS",
    procedure_id=4,
    take_id=2,
    frame_idx=195,
    save_frames = False # Set True to save each camera view separately
)
```

### 2. Visualize Take
```python
from data.utils.visualize_take import visualize_take
visualize_take(
    h5_file=f_path, # hdf5 file path
    surgery_type="MISS",
    procedure_id=4,
    take_id=2,
    output_path="miss_4_2.mp4",
    include_audio=False, # Set True to include audio (requires ffmpeg)
    debug_limit = 45, # Set to None for the entire take
)
```

### 4. Merge into single HDF5
```python
from data.utils.merge_h5 import merge_files
merge_files(
    input_files, # List of paths of individual HDF5 files
    splits_file, # Path to splits.h5
    output_file="EgoExOR.h5"
)
```

## 📂 Dataset Structure

The dataset is available in two formats:

- Individual Files: Hosted on the Hugging Face ([ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR))repository for efficient storage and access.
- Merged HDF5 File: Consolidates all data, including splits, into a single file. Should be done locally. 


### Individual Files (Hugging Face Repository)
Individual files are organized hierarchically by surgery type, procedure, and take, with components like RGB frames, eye gaze, and annotations stored separately for efficiency. The splits.h5 file defines the train, validation, and test splits.

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
- **`data/`**
  - **`<surgery_type>/`**
    - Directory named after the type of surgery (e.g., "MISS").
    - **`<procedure_id>/`**
      - Directory for a specific procedure.
      - **`takes/`**
        - **`<take_id>/`**
          - Directory for a specific recording (subclip) of a procedure.
          - **`sources/`**
            - Attributes: `source_count` (int), `source_0` (e.g., 'head_surgeon'), `source_1`, ...
              - Metadata for take cameras/sources, mapping array indices to camera/source IDs.
              - **Note**: Source names are in the same order as in `metadata/sources`, but for camera/source ID mapping (in gaze), use `metadata/sources` to get accurate source names.
          - **`frames/`**
            - `rgb` (Dataset: `[num_frames, num_cameras, height, width, 3]`, `uint8`)
              - Synchronized video frames with dimensions: number of frames, number of cameras, height, width, and 3 color channels.
          - **`eye_gaze/`**
            - `coordinates` (Dataset: `[num_frames, num_ego_cameras, 3]`, `float32`)
              - Eye gaze data from Egocentric devices with dimensions: number of frames, number of ego cameras, and 3 values (camera/source ID, x-coordinate, y-coordinate).
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
            - `waveform` (Dataset: `[num_samples, 2]`, `float32`)
              - Full stereo audio waveform with dimensions: number of samples and 2 channels (left, right).
            - `snippets` (Dataset: `[num_frames, samples_per_snippet, 2]`, `float32`)
              - 1-second stereo audio snippets aligned with frames, with dimensions: number of frames, samples per snippet, and 2 channels.
          - **`point_cloud/`**
            - `coordinates` (Dataset: `[num_frames, num_points, 3]`, `float32`)
              - Merged 3D point cloud coordinates from external cameras, with dimensions: number of frames, number of points, and 3 coordinates (x, y, z).
            - `colors` (Dataset: `[num_frames, num_points, 3]`, `float32`)
              - RGB colors for point cloud points (0-1 range), with dimensions: number of frames, number of points, and 3 color channels.
          - **`annotations/`**
            - `frame_idx`
              - `rel_annotations` (Dataset: `[n_annotations_per_frame, 3]`, `object` (byte string))
                - Text-based scene graph annotations (e.g., "head_surgeon holding scalpel") for each frame.
              - `scene_graph` (Dataset: `[n_annotations_per_frame, 3]`, `float32`)
                - Tokenized annotations using integer mappings from `metadata/vocabulary`, representing relationships in a structured format.
- **`splits.h5`**
  - A standalone file defining the dataset splits (`train`, `validation`, `test`).
  - Contains columns: `surgery_type`, `procedure_id`, `take_id`, `frame_id`
    - `surgery_type`: Type of surgical procedure (e.g., "appendectomy").
    - `procedure_id`: Unique identifier for a specific procedure.
    - `take_id`: Identifier for a specific recording (subclip) of a procedure.
    - `frame_id`: Identifier for individual frames within a take.

### Merged Dataset File (Locally)
The merged dataset file consolidates all data from the individual files into a single file, including the splits defined in `splits.h5`. This file follows the same structure as above, with an additional `splits/` directory that organizes the data into `train`, `validation`, and `test` subsets.

- **`splits/`**
  - **`train`**, **`validation`**, **`test`**
    - Each split is a dataset with columns: `surgery_type`, `procedure_id`, `take_id`, `frame_id`
      - Links to the corresponding data in the `data/` directory for easy access during machine learning tasks.


## ⚙️ Efficiency and Usability

- **Efficiency**:
  - **HDF5 Format**: Ideal for large, complex datasets with hierarchical organization and partial loading.
  - **Compression**: `gzip` reduces file size, critical for video and point cloud data.
  - **Chunking**: Enables efficient access to specific frame ranges, supporting sequence-based model training.
- **Usability**:
  - **Logical Structure**: Hierarchical organization (`data/surgery/procedure/take/modality`) simplifies navigation.
  - **Embedded Metadata**: Source mappings and vocabularies enhance self-containment.
- **Scalability**: Easily accommodates new surgeries or clips by adding groups to the existing hierarchy.


## 📜 License

Released under the **Apache 2.0 License**, permitting free academic and commercial use with attribution.

---

## 📚 Citation

A formal BibTeX entry will be provided upon publication. For now, please cite the dataset URL.

---

## 🤝 Contributing

Contributions are welcome! Submit pull requests to improve loaders, add visualizers, or share benchmark results.

---

*Dataset URL: [ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR)*  
*Last Updated: May 2025*

