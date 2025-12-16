## 🔍 Overview

This repository provides the official dataset and code for our paper **"High-Fidelity Human-Robot Dexterous Hand Motion Transfer Dataset via Neural Implicit Representation Alignment"**.

We present **HandRobotDataset**, a large-scale dataset for human-to-robot hand motion retargeting, featuring:

- 🖐️ **Multi-View Rendering**: 16 view angles per sample for robust 3D understanding
- 🤖 **Multi-Robot Support**: Shadow Hand, Schunk SIH, and Inspire Hand
- 🎯 **Precise Alignment**: Human hand aligned with robot joint angles via neural implicit representation
- 📐 **Rich Annotations**: MANO pose (θ), 3D joints, and corresponding robot joint configurations

## 📥 Dataset Download

The dataset (HandRobotDataset) is hosted on Baidu Netdisk. Please download it and unzip it into the root directory of this project.

*   **Link:** [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1ohnlfAHt-J11FEfXTjxhwA)
*   **Password:** `ah5r`

After downloading, ensure your directory looks like the structure below.

## 📂 Directory Structure

```text
./
│
├── images/                  # Folder for rendered images
│   ├── {id}_view_00.png     # Each sample contains 16 images from different views
│   └── ...
│
├── hand_embodiment/         # Core library: Kinematics, Meshes, and conversion logic
│   ├── mano.py
│   ├── embodiment.py
│   └── ...
│
├── all_labels.pkl           # Labels: Robot joint angles & MANO pose params (Theta)
├── all_joints.pkl           # Auxiliary: MANO 3D joint coordinates (XYZ)
│
├── HandRobotDataset.py      # PyTorch Dataset loading class
├── vis_dataset_hand.py      # Visualization script for verification
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
## 🛠️ Installation & Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

*Note: This project relies on the local folder `hand_embodiment/`. Please ensure this folder is kept in the project root directory as it contains necessary kinematic definitions.*

## 🚀 Quick Start

### 1. Data Loading

You can use `HandRobotDataset.py` to load the data. The dataset supports automatic splitting (9:1 ratio for train/test). Each unique sample ID is associated with 16 rendered images from different viewing angles.

```python
from torch.utils.data import DataLoader
from torchvision import transforms
from HandRobotDataset import HandRobotDataset

# Configuration
root_dir = "./"  # Ensure this points to where you unzipped the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize dataset (e.g., for Shadow Hand)
dataset = HandRobotDataset(root_dir, robot_name='shadow', transform=transform, train=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate through batch
for batch in dataloader:
    images = batch['image']             # [B, 3, 224, 224]
    mano_pose = batch['mano_pose']      # [B, 48] MANO Theta parameters
    robot_target = batch['target_robot']# [B, N] Robot joint angles
    
    print(f"Image Batch: {images.shape}")
    print(f"Robot Targets: {robot_target.shape}")
    break
```

### 2. Visualization & Verification

Use the `vis_dataset_hand.py` script to inspect the dataset. It visualizes the **orange MANO hand** alongside the **gray Robot hand**, driving the robot joints with the Ground Truth labels to verify alignment.

**Basic Usage:**

```bash
# Default: Shadow Hand, Sample Index 0
python vis_dataset_hand.py
```

**Command Line Arguments:**

*   `--robot`: Choose robot type (`shadow`, `schunk`, `inspire`).
*   `--idx`: Specify the sample index ID to visualize.
*   `--root`: Specify dataset root directory (default is `./`).

**Examples:**

```bash
# Check the 10th sample for Schunk hand
python vis_dataset_hand.py --robot schunk --idx 10

# Check the 50th sample for Inspire hand
python vis_dataset_hand.py --robot inspire --idx 50
```

## 📊 Data Format Description

### Dictionary Keys returned by `HandRobotDataset`:

| Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `image` | Tensor | `[3, 224, 224]` | Rendered RGB Image of the hand |
| `mano_pose` | Tensor | `[48]` or `[51]` | MANO model pose parameters (Theta) |
| `mano_joints` | Tensor | `[21, 3]` | MANO model 3D joint coordinates (XYZ) |
| `target_robot`| Tensor | `[N]` | Flattened Robot joint angles (Learning Target) |
| `robot_dict_data` | Dict | - | Detailed joint data grouped by fingers (e.g., `{'thumb': ...}`) |
| `id` | Str | - | Unique identifier for the raw motion sample |
| `view_idx` | Int | - | View index (0-15) corresponding to the image |

### Supported Robots

*   **`shadow`**: Shadow Dexterous Hand (High DoF)
*   **`schunk`**: Schunk  Hand
*   **`inspire`**: Inspire Hand
## 📄 License

This project is licensed under the [MIT License](LICENSE). The dataset is intended for research purposes only.
