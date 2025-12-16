import os
import sys
import numpy as np
import torch
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from torchvision import transforms
import argparse 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)


try:
    from HandRobotDataset import HandRobotDataset
except ImportError:
    print("error: not found HandRobotDataset.py")
    sys.exit(1)


try:
    from hand_embodiment.mano import HandState
    from hand_embodiment.embodiment import HandEmbodiment
    from hand_embodiment.target_configurations import TARGET_CONFIG
except ImportError:
    print("error:not found hand_embodiment ")
    sys.exit(1)


TRANSFORMS = {
    "shadow": pt.transform_from_exponential_coordinates(
        [-0.340, 2.110, 2.297, -0.385, -0.119, -0.094]),
    
    "schunk": pt.transform_from_exponential_coordinates(
        [-2.228, -0.163, 1.907, -0.04, -0.137, 0.047]),
        
    "inspire": pt.transform_from_exponential_coordinates(
        [2.3, 2.184, 0.196, -0.094, -0.070, 0.151])
}


def visualize_sample(dataset_root, idx, robot_name="shadow"):
    print(f"\n{'='*60}")
    print(f"visualizing -> Index: {idx} | Robot: {robot_name}")
    print(f"{'='*60}")

    dummy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = HandRobotDataset(
        root_dir=dataset_root, 
        robot_name=robot_name, 
        transform=dummy_transform, 
        train=True
    )
    

    try:
        data_sample = dataset[idx]
    except IndexError:

        print(f"Index {idx} out of range (dataset length: {len(dataset)})")
        return

    sample_id = data_sample['id']
    print(f"Sample ID: {sample_id}")

    mano_pose = data_sample['mano_pose'].numpy()

    robot_dict = data_sample['robot_dict_data']

    HAND_CONFIG = TARGET_CONFIG[robot_name]
    

    fingers = ["thumb", "index", "middle", "ring", "little"]
    if robot_name == "inspire":
        fingers = ["thumb", "index", "middle", "ring", "little"] 
    

    hand_state = HandState(left=False)
    hand_state.pose[:] = mano_pose
    

    emb = HandEmbodiment(hand_state, HAND_CONFIG, use_fingers=fingers)


    
    print("applying joint angles...")
    for finger_name in fingers:
        if finger_name in robot_dict:

            angles = robot_dict[finger_name]
            if isinstance(angles, torch.Tensor):
                angles = angles.numpy()
            elif isinstance(angles, list):
                angles = np.array(angles)
            

            joint_names = HAND_CONFIG["joint_names"][finger_name]

            if len(angles) == len(joint_names):
                for j_name, angle in zip(joint_names, angles):
                    emb.transform_manager_.set_joint(j_name, angle)
            else:
                print(f"Warning: {finger_name} the number of joints is not equal (angles_len:{len(angles)} vs joint_names_len:{len(joint_names)})")


    if robot_name in TRANSFORMS:
        base_transform = TRANSFORMS[robot_name]
    else:
        print(f"Warning: unKnown robot {robot_name}")
        base_transform = np.eye(4)


    hand_state.recompute_mesh(base_transform)

    hand_state.hand_mesh.paint_uniform_color((1, 0.5, 0)) 


    fig = pv.figure(window_name=f"Vis: {robot_name} - ID: {sample_id}")
    

    graph = pv.Graph(
        emb.transform_manager_, 
        HAND_CONFIG["base_frame"], 
        show_frames=False, 
        show_connections=False, 
        show_visuals=True, 
        show_collision_objects=False, 
        s=0.02
    )
    graph.add_artist(fig)
    

    fig.add_geometry(hand_state.hand_mesh)


    fig.show()


if __name__ == "__main__":

    DATASET_ROOT = CURRENT_DIR  

    parser = argparse.ArgumentParser(description="Hand Robot Dataset Visualizer")


    parser.add_argument(
        '--robot', 
        type=str, 
        default='shadow', 
        choices=['shadow', 'schunk', 'inspire'],
        help="type of robot: shadow schunk or inspire (default: shadow)"
    )


    parser.add_argument(
        '--idx', 
        type=int, 
        default=50, 
        help="index of the sample in the dataset (default: 0),range: 0-66391"
    ) 

    args = parser.parse_args()          

    visualize_sample(DATASET_ROOT, args.idx, args.robot)