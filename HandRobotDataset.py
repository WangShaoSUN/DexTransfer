import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class HandRobotDataset(Dataset):
    def __init__(self, root_dir, robot_name='shadow', transform=None, train=True, split_ratio=0.9):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g. "/path/to/hand-dataset")
            robot_name (str): 机器人名称 ['schunk', 'shadow', 'inspire']
            transform (callable, optional): 图片预处理
            train (bool): 加载训练集还是测试集
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.robot_name = robot_name


        labels_path = os.path.join(root_dir, "all_labels.pkl") 
        joints_path = os.path.join(root_dir, "all_joints.pkl") 

        if not os.path.exists(labels_path) or not os.path.exists(joints_path):
            raise FileNotFoundError(f"Missing pkl files in {root_dir}")

        print(f"Loading data from {root_dir} ...")
        

        with open(labels_path, 'rb') as f:
            self.labels_data = pickle.load(f) 
        

        with open(joints_path, 'rb') as f:
            self.mano_joints_data = pickle.load(f)


        valid_ids = sorted(list(set(self.labels_data.keys()) & set(self.mano_joints_data.keys())))
        print(f"Valid samples aligned: {len(valid_ids)}")


        split_idx = int(len(valid_ids) * split_ratio)
        if train:
            self.ids = valid_ids[:split_idx]
        else:
            self.ids = valid_ids[split_idx:]
            
        self.views_per_sample = 16

    def __len__(self):
        return len(self.ids) * self.views_per_sample

    def __getitem__(self, idx):
        # 计算 ID 和 View
        sample_idx = idx // self.views_per_sample
        view_idx = idx % self.views_per_sample
        sample_id = self.ids[sample_idx]

        img_name = f"{sample_id}_view_{view_idx:02d}.png"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:

            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)


        # 1. Robot Labels (Target)
        labels_entry = self.labels_data[sample_id]

        robot_target = torch.from_numpy(labels_entry[self.robot_name]['array']).float()
        robot_dict_data = labels_entry[self.robot_name]['dict']
        
        # 2. MANO Pose Parameters 
        mano_pose_params = torch.from_numpy(labels_entry['mano']).float()

        # 3. MANO Joint Coordinates
        mano_joints_3d = torch.from_numpy(self.mano_joints_data[sample_id]).float()

        return {
            'image': image,                  # [3, 224, 224]
            'target_robot': robot_target,   
            'robot_dict_data': robot_dict_data,
            'mano_pose': mano_pose_params,   # [48/51]
            'mano_joints': mano_joints_3d,   # [21, 3]    
            'id': sample_id,
            'view_idx': view_idx
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch 


    ROOT_DIR = "E:\hand_dataset" 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    dataset = HandRobotDataset(ROOT_DIR, robot_name='shadow', transform=transform, train=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset Loaded. Total samples: {len(dataset)}")

    for batch in loader:
        print("\n" + "="*40)
        print("Data Batch Inspection")
        print("="*40)
        

        print(f"ID: {batch['id']}")
        print(f"Image: {batch['image'].shape}")
        print(f"MANO Pose Params (Theta): {batch['mano_pose'].shape}")
        print(f"MANO Joints 3D (XYZ):     {batch['mano_joints'].shape}")
        print(f"Target Robot (Shadow Array): {batch['target_robot'].shape}")


        print("\n--- Checking 'robot_dict_data' ---")
        
        if 'robot_dict_data' in batch:
            r_dict = batch['robot_dict_data']
            

            keys = list(r_dict.keys())
            print(f"Found Keys: {keys}")

            print("   Details per finger:")
            for key in keys:
                val = r_dict[key]
                

                if isinstance(val, torch.Tensor):
                    print(f"[{key}]: Type=Tensor, Shape={val.shape}")

                elif isinstance(val, list):
                    print(f"[{key}]: Type=List, Length={len(val)}")
                else:
                    print(f"[{key}]: Type={type(val)}")
                    
        else:
            print(" 'robot_dict_data' key NOT found in batch.")


        print("="*40)
        break