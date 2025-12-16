import pybullet as p
import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# 启动 pybullet 物理引擎
p.connect(p.DIRECT)  # 使用无渲染模式

# 加载 URDF 文件
urdf_path = "/data/hand_embodiment222/hand_embodiment/model/shadow_hand/shadow_hand_right.urdf"
robot_id = p.loadURDF(urdf_path)

# 获取机器人的所有链接数量
num_joints = p.getNumJoints(robot_id)
print("Total number of joints:", num_joints)

# 存储网格数据
mesh_data = []

# 获取每个关节的几何体信息
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    link_name = joint_info[1].decode("utf-8")  # 获取链接名称

    # 获取视觉数据
    visual_data = p.getVisualShapeData(robot_id, i)
    
    # 如果有视觉数据，打印该关节的名称及其网格数量
    if visual_data:
        print(f"Joint {i} ({link_name}) has {len(visual_data)} visual shapes.")
    
    for shape_index in range(len(visual_data)):
        shape = visual_data[shape_index]
        
        if shape[2] == p.GEOM_MESH:  # 如果是网格类型
            mesh_file = shape[4].decode("utf-8")  # 获取网格文件路径
            mesh_position = np.array(shape[5])  # 获取网格的位置信息
            mesh_orientation = np.array(shape[6])  # 获取网格的旋转信息（四元数）
            
            # 将四元数转换为旋转矩阵
            rotation_matrix = R.from_quat(mesh_orientation).as_matrix()
            
            # 确保网格文件存在
            if os.path.exists(mesh_file):
                mesh = o3d.io.read_triangle_mesh(mesh_file)
                if mesh.is_empty():
                    print(f"Failed to load mesh from {mesh_file}")
                    continue
                
                # 应用网格的变换（位置和旋转）
                mesh.translate(mesh_position)
                mesh.rotate(rotation_matrix)  # 使用旋转矩阵来旋转网格
                
                # 收集网格数据
                mesh_data.append(mesh)

# 关闭 pybullet
p.disconnect()

# 如果没有加载到网格，给出提示
if not mesh_data:
    print("No meshes found in the URDF.")

# 合并所有网格（如果有多个网格）
combined_mesh = o3d.geometry.TriangleMesh()
for m in mesh_data:
    combined_mesh += m

# 打印加载的网格数量
print(f"Total meshes loaded: {len(mesh_data)}")

# 可视化合并后的网格
o3d.visualization.draw_geometries([combined_mesh])
