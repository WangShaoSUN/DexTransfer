"""Configurations of robotic hands for embodiment mapping."""
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from pkg_resources import resource_filename

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# Target system configurations
# (See README.md > "Integrating a New Robotic Hand".)
# Each target system needs its own configuration, which is essentially a large
# dict. Make sure to add this to the TARGET_CONFIG dict at the end of the file.

# Required fields:
# ROBOT_CONFIG = {
#     "joint_names":
#         {  # map finger names to a list of joint names that control the finger
#             "thumb": [...],
#             "index": [...],
#             "middle": [...],
#             "ring": [...],
#             "little": [...],
#         },
#     "base_frame": "...",  # base frame of the hand
#     "ee_frames":
#         {  # map finger name to the name of the tip frame in the kinematic model
#             "thumb": "...",
#             "index": "...",
#             "middle": "...",
#             "ring": "...",
#             "little": "..."
#         },
#     "intermediate_frames":
#         {
#             "thumb": "...",
#             "index": "...",
#             "middle": "...",
#             "ring": "...",
#             "little": "..."
#         },
#     # transform from MANO base to hand base, array with shape (4, 4)
#     "handbase2robotbase": ...,
#     "model":  # kinematic model definition
#         {
#             # path to URDF file, resource_filename might be useful
#             "urdf": "...",
#             # base path of ROS package that contains URDF
#             "package_dir": "...",
#             "kinematic_model_hook": ...  # a callback, if required
#         },
#     "virtual_joints_callbacks":
#         {  # here we can introduce virtual joints that compute the state of real joints in a callback
#             # example:
#             "j_thumb_opp_binary": MiaVirtualThumbJoint("j_thumb_opp"),
#         },
#     "coupled_joints":
#     [  # coupled joints always have the same angle
#         # example:
#         ("middle", "j_mrl_fle"),
#         ("ring", "j_ring_fle"),
#         ("little", "j_little_fle"),
#     ],
#     # function will be executed after embodiment mapping to modify the result
#     "post_embodiment_hook": ...,
#     # additional arguments for the kinematic model hook
#     "kinematic_model_hook_args": ...
# }


###############################################################################
# Mia hand
###############################################################################

def kinematic_model_hook_mia(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    # adjust index finger limit
    joint_info = kin.tm._joints["j_index_fle"]
    joint_info = (joint_info[0], joint_info[1], joint_info[2], joint_info[3],
                  (0.0, joint_info[4][1]), joint_info[5])
    kin.tm._joints["j_index_fle"] = joint_info

    default_offsets = {
        ("thumb_tip", "thumb_fle"): [0.03, 0.07, 0.0],
        ("thumb_middle", "thumb_fle"): [0.03, 0.015, 0.0],
        ("index_tip", "index_fle"): [-0.015, 0.09, 0.0],
        ("index_middle", "index_fle"): [0.02, 0.015, 0.0],
        ("middle_tip", "middle_fle"): [-0.015, 0.09, 0.0],
        ("middle_middle", "middle_fle"): [0.02, 0.015, 0.0],
        ("ring_tip", "ring_fle"): [-0.012, 0.083, 0.0],
        ("ring_middle", "ring_fle"): [0.017, 0.015, 0.0],
        ("little_tip", "little_fle"): [-0.01, 0.068, 0.0],
        ("little_middle", "little_fle"): [0.015, 0.015, 0.0],
    }

    for key in default_offsets:
        new_frame, existing_frame = key
        position = kwargs.get(f"{new_frame}_to_{existing_frame}", default_offsets[key])
        new_frame2existing_frame = np.eye(4)
        new_frame2existing_frame[:3, 3] = position
        kin.tm.add_transform(new_frame, existing_frame, new_frame2existing_frame)


class MiaVirtualThumbJoint:
    """Positive values will be mapped to max limit, negative values to min."""
    def __init__(self, real_joint_name):
        self.real_joint_name = real_joint_name
        self.min_angle = -0.628
        self.max_angle = 0.0
        self.angle_threshold = 0.5 * (self.min_angle + self.max_angle)

    def make_virtual_joint(self, joint_name, tm):
        limits = tm.get_joint_limits(self.real_joint_name)

        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), limits, "revolute")

    def __call__(self, value):
        if value >= 0:
            angle = self.max_angle
        else:
            angle = self.min_angle
        return {self.real_joint_name: angle}


manobase2miabase = pt.transform_from_exponential_coordinates(
    [-1.006, 0.865, -1.723, -0.108, 0.088, 0.011])
MIA_CONFIG = {
    "joint_names":
        {  # map finger names to a list of joint names that control the finger
            "thumb": ["j_thumb_fle"],
            "index": ["j_index_fle"],
            "middle": ["j_mrl_fle"],
            "ring": ["j_ring_fle"],
            "little": ["j_little_fle"],
        },
    "base_frame": "palm",  # base frame of the hand
    "ee_frames":
        {  # map finger name to the name of the tip frame in the kinematic model
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2miabase,  # transform from MANO base to hand base
    # "model":  # kinematic model definition
    #     {
    #         # this xacro is actually just plain urdf:
    #         "urdf": resource_filename(
    #             "hand_embodiment",
    #             "model/mia_hand_ros_pkgs/mia_hand_description/urdf/"
    #             "mia_hand_SSSA.urdf.xacro"),
    #         "package_dir": resource_filename(
    #             "hand_embodiment", "model/mia_hand_ros_pkgs/"),
    #         "kinematic_model_hook": kinematic_model_hook_mia
    #     },

    "model": 
        {
            # 使用 os.path.join 拼接绝对路径
            "urdf": os.path.join(
                CURRENT_DIR, 
                "model/mia_hand_ros_pkgs/mia_hand_description/urdf/mia_hand_SSSA.urdf.xacro"
            ),
            "package_dir": os.path.join(
                CURRENT_DIR, 
                "model/mia_hand_ros_pkgs/"
            ),
            "kinematic_model_hook": kinematic_model_hook_mia
        },
    "virtual_joints_callbacks":
        {  # here we can introduce virtual joints that compute the state of real joints in a callback
            "j_thumb_opp_binary": MiaVirtualThumbJoint("j_thumb_opp"),
        },
    "coupled_joints":
    [  # coupled joints always have the same angle
        ("middle", "j_mrl_fle"),
        ("ring", "j_ring_fle"),
        ("little", "j_little_fle"),
    ]
}


###############################################################################
# Shadow dexterous hand
###############################################################################

def kinematic_model_hook_shadow(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "rh_thtip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.013],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "rh_thmiddle",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0],
            [0, 0, 1, 0.015],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "rh_fftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "rh_ffproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "rh_mftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "rh_mfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "rh_rftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_middle", "rh_rfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "rh_lftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "rh_lfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))


class ShadowVirtualF0Joint:
    def __init__(self, first_real_joint_name, second_real_joint_name):
        self.first_real_joint_name = first_real_joint_name
        self.second_real_joint_name = second_real_joint_name
        self.first_joint_max = 0.5 * np.pi

    def make_virtual_joint(self, joint_name, tm):
        first_limits = tm.get_joint_limits(self.first_real_joint_name)
        second_limits = tm.get_joint_limits(self.second_real_joint_name)

        joint_range = (second_limits[1] - second_limits[0] +
                       first_limits[1] - first_limits[0])
        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), (0, joint_range), "revolute")

    def __call__(self, value):
        if value > self.first_joint_max:
            first_joint_value = self.first_joint_max
            second_joint_value = value - self.first_joint_max
        else:
            first_joint_value = value
            second_joint_value = 0.0
        return {self.first_real_joint_name: first_joint_value,
                self.second_real_joint_name: second_joint_value}


manobase2shadowbase = pt.transform_from_exponential_coordinates(
    [-0.07, 1.77, -0.148, -0.309, -0.021, 0.272])
# SHADOW_HAND_CONFIG = {
#     "joint_names":
#         {  # wrist: rh_WRJ2, rh_WRJ1
#             "thumb": ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"],
#             "index": ["rh_FFJ4", "rh_FFJ3", "rh_FFJ0"],
#             "middle": ["rh_MFJ4", "rh_MFJ3", "rh_MFJ0"],
#             "ring": ["rh_RFJ4", "rh_RFJ3", "rh_RFJ0"],
#             "little": ["rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ0"],
#         },
#     "base_frame": "rh_forearm",
#     "ee_frames":
#         {
#             "thumb": "thumb_tip",
#             "index": "index_tip",
#             "middle": "middle_tip",
#             "ring": "ring_tip",
#             "little": "little_tip"
#         },
#     "intermediate_frames":
#         {
#             "thumb": "thumb_middle",
#             "index": "index_middle",
#             "middle": "middle_middle",
#             "ring": "ring_middle",
#             "little": "little_middle"
#         },
#     "handbase2robotbase": manobase2shadowbase,
#     "model":
#         {
#             "urdf": resource_filename(
#                 "hand_embodiment",
#                 "model/sr_common/sr_description/urdf/shadow_hand.urdf"),
#             "package_dir": resource_filename(
#                 "hand_embodiment", "model/sr_common/"),
#             "kinematic_model_hook": kinematic_model_hook_shadow
#         },
#     "virtual_joints_callbacks":
#         {
#             "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
#             "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
#             "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
#             "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
#         }
# }
SHADOW_HAND_CONFIG = {
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "thumb": ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"],
            "index": ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2","rh_FFJ1"],
            "middle": ["rh_MFJ4", "rh_MFJ3", "rh_MFJ2","rh_MFJ1"],
            "ring": ["rh_RFJ4", "rh_RFJ3", "rh_RFJ2","rh_RFJ1"],
            "little": ["rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2","rh_LFJ1"],
        },
    "base_frame": "rh_forearm",
    "ee_frames":
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2shadowbase,
    "model":
        {
            # "urdf": resource_filename(
            #     "hand_embodiment",
            #     "model/sr_common/sr_description/urdf/shadow_hand.urdf"),
            # "package_dir": resource_filename(
            #     "hand_embodiment", "model/sr_common/"),
            # "kinematic_model_hook": kinematic_model_hook_shadow
            "urdf": os.path.join(CURRENT_DIR,"model/sr_common/sr_description/urdf/shadow_hand.urdf"),
            "package_dir": os.path.join(CURRENT_DIR,"model/sr_common/"),
            "kinematic_model_hook": kinematic_model_hook_shadow
        },
    "virtual_joints_callbacks":
        {
            "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
            "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
            "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
            "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
        }
}

###############################################################################
# Robotiq 2F-140
###############################################################################


def kinematic_model_hook_robotiq(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "left_finger_tip", "left_inner_finger_pad",
        np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.03],
            [0, 0, 1, 0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "left_finger_middle", "left_inner_finger",
        np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.02],
            [0, 0, 1, -0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "right_finger_tip", "right_inner_finger_pad",
        np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.03],
            [0, 0, 1, 0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "right_finger_middle", "right_inner_finger",
        np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.02],
            [0, 0, 1, -0.01],
            [0, 0, 0, 1]]))


def robotiq_post_embodiment_hook(joint_angles):
    angle1 = joint_angles["thumb"][0]
    angle2 = joint_angles["index"][0]
    angle = max(angle1, angle2)
    joint_angles["thumb"][0] = angle
    joint_angles["index"][0] = angle
    return {"thumb", "index"}


class RobotiqJoint:
    def __init__(self):
        pass

    def make_virtual_joint(self, joint_name, tm):
        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), (0, 0.7), "revolute")

    def __call__(self, value):
        return {"finger_joint": value,
                "right_outer_knuckle_joint": -value,
                "left_inner_knuckle_joint": -value,
                "right_inner_knuckle_joint": -value,
                "left_inner_finger_joint": value,
                "right_inner_finger_joint": value}


manobase2robotiqbase = pt.transform_from_exponential_coordinates(
    [-0.148, 1.489, -0.881, -0.148, -0.009, 0.083])
ROBOTIQ_CONFIG = {
    "joint_names":
        {
            "thumb": ["joint"],
            "index": ["joint"],
        },
    "base_frame": "robotiq_arg2f_base_link",
    "ee_frames":
        {
            "thumb": "left_finger_tip",
            "index": "right_finger_tip",
        },
    "intermediate_frames":
        {
            "thumb": "left_finger_middle",
            "index": "right_finger_middle",
        },
    "handbase2robotbase": manobase2robotiqbase,
    "model":
        {
            "urdf": os.path.join(
                CURRENT_DIR,
                "model/robotiq_2f_140_gripper_visualization/urdf/robotiq_2f_140.urdf"),
            "mesh_path": os.path.join(
                CURRENT_DIR,
                "model/robotiq_2f_140_gripper_visualization/meshes/"),
            "kinematic_model_hook": kinematic_model_hook_robotiq
        },
    "virtual_joints_callbacks":
        {
            "joint": RobotiqJoint(),
        },
    "post_embodiment_hook": robotiq_post_embodiment_hook
}


###############################################################################
# Barret hand
###############################################################################


class OffsetJoint:
    def __init__(self, joint_name, offset, range):
        self.joint_name = joint_name
        self.offset = offset
        self.range = range

    def make_virtual_joint(self, joint_name, tm):
        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), self.range, "revolute")

    def __call__(self, value):
        return {self.joint_name: value + self.offset}


def kinematic_model_hook_barrett(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "finger_1_tip", "finger_1_dist_link",
        np.array([
            [1, 0, 0, -0.045],
            [0, 1, 0, 0.027],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "finger_1_middle", "finger_1_dist_link",
        np.array([
            [1, 0, 0, -0.01],
            [0, 1, 0, -0.01],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    """
    kin.tm.add_transform(
        "finger_2_tip", "finger_2_dist_link",
        np.array([
            [1, 0, 0, -0.045],
            [0, 1, 0, 0.027],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "finger_2_middle", "finger_2_dist_link",
        np.array([
            [1, 0, 0, -0.01],
            [0, 1, 0, -0.01],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    """
    kin.tm.add_transform(
        "finger_2_tip", "finger_2_dist_link",
        np.array([
            [1, 0, 0, -0.015],
            [0, 1, 0, -0.005],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "finger_2_middle", "finger_2_med_link",
        np.array([
            [1, 0, 0, -0.05],
            [0, 1, 0, -0.012],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "finger_3_tip", "finger_3_dist_link",
        np.array([
            [1, 0, 0, -0.045],
            [0, 1, 0, 0.027],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "finger_3_middle", "finger_3_dist_link",
        np.array([
            [1, 0, 0, -0.01],
            [0, 1, 0, -0.01],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))


manobase2barrettbase = pt.transform_from_exponential_coordinates(
    [-1.779, -0.600, 1.536, -0.033, -0.155, 0.016])

BARRETT_CONFIG = {
    "joint_names":
        {
            "thumb": ["finger_2_prox_joint", "finger_2_med_joint",
                      "finger_2_dist_joint"],
            "index": ["finger_3_med_joint", "finger_3_dist_joint"],
            "middle": ["finger_1_prox_joint_inverted", "finger_1_med_joint",
                       "finger_1_dist_joint"],
        },
    "base_frame": "base_link",
    "ee_frames":
        {
            "thumb": "finger_2_tip",
            "index": "finger_3_tip",
            "middle": "finger_1_tip",
        },
    "intermediate_frames":
        {
            "thumb": "finger_2_middle",
            "index": "finger_3_middle",
            "middle": "finger_1_middle",
        },
    "handbase2robotbase": manobase2barrettbase,
    "model":
        {
            "urdf": os.path.join(
                CURRENT_DIR, "model/barrett_hand/bhand_model.urdf"),
            "mesh_path": os.path.join(
                CURRENT_DIR, "model/barrett_hand/"),
            "kinematic_model_hook": kinematic_model_hook_barrett
        },
    "virtual_joints_callbacks":
        {
            "finger_1_prox_joint_inverted": OffsetJoint(
                "finger_1_prox_joint", -np.pi, (0, np.pi))
        }
}


###############################################################################
# Selection
###############################################################################

def kinematic_model_hook_Inspire(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "thumb_tip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.013],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "thumb_intermediate",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0],
            [0, 0, 1, 0.015],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "index_tip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "index_intermediate",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "middle_tip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "middle_intermediate",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "ring_tip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_middle", "ring_intermediate",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "pinky_tip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "pinky_intermediate",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    
# manobase2inspire = pt.transform_from_exponential_coordinates(
#     [-0.07, 1.77, -0.148, -0.309, -0.021, 0.172])
manobase2inspire = pt.transform_from_exponential_coordinates(
   [2.3, 2.184, 0.196, -0.094, -0.070, 0.151])
Inspire_Config={
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "thumb": ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint"],
            "index": ["index_proximal_joint", "index_intermediate_joint",],
            "middle": ["middle_proximal_joint", "middle_intermediate_joint"],
            "ring": ["ring_proximal_joint", "ring_intermediate_joint", ],
            "little": ["pinky_proximal_joint", "pinky_intermediate_joint"],
        },
    "base_frame": "hand_base_link",
    "ee_frames":
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2inspire,
    "model":
        {
            "urdf": os.path.join(
                CURRENT_DIR,
                "model/inspire_hand/inspire_hand_right.urdf").replace("\\", "/"),
            "mesh_path": os.path.join(
                 CURRENT_DIR,
                "model/inspire_hand/").replace("\\", "/"),
            "package_dir": os.path.join(
                CURRENT_DIR, "model/inspire_hand/").replace("\\", "/"),
            "kinematic_model_hook": kinematic_model_hook_Inspire
        },
    # "virtual_joints_callbacks":
    #     {
    #         "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
    #         "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
    #         "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
    #         "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
    #     }
}







def kinematic_model_hook_leap(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "thumb_tip_head",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.013],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "thumb_pip",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0],
            [0, 0, 1, 0.015],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "index_tip_head",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "pip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "middle_tip_head",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "pip_2",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    # kin.tm.add_transform(
    #     "ring_tip", "ring_tip_head",
    #     np.array([
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0.01],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1]]))
    # kin.tm.add_transform(
    #     "ring_middle", "pip_3",
    #     np.array([
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0.015],
    #         [0, 0, 1, 0.02],
    #         [0, 0, 0, 1]]))
    
    kin.tm.add_transform(
        "little_tip", "ring_tip_head",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "pip_3",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))


manobase2leap = pt.transform_from_exponential_coordinates(
   [-0.428, 2.111, 2.373, -0.024, -0.057, 0.003])
leap_Config={
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "index": [ "1", "2", "3"],
            "middle": [ "5","6","7"],
            "little": [ "9","10","11"],
            "thumb": ["12", "13","14","15" ],
            # "little": ["pinky_proximal_joint", "pinky_intermediate_joint"],
        },
    "base_frame": "palm_lower",
    "ee_frames":
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "little": "little_tip",
            # "ring": "ring_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            # "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2leap,
    "model":
        {
            "urdf": os.path.join(
                CURRENT_DIR,
                "model/leap_hand/leap_hand_right.urdf"),
            "mesh_path": os.path.join(
                CURRENT_DIR,
                "model/leap_hand/"),
            "package_dir": os.path.join(
                CURRENT_DIR, "model/leap_hand/"),
            "kinematic_model_hook": kinematic_model_hook_leap
        },
    # "virtual_joints_callbacks":
    #     {
    #         "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
    #         "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
    #         "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
    #         "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
    #     }
}



def kinematic_model_hook_schunk(kin, **kwargs):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "thtip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.013],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "right_hand_b",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0],
            [0, 0, 1, 0.015],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "fftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "right_hand_p",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "mftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "right_hand_o",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "rftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_middle", "right_hand_n",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "lftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "right_hand_m",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))






manobase2schunk = pt.transform_from_exponential_coordinates(
   [-2.228, -0.163, 1.907, -0.04, -0.137, 0.047])
schunk_Config={
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "thumb": ["right_hand_Thumb_Flexion", "right_hand_Thumb_Opposition", "right_hand_j3","right_hand_j4"],
            "index": ["right_hand_Index_Finger_Distal", "right_hand_Index_Finger_Proximal","right_hand_j14","right_hand_index_spread"],
            "middle": ["right_hand_Middle_Finger_Proximal", "right_hand_Middle_Finger_Distal","right_hand_j15",],
            "ring": ["right_hand_Ring_Finger", "right_hand_j12","right_hand_j16","right_hand_ring_spread"],
            "little": ["right_hand_Pinky", "right_hand_j13","right_hand_j17","right_hand_Finger_Spread"],
        },
    "base_frame": "right_hand_base_link",
    "ee_frames":
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2schunk,
    "model":
        {
            # 注意：这里的路径前面不要加斜杠 /，改成相对路径
            "urdf": os.path.join(
                CURRENT_DIR,
                "model/schunk_hand/schunk_svh_hand_right.urdf"),
            "mesh_path": os.path.join(
                CURRENT_DIR,
                "model/schunk_hand/"),
            "package_dir": os.path.join(
                CURRENT_DIR, "model/schunk_hand/"),
            "kinematic_model_hook": kinematic_model_hook_schunk
        },
    # "virtual_joints_callbacks":
    #     {
    #         "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
    #         "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
    #         "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
    #         "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
    #     }
}
TARGET_CONFIG = {
    "inspire": Inspire_Config,
    "mia": MIA_CONFIG,
    "shadow_hand": SHADOW_HAND_CONFIG,
    "shadow": SHADOW_HAND_CONFIG,
    "robotiq": ROBOTIQ_CONFIG,
    "robotiq_2f_140": ROBOTIQ_CONFIG,
    "barrett": BARRETT_CONFIG,
    "leap":leap_Config,
    'schunk':schunk_Config,

}
