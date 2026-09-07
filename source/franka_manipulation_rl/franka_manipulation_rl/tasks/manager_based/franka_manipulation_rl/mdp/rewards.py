from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def reaching_before_lift(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reaching reward that is active only before the object is lifted."""

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.linalg.norm(object_pos_w - ee_pos_w, dim=1)

    reach_reward = 1.0 - torch.tanh(distance / std)

    not_lifted = object_pos_w[:, 2] <= minimal_height

    return not_lifted.float() * reach_reward


def grasping_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["panda_finger.*"]
    ),
) -> torch.Tensor:
    """Reward closing the gripper when the end effector is close to the object."""

    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.linalg.norm(object_pos_w - ee_pos_w, dim=1)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    mean_finger_pos = torch.mean(finger_pos, dim=1)

    # Franka finger joint:
    # open ≈ 0.04 m
    # closed ≈ 0.00 m
    closure = 1.0 - torch.clamp(mean_finger_pos / 0.04, 0.0, 1.0)

    near_object = distance < distance_threshold
    not_lifted = object_pos_w[:, 2] <= minimal_height

    return near_object.float() * not_lifted.float() * closure