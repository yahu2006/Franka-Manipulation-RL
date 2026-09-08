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

def grasp_alignment_before_lift(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["panda_leftfinger", "panda_rightfinger"],
    ),
) -> torch.Tensor:
    """Reward both Franka fingers approaching the object before lifting."""

    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    object_pos_w = object.data.root_pos_w

    # Shape: [num_envs, 2, 3]
    finger_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]

    # Distance from each finger to the cube.
    finger_dist = torch.linalg.norm(
        finger_pos_w - object_pos_w.unsqueeze(1),
        dim=-1,
    )

    mean_finger_dist = torch.mean(finger_dist, dim=1)

    alignment_reward = 1.0 - torch.tanh(mean_finger_dist / std)

    not_lifted = object_pos_w[:, 2] <= minimal_height

    return not_lifted.float() * alignment_reward