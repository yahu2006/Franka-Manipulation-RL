from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


# =============================================================================
# Helper: relative lift state
# =============================================================================


def _object_is_lifted_relative(
    env: ManagerBasedRLEnv,
    object: RigidObject,
    minimal_height: float,
) -> torch.Tensor:
    """Check whether the object has been lifted relative to its initial height.

    The official Lift task uses an absolute world-z threshold. However, the cube
    already starts above z=0 because it is resting on the table.

    Here, "lifted" means:

        current local z > initial local z + minimal_height
    """

    # Current object height in the local environment frame.
    object_z_local = (
        object.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
    )

    # Default cube height before lifting.
    initial_z_local = object.data.default_root_state[:, 2]

    return object_z_local > (
        initial_z_local + minimal_height
    )


# =============================================================================
# Reach reward
# =============================================================================


def reaching_before_lift(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the end-effector for approaching the object before lifting."""

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.linalg.norm(
        object_pos_w - ee_pos_w,
        dim=1,
    )

    reach_reward = 1.0 - torch.tanh(
        distance / std
    )

    lifted = _object_is_lifted_relative(
        env,
        object,
        minimal_height,
    )

    not_lifted = ~lifted

    return (
        not_lifted.float()
        * reach_reward
    )


# =============================================================================
# Grasp alignment reward
# =============================================================================


def grasp_alignment_before_lift(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=[
            "panda_leftfinger",
            "panda_rightfinger",
        ],
    ),
) -> torch.Tensor:
    """Reward both Franka fingers for approaching the cube before lifting.

    This avoids rewarding gripper closure itself, which previously created
    a reward-hacking behavior where the policy could close its fingers near
    the cube without actually lifting it.
    """

    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    object_pos_w = object.data.root_pos_w

    # Shape:
    # [num_envs, 2 fingers, xyz]
    finger_pos_w = robot.data.body_pos_w[
        :, robot_cfg.body_ids, :
    ]

    # Distance from each finger to the cube.
    finger_dist = torch.linalg.norm(
        finger_pos_w
        - object_pos_w.unsqueeze(1),
        dim=-1,
    )

    # Average distance of the two fingers.
    mean_finger_dist = torch.mean(
        finger_dist,
        dim=1,
    )

    alignment_reward = 1.0 - torch.tanh(
        mean_finger_dist / std
    )

    lifted = _object_is_lifted_relative(
        env,
        object,
        minimal_height,
    )

    not_lifted = ~lifted

    return (
        not_lifted.float()
        * alignment_reward
    )


# =============================================================================
# Lift reward
# =============================================================================


def object_is_lifted_relative(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the policy once the cube is truly lifted above its start height."""

    object: RigidObject = env.scene[object_cfg.name]

    lifted = _object_is_lifted_relative(
        env,
        object,
        minimal_height,
    )

    return lifted.float()


# =============================================================================
# Goal / transport reward
# =============================================================================


def object_goal_distance_relative(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward transporting the lifted cube toward the commanded target.

    The goal reward is active only after the object has actually been lifted
    relative to its initial table height.
    """

    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Target pose generated by the command manager.
    command = env.command_manager.get_command(
        command_name
    )

    # Desired object position expressed in the robot-base frame.
    desired_pos_b = command[:, :3]

    # Transform target position into world coordinates.
    desired_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        desired_pos_b,
    )

    # Distance between cube and desired position.
    distance = torch.linalg.norm(
        desired_pos_w
        - object.data.root_pos_w[:, :3],
        dim=1,
    )

    tracking_reward = 1.0 - torch.tanh(
        distance / std
    )

    lifted = _object_is_lifted_relative(
        env,
        object,
        minimal_height,
    )

    return (
        lifted.float()
        * tracking_reward
    )