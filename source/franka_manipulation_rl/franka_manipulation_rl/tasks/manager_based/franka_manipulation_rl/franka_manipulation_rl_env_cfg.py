from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    RewardsCfg as LiftRewardsCfg,
)

from . import mdp


@configclass
class StagedRewardsCfg(LiftRewardsCfg):
    """Stage-structured rewards for Franka manipulation."""

    # Stage 1: Reach
    reaching_object = RewTerm(
        func=mdp.reaching_before_lift,
        params={
            "std": 0.1,
            "minimal_height": 0.04,
        },
        weight=1.0,
    )

    # Stage 2: Grasp
    grasping_object = RewTerm(
        func=mdp.grasping_object,
        params={
            "distance_threshold": 0.08,
            "minimal_height": 0.04,
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_finger.*"],
            ),
        },
        weight=4.0,
    )

    # Stage 3: Lift
    lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )

    # Stage 4: Transport
    object_goal_tracking = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "object_pose",
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose",
        },
        weight=5.0,
    )

    # Smoothness penalties: unchanged from baseline
    action_rate = RewTerm(
        func=lift_mdp.action_rate_l2,
        weight=-1e-4,
    )

    joint_vel = RewTerm(
        func=lift_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FrankaManipulationRlEnvCfg(FrankaCubeLiftEnvCfg):
    """Official Isaac Lab reward baseline."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5


@configclass
class FrankaManipulationStagedRewardEnvCfg(FrankaManipulationRlEnvCfg):
    """Franka manipulation environment using stage-structured rewards."""

    rewards: StagedRewardsCfg = StagedRewardsCfg()