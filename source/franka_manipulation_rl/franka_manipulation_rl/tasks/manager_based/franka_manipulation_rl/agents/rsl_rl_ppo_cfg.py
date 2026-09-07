from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LiftCubePPORunnerCfg,
)


@configclass
class FrankaManipulationPPORunnerCfg(LiftCubePPORunnerCfg):
    """PPO configuration for the Franka manipulation task."""

    experiment_name = "franka_manipulation_rl"