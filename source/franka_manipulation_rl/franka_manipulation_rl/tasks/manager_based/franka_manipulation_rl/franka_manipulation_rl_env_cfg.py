from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)


@configclass
class FrankaManipulationRlEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka manipulation environment based on the official Lift-Cube task."""

    def __post_init__(self):
        super().__post_init__()

        # Keep a moderate number of environments for development/debugging.
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5