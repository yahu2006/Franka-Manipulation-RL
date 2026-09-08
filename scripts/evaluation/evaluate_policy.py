# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained Franka manipulation policy with RSL-RL."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Import the RSL-RL CLI helper from scripts/rsl_rl
# -----------------------------------------------------------------------------

RSL_RL_SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rsl_rl")
)
sys.path.insert(0, RSL_RL_SCRIPT_DIR)

import cli_args  # noqa: E402


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate a Franka manipulation policy."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Number of parallel evaluation environments.",
)

parser.add_argument(
    "--num_episodes",
    type=int,
    default=64,
    help="Total number of episodes to evaluate.",
)

parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Isaac Lab task name.",
)

parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=2026,
    help="Evaluation environment seed.",
)

parser.add_argument(
    "--lift_threshold",
    type=float,
    default=0.04,
    help="Object height threshold for lift success in meters.",
)

parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.05,
    help="Object-to-goal position threshold for goal success in meters.",
)

# RSL-RL arguments, including --checkpoint
cli_args.add_rsl_rl_args(parser)

# Isaac Sim launcher arguments, including --headless
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Clear command line for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require Isaac Sim to be launched
# -----------------------------------------------------------------------------

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

import franka_manipulation_rl.tasks  # noqa: F401


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Evaluate one trained Franka policy."""

    if args_cli.checkpoint is None:
        raise ValueError(
            "Evaluation requires --checkpoint with the full checkpoint path."
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs

    # IMPORTANT:
    # all policies will later use the same evaluation seed
    env_cfg.seed = args_cli.seed

    env_cfg.sim.device = (
        args_cli.device
        if args_cli.device is not None
        else env_cfg.sim.device
    )

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)

    env_cfg.log_dir = log_dir

    print("\n" + "=" * 80)
    print("Franka Manipulation Policy Evaluation")
    print("=" * 80)
    print(f"Task              : {args_cli.task}")
    print(f"Checkpoint        : {resume_path}")
    print(f"Evaluation seed   : {args_cli.seed}")
    print(f"Num environments  : {args_cli.num_envs}")
    print(f"Num episodes      : {args_cli.num_episodes}")
    print(f"Lift threshold    : {args_cli.lift_threshold:.3f} m")
    print(f"Goal threshold    : {args_cli.goal_threshold:.3f} m")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Create environment
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # -------------------------------------------------------------------------
    # Load trained policy
    # -------------------------------------------------------------------------

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )

    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )

    else:
        raise ValueError(
            f"Unsupported runner class: {agent_cfg.class_name}"
        )

    runner.load(resume_path)

    # RSL-RL inference policy is deterministic inference.
    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # -------------------------------------------------------------------------
    # Access scene objects
    # -------------------------------------------------------------------------

    base_env = env.unwrapped

    object_asset = base_env.scene["object"]
    robot_asset = base_env.scene["robot"]

    num_envs = env.num_envs
    device = env.device

    # Per-environment state for the CURRENT episode
    episode_lift_success = torch.zeros(
        num_envs,
        dtype=torch.bool,
        device=device,
    )

    episode_goal_success = torch.zeros(
        num_envs,
        dtype=torch.bool,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Completed episode results
    # -------------------------------------------------------------------------

    lift_results = []
    goal_results = []
    drop_results = []
    final_position_errors = []

    completed_episodes = 0

    # Wrapper resets the environment during construction.
    obs = env.get_observations()

    initial_object_z = object_asset.data.root_pos_w[:, 2]

    print(
        f"[DEBUG] Initial object z: "
        f"min={initial_object_z.min().item():.4f} m, "
        f"max={initial_object_z.max().item():.4f} m, "
        f"mean={initial_object_z.mean().item():.4f} m"
    )

    print(
        f"[DEBUG] Lift threshold: "
        f"{args_cli.lift_threshold:.4f} m"
    )

    # -------------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------------

    while (
        simulation_app.is_running()
        and completed_episodes < args_cli.num_episodes
    ):

        # ---------------------------------------------------------------------
        # Measure state BEFORE env.step().
        #
        # Isaac Lab automatically resets terminated environments inside step().
        # Therefore this is the last accessible state before a possible reset.
        # ---------------------------------------------------------------------

        object_pos_w = object_asset.data.root_pos_w

        command = base_env.command_manager.get_command("object_pose")

        desired_pos_robot = command[:, :3]

        desired_pos_w, _ = combine_frame_transforms(
            robot_asset.data.root_pos_w,
            robot_asset.data.root_quat_w,
            desired_pos_robot,
        )

        position_error = torch.linalg.norm(
            desired_pos_w - object_pos_w,
            dim=1,
        )

        # Convert current object height to the local environment frame.
        object_z_local = (
            object_pos_w[:, 2]
            - base_env.scene.env_origins[:, 2]
        )
        
        # Default object height in the local environment frame.
        initial_object_z_local = (
            object_asset.data.default_root_state[:, 2]
        )
        
        # Lift means the object has actually moved upward by the required amount.
        lifted = (
            object_z_local
            > initial_object_z_local + args_cli.lift_threshold
        )

        goal_reached = (
            lifted
            & (position_error < args_cli.goal_threshold)
        )

        # Success means "achieved at least once during this episode".
        episode_lift_success |= lifted
        episode_goal_success |= goal_reached

        # Keep the final accessible error for this step.
        last_position_error = position_error.clone()

        # ---------------------------------------------------------------------
        # Policy + environment step
        # ---------------------------------------------------------------------

        with torch.inference_mode():

            actions = policy(obs)

            obs, _, dones, extras = env.step(actions)

            policy_nn.reset(dones)

        done_mask = dones.bool()

        if not torch.any(done_mask):
            continue

        # ---------------------------------------------------------------------
        # Determine whether termination was timeout or object dropping
        # ---------------------------------------------------------------------

        if "time_outs" in extras:

            time_outs = extras["time_outs"].bool()

        elif hasattr(base_env, "reset_time_outs"):

            time_outs = base_env.reset_time_outs.bool()

        else:
            raise RuntimeError(
                "Unable to retrieve timeout information from the environment."
            )

        # For this task:
        # done AND not timeout == object_dropping
        drop_mask = done_mask & (~time_outs)

        done_indices = torch.nonzero(
            done_mask,
            as_tuple=False,
        ).squeeze(-1)

        # ---------------------------------------------------------------------
        # Record completed episodes
        # ---------------------------------------------------------------------

        for env_id in done_indices.tolist():

            if completed_episodes >= args_cli.num_episodes:
                break

            lift_results.append(
                float(episode_lift_success[env_id].item())
            )

            goal_results.append(
                float(episode_goal_success[env_id].item())
            )

            drop_results.append(
                float(drop_mask[env_id].item())
            )

            final_position_errors.append(
                float(last_position_error[env_id].item())
            )

            completed_episodes += 1

        # Reset per-episode tracking for environments that just terminated.
        episode_lift_success[done_mask] = False
        episode_goal_success[done_mask] = False

        # Progress output
        print(
            f"\r[INFO] Completed episodes: "
            f"{completed_episodes}/{args_cli.num_episodes}",
            end="",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------

    print("\n")

    lift_tensor = torch.tensor(lift_results)
    goal_tensor = torch.tensor(goal_results)
    drop_tensor = torch.tensor(drop_results)
    error_tensor = torch.tensor(final_position_errors)

    lift_success_rate = 100.0 * lift_tensor.mean().item()
    goal_success_rate = 100.0 * goal_tensor.mean().item()
    drop_rate = 100.0 * drop_tensor.mean().item()

    mean_final_error = error_tensor.mean().item()
    std_final_error = error_tensor.std(unbiased=False).item()

    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    print(
        f"Episodes                  : "
        f"{completed_episodes}"
    )

    print(
        f"Lift Success Rate         : "
        f"{lift_success_rate:.2f}%"
    )

    print(
        f"Goal Success Rate         : "
        f"{goal_success_rate:.2f}%"
    )

    print(
        f"Drop Rate                 : "
        f"{drop_rate:.2f}%"
    )

    print(
        f"Mean Final Position Error : "
        f"{mean_final_error:.4f} m"
    )

    print(
        f"Std Final Position Error  : "
        f"{std_final_error:.4f} m"
    )

    print("=" * 80)

    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()