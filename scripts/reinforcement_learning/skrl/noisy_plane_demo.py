# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate an RL agent on a noisy ground plane.

For each (noise_type, noise_seed, downsampled_scale, noise_amplitude) tuple, the
script iterates over a range of desired speeds and logs actual speed /
reward / success to a CSV file. Results are averaged across seeds offline.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
from isaaclab.app import AppLauncher
import time

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl on a noisy plane.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="AMP",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--noise_amplitude", type=float, default=0.0, help="Max height perturbation (metres) for the noisy plane.")
parser.add_argument("--noise_type", type=str, default="random", choices=["random", "wave"], help="Type of noisy terrain: 'random' or 'wave'.")
parser.add_argument("--noise_seed", type=int, default=42, help="Seed for reproducible heightfield realization.")
parser.add_argument(
    "--downsampled_scale",
    type=float,
    default=None,
    help="Distance between random samples for random terrain (m). None uses terrain default.",
)
parser.add_argument(
    "--noise_step",
    type=float,
    default=None,
    help="Deprecated alias for downsampled_scale. Ignored if --downsampled_scale is provided.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import csv
import pandas as pd
import matplotlib.pyplot as plt

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent (multi-speed sweep on a noisy ground plane)."""

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    noise_amplitude = args_cli.noise_amplitude
    noise_seed = args_cli.noise_seed
    downsampled_scale = args_cli.downsampled_scale
    if downsampled_scale is None:
        downsampled_scale = args_cli.noise_step

    # ---------- configure env for noisy-plane demo ----------
    env_cfg.demo_type = "noise"
    env_cfg.noise_amplitude = noise_amplitude
    env_cfg.noise_type = args_cli.noise_type
    env_cfg.noise_seed = noise_seed
    env_cfg.downsampled_scale = downsampled_scale
    env_cfg.test_slope_deg = 0.0          # flat base — noise only
    env_cfg.episode_length_s = 5
    env_cfg.termination_height = 0.55

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # ---------------------------
    # MULTI-SPEED DEMO SETTINGS
    # ---------------------------
    speed_schedule = np.linspace(0.2, 2.4, 23)  # same schedule as ramp demo
    episode_current_vel = []
    results_data = []  # rows for CSV

    # Keep these constant across runs
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # downsampled_scale is meaningful only for 'random'; record NaN for 'wave'
    logged_downsampled_scale = downsampled_scale if args_cli.noise_type == "random" else float("nan")

    # ---------------------------
    # CREATE ENV ONCE
    # ---------------------------
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(base_env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        base_env = multi_agent_to_single_agent(base_env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = base_env.step_dt
    except AttributeError:
        dt = base_env.unwrapped.step_dt

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_folder, exist_ok=True)

        video_kwargs = {
            "video_folder": video_folder,
            "episode_trigger": lambda episode_id: True,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "noise_init",
            "fps": 30,
        }
        print("[INFO] Recording videos during play (one video per speed).")
        print_dict(video_kwargs, nesting=4)

    # Now wrap for skrl
    env = SkrlVecEnvWrapper(base_env, ml_framework=args_cli.ml_framework)
    wrapped_env = env.env

    # Configure and instantiate the skrl runner ONCE
    runner = Runner(env, experiment_cfg)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")
    print("[INFO] Agent set to eval mode")

    # ---------------------------
    # RUN ONE EPISODE PER SPEED
    # ---------------------------
    t0 = time.time()
    for desired_speed in speed_schedule:
        if not simulation_app.is_running():
            print("[WARN] simulation_app not running; exiting.")
            break

        # set speed BEFORE reset so env picks it up in _reset_idx
        env._unwrapped.set_test_speed(desired_speed)

        # reset => starts a new episode
        obs, _ = env.reset()
        demo_step = int(env.unwrapped.episode_length_buf.item())
        reward_sum = 0.0
        print("------------------------------------------------------------")
        while True:
            start_time = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)

                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, rews, done, _, _ = env.step(actions)
                # safe reward sum
                if isinstance(rews, torch.Tensor):
                    reward_sum += float(rews.sum().item())
                elif isinstance(rews, np.ndarray):
                    reward_sum += float(rews.sum())
                else:
                    reward_sum += float(rews)

            demo_step = int(wrapped_env.unwrapped.episode_length_buf.item())
            if demo_step == 148:
                avg_speed = wrapped_env.unwrapped.avg_speed
                print(f"type={args_cli.noise_type} seed={noise_seed} downsampled_scale={logged_downsampled_scale} "
                      f"amp={noise_amplitude} desired={desired_speed:.2f} actual={avg_speed:.3f} success=True")
                print("------------------------------------------------------------")
                episode_current_vel.append(avg_speed)
                results_data.append({
                    'desired_speed': desired_speed,
                    'actual_speed': avg_speed,
                    'reward': reward_sum,
                    'x_pos': wrapped_env.unwrapped.x_pos,
                    'noise_amplitude': noise_amplitude,
                    'noise_type': args_cli.noise_type,
                    'noise_seed': noise_seed,
                    'downsampled_scale': logged_downsampled_scale,
                    'success': True,
                })
                break

            if done == True:
                avg_speed = wrapped_env.unwrapped.avg_speed
                print(f"type={args_cli.noise_type} seed={noise_seed} downsampled_scale={logged_downsampled_scale} "
                      f"amp={noise_amplitude} desired={desired_speed:.2f} actual={avg_speed:.3f} success=False")
                print("------------------------------------------------------------")
                episode_current_vel.append(avg_speed)
                results_data.append({
                    'desired_speed': desired_speed,
                    'actual_speed': avg_speed,
                    'x_pos': wrapped_env.unwrapped.x_pos,
                    'reward': reward_sum,
                    'noise_amplitude': noise_amplitude,
                    'noise_type': args_cli.noise_type,
                    'noise_seed': noise_seed,
                    'downsampled_scale': logged_downsampled_scale,
                    'success': False,
                })
                break

    print("[INFO] Finished all speeds.")
    print(f"Time taken for type={args_cli.noise_type} seed={noise_seed} downsampled_scale={logged_downsampled_scale} "
          f"amp={noise_amplitude}: {time.time() - t0:.1f} s")

    # Save results to CSV file (append mode — write header only if file is new)
    csv_file_path = os.path.join(log_dir, "noisy_plane_demo_results.csv")
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['desired_speed', 'actual_speed', 'reward', 'x_pos',
                      'noise_amplitude', 'noise_type', 'noise_seed', 'downsampled_scale', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results_data)

    print(f"[INFO] Results saved to: {csv_file_path}")

    env.close()


if __name__ == "__main__":
    main()
    print("Closing simulation app")
    simulation_app.close()
    print("Simulation app closed")