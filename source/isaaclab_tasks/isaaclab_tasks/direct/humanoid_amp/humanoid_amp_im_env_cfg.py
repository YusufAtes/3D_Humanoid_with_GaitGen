# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# from .custom_terrains import slope_terrain, noisy_terrain
# from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
# import isaaclab.sim as sim_utils

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 83
    action_space = 28
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.7
    
    test_mode = None # noisy or rotation
    motion_file: str = r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\motions\humanoid_walk.npz"
    reference_body = "torso"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )

    # if test_mode != None:
    #     terrain = TerrainImporterCfg(
    #             prim_path="/World/ground",
    #             terrain_type="generator",
    #             terrain_generator=TerrainGeneratorCfg(
    #                 seed=42,
    #                 curriculum=True, # Set False if you don't want difficulty to change
                    
    #                 # Map parameters: Total size of the terrain grid
    #                 size=(20.0, 20.0), 
    #                 border_width=2.5,
    #                 num_rows=4,
    #                 num_cols=4,
    #                 horizontal_scale=0.1, # 1 pixel = 0.1 meters
    #                 vertical_scale=0.005, # Height precision
                    
    #                 sub_terrains={
    #                     # 1. SLOPED TERRAIN
    #                     "sloped_ramp": TerrainGeneratorCfg.SubTerrainConfig(
    #                         proportion=0.5, # 50% of the world is slopes
    #                         function=slope_terrain,
    #                         args={
    #                             "slope_angle_deg": 15.0 # Max angle (at difficulty=1.0)
    #                         }
    #                     ),
                        
    #                     # 2. NOISY TERRAIN
    #                     "noisy_ground": TerrainGeneratorCfg.SubTerrainConfig(
    #                         proportion=0.5, # 50% of the world is noisy
    #                         function=noisy_terrain,
    #                         args={
    #                             "amplitude_scale": 0.5, # Max noise height = 0.5m
    #                             # "base_heightfield": my_numpy_array # Optional: pass specific data
    #                         }
    #                     )
    #                 }
    #             ),
    #             max_init_terrain_level=5,
    #             collision_group=-1,
    #             physics_material=sim_utils.RigidBodyMaterialCfg(
    #                 static_friction=1.0,
    #                 dynamic_friction=1.0,
    #                 restitution=0.0,
    #             ),
    #             debug_vis=False,
    #         )


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
