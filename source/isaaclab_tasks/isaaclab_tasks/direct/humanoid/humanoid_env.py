# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg   # <-- add this
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg, HfWaveTerrainCfg
from isaaclab.utils import configclass


from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


# ---------------------------------------------------------------------------
# Noisy terrain generator — only random rough sub-terrain, 20 m × 20 m
# noise_range is a placeholder; it is overridden from noise_amplitude in
# _setup_scene before the TerrainImporter is instantiated.
# ---------------------------------------------------------------------------
NOISY_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    color_scheme="height",
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.05, 0.05),  # overridden by noise_amplitude at runtime
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 4
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 87
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # Demo mode settings
    demo_mode: bool = True
    demo_type: str = "vel"   # Possible choices are vel, ramp, and noise
    test_slope_deg: float = 0.0
    
    # Noisy plane demo settings
    noise_amplitude: float = 0.05  # Max height perturbation (meters) for noisy plane demo
    noise_seed: int = 42           # Seed for reproducible noise pattern across trials
    noise_type: str = "random"     # "random" or "wave"
    
    # Terrain configuration
    # When demo_type == "noise", terrain_type will be set to "generator" in _setup_scene
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    early_termination: bool = True

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )
    #camera viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.0, 6.0, 0.0),        # move to LEFT (y=3.0), not behind; lower Z=1.0
        lookat=(0.0, 0.0, 0.0),      # look at torso but from the side
        origin_type="asset_root",    # camera follows robot
        asset_name="robot",
        env_index=0,
        resolution=(1280, 720),
    )
    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    heading_weight: float = 1.0
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 1.0
    dof_vel_scale: float = 0.1

    death_cost: float = -100.0
    termination_height: float = 0.6

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class HumanoidEnv(LocomotionEnv):
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
