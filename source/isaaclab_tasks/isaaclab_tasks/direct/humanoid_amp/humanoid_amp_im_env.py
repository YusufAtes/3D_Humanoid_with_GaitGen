# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import math
from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_amp_im_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader
from .gait_generator_net import SimpleFCNN
from scipy.signal import resample
import time

class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = 0.5 * (dof_upper_limits - dof_lower_limits)

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # # --------------------------    Gait Generator System    --------------------------#

        # self.gaitgen_net = SimpleFCNN()
        # self.gaitgen_net.load_state_dict(torch.load(rf'C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\FINAL_BEST_MODEL.pth',weights_only=True))
        # self.gaitgen_net.to(self.sim.device)
        # self.gaitgen_net.eval()

        # self.mean = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\mean.npy")
        # self.std = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\std.npy")

        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load(rf'C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\locomotion\FINAL_BEST_MODEL.pth',weights_only=True))
        self.gaitgen_net.to(self.sim.device)
        self.gaitgen_net.eval()

        self.mean = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\gait reference phase 2\mean.npy")
        self.std = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\gait reference phase 2\std.npy")

        self.mean = torch.tensor(self.mean, device=self.sim.device, dtype=torch.float32)
        self.std = torch.tensor(self.std, device=self.sim.device, dtype=torch.float32)

        self.leg_len = 0.83
        self.hip_knee = 0.4
        self.knee_ankle = 0.39
        self.dt = self.step_dt 
        self.episode_len = 32 / (10*self.dt)

        self.reference = torch.zeros(
            (self.num_envs, 96, 4),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.periods = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.int,
        )

        self.phase = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.phase_step = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.int,
        )

        self.desired_fwd_speeds = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.float32,
        )

        # TESTING SETTING
        self.test_fwd_speed: float | None = None
        self.test_ang_speed: float | None = None
        self.demo_mode= self.cfg.demo_mode
        if self.demo_mode:
            self.demo_type = self.cfg.demo_type
        else:
            self.demo_type = None
        self.random_start_idx = torch.zeros((self.num_envs,),device=self.sim.device,dtype=self.episode_length_buf.dtype) 
        self.avg_speed = 0
        self.test_slope_deg = self.cfg.test_slope_deg # Load from config
        self.ramp_demo = False
        if self.test_slope_deg != 0:
            self.ramp_demo = True

        ## NOTE FOR AI AGENT THE FOLLOWING CURRICULUM SYSTEM WAS ORIGINALLY IMPLEMENTED FOR TURNING BEHAVIOR. IT IS NOT USED IN THIS ENVIRONMENT. BUT IT IS LEFT FROR REFERENCE FOR FUTURE FORWARD
        ## SPEED CURRICULUM IMPLEMENTATION.
        # # --------------------------    Curriculum System    --------------------------#
        self._curriculum_ang_half_range = 0.4
        self._curriculum_max_half_range = 1.0
        self._curriculum_step = 0.1
        self._curriculum_threshold = 0.45
        self._curriculum_N = 60000
        self._curriculum_ep_buf = torch.zeros(self._curriculum_N, device=self.sim.device)
        self._curriculum_ep_write_idx = 0
        self._curriculum_ep_count = 0
        self._yaw_reward_sum = torch.zeros(self.num_envs, device=self.sim.device)
        self._yaw_reward_steps = torch.zeros(self.num_envs, device=self.sim.device)
        self._yaw_vel_weight = 0.8
        self._yaw_vel_weight_max = 1.4

    def set_test_speed(self, fwd_speed: float | None):
        """If speed is not None, all envs will use this fixed speed at reset."""
        if fwd_speed is None:
            self.test_fwd_speed = None
        else:
            self.test_fwd_speed = float(fwd_speed)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        if self.cfg.demo_mode and self.cfg.demo_type == "noise" and self.cfg.noise_amplitude > 0.0:
            from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg, HfWaveTerrainCfg
            amp = self.cfg.noise_amplitude
            self.cfg.terrain.terrain_generator.seed = self.cfg.noise_seed
            # Select sub-terrain type based on noise_type
            noise_type = getattr(self.cfg, "noise_type", "random")
            if noise_type == "wave":
                self.cfg.terrain.terrain_generator.sub_terrains = {
                    "wave_terrain": HfWaveTerrainCfg(
                        proportion=1.0,
                        amplitude_range=(amp, amp),
                        num_waves=4,
                        border_width=0.25,
                    ),
                }
            else:  # "random" (default)
                self.cfg.terrain.terrain_generator.sub_terrains = {
                    "random_rough": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        noise_range=(-amp, amp),
                        noise_step=0.005,
                        border_width=0.25,
                    ),
                }
            # Import the noisy terrain via TerrainImporter (same pattern as AnymalC)
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )

            if self.cfg.demo_mode:
                if self.cfg.demo_type == "ramp":
                    slope = float(self.cfg.test_slope_deg)
                    if slope != 0.0:
                        stage = self.sim.stage
                        prim = stage.GetPrimAtPath("/World/ground")
                        if prim.IsValid():
                            xform = UsdGeom.Xformable(prim)
                            xform.ClearXformOpOrder()
                            rot_op = xform.AddRotateXYZOp()
                            # Rotate around Y axis for slope
                            rot_op.Set(Gf.Vec3f(0.0, slope, 0.0))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        torso_quat = self.robot.data.body_quat_w[:, self.ref_body_index] 

        # # --------------------------    Gait Generator System    --------------------------#
        if self.reference is not None:
            self.phase_step = (self.episode_length_buf + self.random_start_idx) % self.periods
            self.phase = self.phase_step / self.periods

            obs = torch.cat(
                (
                    obs,
                    self.phase.unsqueeze(1),
                    self.desired_fwd_speeds.unsqueeze(1)
                ),
                dim=-1
                )
        # # --------------------------    Gait Generator System    --------------------------#

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]

        # --- WORLD-FRAME AMP obs ---
        self.amp_observation_buffer[:, 0] = obs[:,:81].clone()

        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        #config_name: decayfactor_trainno
        total_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        imitation_reward =  torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        # decay_factor = 0.75
        # imitation_coeff = torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        # imitation_coeff = torch.where(self.desired_fwd_speeds > decay_factor, imitation_coeff * torch.exp( -1.0 *  (self.desired_fwd_speeds - decay_factor)), imitation_coeff)

        # First training: straight walking only, no yaw tracking.
        imitation_weight_hip_pos  = 0.375 
        imitation_weight_knee_pos = 0.375
        fwd_vel_weight = 0.70
        yaw_vel_weight = 0.15
        lat_vel_weight = 0.15
        death_cost = -1.75

        env_ids = torch.arange(self.reference.shape[0], device=self.sim.device)

        current_ref_pos   = self.reference[env_ids,self.phase_step,:]           # [N,6]

        # hips
        hip_joint_pos = self.robot.data.joint_pos[:, [12, 15]]          # [N, 2]
        hip_ref_pos = current_ref_pos[:,[0,2]]
        hip_diff      = hip_joint_pos - hip_ref_pos          # [N, 2]
        hip_dist      = torch.norm(hip_diff, p=2, dim=-1)    # [N]
        imitation_reward = imitation_reward + imitation_weight_hip_pos * torch.exp(-5.0 * hip_dist)

        # knees
        knee_joint_pos = self.robot.data.joint_pos[:, [20, 21]]         # [N, 2]
        knee_ref_pos   = current_ref_pos[:,[1,3]]
        knee_diff      = knee_joint_pos - knee_ref_pos
        knee_dist      = torch.norm(knee_diff, p=2, dim=-1)
        imitation_reward = imitation_reward + imitation_weight_knee_pos * torch.exp(-5.0 * knee_dist)

        # Body velocity
        vel_b = self.robot.data.root_com_lin_vel_b  # [N, 3]
        vel_ang = self.robot.data.root_com_ang_vel_b  # [N, 3]

        # Decompose
        forward_speed  = vel_b[:, 0]   # along robot-forward (x)
        lateral_speed  = vel_b[:, 1]   # sideways (y)
        yaw_speed = vel_ang[:, 2]   # yaw (w)

        vel_reward_fwd = torch.exp(-4.0 * torch.abs(forward_speed - (self.desired_fwd_speeds * 2.4)))
        vel_reward = fwd_vel_weight * vel_reward_fwd

        vel_reward += yaw_vel_weight * torch.exp(-4.0 * torch.abs(yaw_speed))

        lat_vel_reward = lat_vel_weight * torch.exp(-4.0 * torch.abs(lateral_speed))
        vel_reward += lat_vel_reward

        total_reward +=  imitation_reward + vel_reward

        death_penalty = (self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height).float() * death_cost
        total_reward +=  death_penalty
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1

            # --------------------------------------------------------
            # DEMO MODE METRICS
            # --------------------------------------------------------
            if self.demo_mode:
                x_pos = self.robot.data.body_pos_w[0, self.ref_body_index, 0].item()
                step = int(self.episode_length_buf.item())
                t = step * float(self.dt)
                self.current_t = t
                if step <= 30:
                    self.start_t = t
                    self.start_x_pos = x_pos
                else:
                    self.avg_speed = (x_pos - self.start_x_pos) / (t - self.start_t)

            # --------------------------------------------------------
            # TERMINATION LOGIC (RAMP AWARE)
            # --------------------------------------------------------
            if self.cfg.early_termination:
                # 1. Get Robot Root positions
                # Assumes self.ref_body_index is the torso/root
                root_y = self.robot.data.body_pos_w[:, self.ref_body_index, 0] 
                root_z = self.robot.data.body_pos_w[:, self.ref_body_index, 2]

                # 2. Calculate Slope Angle in Radians
                # We create a tensor on the same device to ensure GPU compatibility
                slope_deg = torch.tensor(self.cfg.test_slope_deg, device=self.device)
                slope_rad = torch.deg2rad(slope_deg)

                # 3. Calculate Floor Z at the current Y position
                # Formula: z_floor = y * tan(theta)
                floor_z = root_y * torch.tan(-slope_rad)
                # print(f"Floor Z: {floor_z}")
                # 4. Calculate Relative Height
                # This is the height of the robot ABOVE the calculated inclined floor
                relative_height = root_z - floor_z
                # 5. Check Termination
                died = relative_height < self.cfg.termination_height
                
                # Handle Demo Mode Stats if died
                if self.demo_mode and died:
                    x_pos = self.robot.data.body_pos_w[0, self.ref_body_index, 0].item()
                    step = int(self.episode_length_buf.item())
                    t = step * float(self.dt)
                    if t > 0:
                        self.avg_speed = x_pos / t
            else:
                died = torch.zeros_like(time_out)

            return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = self.robot._ALL_INDICES
            self.robot.reset(env_ids)
            super()._reset_idx(env_ids)

            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)

            # # --------------------------    Gait Generator System    --------------------------#
            num_reset = len(env_ids)
            device = self.sim.device

            if self.demo_mode and (self.test_fwd_speed is not None):
                # Move the episode length buffer reset outside the loop so it only runs once
                self.episode_length_buf[env_ids] = 0

                # Loop kept ONLY for the demo mode case, as requested
                for i, env_id in enumerate(env_ids):
                    self.desired_fwd_speeds[env_id] = self.test_fwd_speed / 2.4
                    self.reference_fwd_speed = self.test_fwd_speed / 2.4
                    encoder_vec = torch.empty((3), device=device)   
                    encoder_vec[0] = self.reference_fwd_speed
                    encoder_vec[1] = self.leg_len / 1.0
                    encoder_vec[2] = self.leg_len / 1.0

                    self.reference[env_id,:], self.periods[env_id] = self.findgait(encoder_vec) 
                    
                    self.phase_step[env_id] = 0             # fix this part it should not be 0 it should be closest index to 0 degree
                    self.reference[env_id, ..., [0,2]]  = torch.clamp(self.reference[env_id, ..., [0,2]] , -np.pi/2, np.pi/2)     # Clip the hip angle
                    self.reference[env_id, ..., [1,3]]  = torch.clamp(self.reference[env_id, ..., [1,3]] , -np.pi, 0)    
                    root_state[i, 2] += 0.4

            elif self.demo_mode and (self.test_fwd_speed is None):
                raise ValueError("NO TEST SPEED IS DEFINED!!!!!")

            else:
                # -------------------------- VECTORIZED RL RESET -------------------------- #
                # 1. Sample random values for the entire batch
                # PyTorch rand generates [0, 1). We scale it to [0.1, 2.4)
                reference_fwd_speed = (torch.rand(num_reset, device=device) * 2.3 + 0.1) / 2.4
                
                # int(3/self.dt) bounds
                max_idx = int(3 / self.dt)
                random_idx = torch.randint(0, max_idx, (num_reset,), device=device)

                self.desired_fwd_speeds[env_ids] = reference_fwd_speed

                # 2. Build batched encoder_vec: Shape (num_reset, 3)
                encoder_vec = torch.empty((num_reset, 3), device=device)   
                encoder_vec[:, 0] = reference_fwd_speed
                encoder_vec[:, 1] = self.leg_len / 1.0
                encoder_vec[:, 2] = self.leg_len / 1.0

                # 3. Call findgait in batch mode (Requires self.findgait to support N-dim tensors)
                batch_references, batch_periods = self.findgait(encoder_vec)
                self.reference[env_ids, :] = batch_references
                self.periods[env_ids] = batch_periods
                
                # 4. Compute phase steps
                self.phase_step[env_ids] = (random_idx % self.periods[env_ids]).to(self.phase_step.dtype)
                
                # 5. Clip the angles
                # Clip Hip angles (indices 0 and 2) independently
                self.reference[env_ids, ..., 0] = torch.clamp(self.reference[env_ids, ..., 0], -np.pi/2, np.pi/2)
                self.reference[env_ids, ..., 2] = torch.clamp(self.reference[env_ids, ..., 2], -np.pi/2, np.pi/2)

                # Clip Knee angles (indices 1 and 3) independently
                self.reference[env_ids, ..., 1] = torch.clamp(self.reference[env_ids, ..., 1], -np.pi, 0)
                self.reference[env_ids, ..., 3] = torch.clamp(self.reference[env_ids, ..., 3], -np.pi, 0)
                
                self.random_start_idx[env_ids] = self.phase_step[env_ids].to(self.random_start_idx.dtype)

                # 6. Extract positions using advanced indexing
                # phase_steps is 1D tensor of shape (num_reset,)
                phase_steps = self.phase_step[env_ids]

                rhip_pos = self.reference[env_ids, phase_steps, 0]
                rknee_pos = self.reference[env_ids, phase_steps, 1]
                lhip_pos = self.reference[env_ids, phase_steps, 2]
                lknee_pos = self.reference[env_ids, phase_steps, 3]

                # 7. Unused in original, but vectorized here: hip_short, knee_short calculations
                # torch.where acts like an element-wise ternary operator (if-else)
                hip_rad = torch.where(torch.abs(rhip_pos) > torch.abs(lhip_pos), lhip_pos, rhip_pos)
                knee_rad = torch.where(torch.abs(rknee_pos) > torch.abs(lknee_pos), lknee_pos, rknee_pos)

                hip_short = self.hip_knee - (torch.cos(hip_rad) * self.hip_knee)
                knee_short = self.knee_ankle - (torch.cos(knee_rad) * self.knee_ankle)
                
                sum_short = hip_short + knee_short
                total_short = torch.where(sum_short > 0.05, sum_short - 0.03, torch.zeros_like(sum_short))

                # 8. RSI ON - Update joint positions in bulk
                # Note: joint_pos has shape (num_reset, num_joints), so we use `:` to select all reset envs
                joint_pos[:, 12] = rhip_pos
                joint_pos[:, 20] = rknee_pos
                joint_pos[:, 15] = lhip_pos
                joint_pos[:, 21] = lknee_pos

                # Update root height in bulk
                root_state[:, 2] += 0.4

            # # --------------------------    Gait Generator System    --------------------------#
            self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
            self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15 
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    @staticmethod
    def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """
        Extract yaw (rotation around world Z) from quaternion.
        IsaacLab body_quat_w uses (w, x, y, z) ordering.
        Returns shape [N].
        """
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)          # [-π, π]

    # ──────────────────────────────────────────────────────────────────────
    # HELPER: wrap angle to [-π, π]
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle)) / torch.pi

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # --- WORLD-FRAME AMP obs ---
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        return amp_observation.view(-1, self.amp_observation_size)

    def findgait(self, input_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if input_vec.ndim == 1:
                input_vec = input_vec.unsqueeze(0)        # [1, D]

            # 1. Forward Pass (Already batched)
            freqs = self.gaitgen_net(input_vec)

            # 2. Slice for the entire batch
            preds = freqs[:, :136]       # [N, 136]
            periods = freqs[:, 136]      # [N]

            # 3. Apply transformations across the batch
            preds = self.denormalize(preds, self.mean, self.std)      
            preds = self.recover_shape(preds)
            pred_torch = self.pred_ifft(preds)           
            
            # 4. Process periods across the batch
            periods = (periods * self.episode_len).to(torch.int32)
            periods = torch.clamp(periods, min=12)
            
            return pred_torch, periods 

    def denormalize(self, pred_flat: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # NOTE: Ensure self.mean and self.std are PyTorch tensors on the same device as pred_flat
        return (pred_flat * std) + mean

    def recover_shape(self, flat_data: torch.Tensor) -> torch.Tensor:
        # flat_data shape: [N, 136]
        # view() is PyTorch's equivalent to reshape()
        recovered = flat_data.view(-1, 17, 4, 2) 
        
        # permute() handles multidimensional transpositions
        # Original (1, 2, 0) on [17, 4, 2] becomes (0, 2, 3, 1) on [N, 17, 4, 2]
        # Resulting shape: [N, 4, 2, 17]
        structured = recovered.permute(0, 2, 3, 1) 
        return structured

    def pred_ifft(self, predictions: torch.Tensor) -> torch.Tensor:
        # predictions shape: [N, 4, 2, 17]
        # Combine real (index 0) and imaginary (index 1) parts
        complex_pred = torch.complex(predictions[:, :, 0, :], predictions[:, :, 1, :]) 
        
        org_rate = 10
        
        # Batched Inverse FFT over the last dimension (dim=2)
        # Resulting shape: [N, 4, 32]
        pred_time = torch.fft.irfft(complex_pred, n=32, dim=2) 
        
        if self.dt < 0.1:
            num_samples = int((pred_time.shape[2]) * (1.0 / self.dt) / org_rate)
            
            # Use PyTorch 1D interpolation for resampling. 
            # F.interpolate expects shape [Batch, Channels, Length], which perfectly matches [N, 4, 32]
            pred_time = F.interpolate(pred_time, size=num_samples, mode='linear', align_corners=True)
        
        # Transpose time and channel dimensions to get final shape [N, TimeSteps, 4]
        pred_time = pred_time.permute(0, 2, 1)
        
        return pred_time
    
    def find_closest_index(self, reference, period):
        if not (self.demo_mode and self.test_fwd_speed is not None):
            raise ValueError("No test speed provided")
        
        rhip = reference[:period, 0]  # right hip trajectory
        
        # Find zero-crossings: sign changes from negative to positive
        # (extension → flexion = start of swing = double support transition)
        signs = torch.sign(rhip)
        crossings = (signs[:-1] < 0) & (signs[1:] >= 0)
        crossing_indices = torch.where(crossings)[0]
        
        if len(crossing_indices) > 0:
            # Linear interpolation for sub-index precision
            idx = crossing_indices[0].item()
            v0, v1 = rhip[idx].item(), rhip[idx + 1].item()
            frac = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0
            return int(round(idx + frac))
        else:
            # Fallback: closest to zero overall
            return torch.argmin(torch.abs(rhip)).item()

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

