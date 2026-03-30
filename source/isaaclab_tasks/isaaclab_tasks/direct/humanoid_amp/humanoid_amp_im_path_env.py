# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
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
        print("---------------------------------------------------------------------")
        print("---------------------------------------------------------------------")
        print("PATH ENV INITIALIZED")
        print("---------------------------------------------------------------------")
        print("---------------------------------------------------------------------")

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

        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load(rf'C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\locomotion\FINAL_BEST_MODEL.pth',weights_only=True))
        self.gaitgen_net.to(self.sim.device)
        self.gaitgen_net.eval()

        self.mean = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\gait reference phase 2\mean.npy")
        self.std = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\humanoid_amp\gait reference phase 2\std.npy")
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

        self.desired_headings = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.robot_yaw = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.float32,
        )

        # TESTING SETTING
        self.test_fwd_speed: float | None = None
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

    def set_test_speed(self, fwd_speed: float | None):
        """If speed is not None, all envs will use this fixed speed at reset."""
        if fwd_speed is None:
            self.test_fwd_speed = None
        else:
            self.test_fwd_speed = float(fwd_speed)

    def set_test_heading(self, fwd_heading: float | None):
        """If speed is not None, all envs will use this fixed speed at reset."""
        if fwd_heading is None:
            self.test_heading = None
        else:
            self.test_heading = float(fwd_heading)

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
        self.robot_yaw = self._quat_to_yaw(torso_quat)
        self.heading_error = self._wrap_angle(self.desired_headings - self.robot_yaw)
        # # --------------------------    Gait Generator System    --------------------------#
        if self.reference is not None:
            self.phase_step = (self.episode_length_buf + self.random_start_idx) % self.periods
            self.phase = self.phase_step / self.periods

            if self.cfg.second_training:
                obs = torch.cat(
                    (
                        obs,
                        self.phase.unsqueeze(1),
                        self.desired_fwd_speeds.unsqueeze(1),
                        self.heading_error.unsqueeze(1)
                    ),
                    dim=-1
                )

            else:
                obs = torch.cat(
                    (
                        obs,
                        self.phase.unsqueeze(1),
                        self.desired_fwd_speeds.unsqueeze(1),
                        torch.zeros((self.num_envs,1), device=self.sim.device, dtype=torch.float32)
                    ),
                    dim=-1
                    )
        # # --------------------------    Gait Generator System    --------------------------#

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs[:,:81].clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        imitation_reward =  torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        decay_factor = 0.75
        imitation_coeff = torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        imitation_coeff = torch.where(self.desired_fwd_speeds > decay_factor, imitation_coeff * torch.exp( -1.0 *  (self.desired_fwd_speeds - decay_factor)), imitation_coeff)

        if self.cfg.second_training:

            imitation_weight_hip_pos  = 0.1 * imitation_coeff
            imitation_weight_knee_pos = 0.1 * imitation_coeff

            fwd_vel_weight = 0.3             * (1 /imitation_coeff)
            lat_vel_weight = 0.0
            heading_weight = 1.0
            death_cost = -1.0
        else:
            imitation_weight_hip_pos  = 0.375 * imitation_coeff
            imitation_weight_knee_pos = 0.375 * imitation_coeff

            fwd_vel_weight = 0.75             * (1 /imitation_coeff)
            lat_vel_weight = 0.0
            heading_weight = 0.0
            death_cost = -1.5

        env_ids = torch.arange(self.reference.shape[0], device=self.sim.device)

        current_ref_pos   = self.reference[env_ids,self.phase_step,:]           # [N,6]

        # hips
        hip_joint_pos = self.robot.data.joint_pos[:, [12, 15]]          # [N, 2]
        hip_ref_pos = current_ref_pos[:,[0,2]]
        hip_diff      = hip_joint_pos - hip_ref_pos          # [N, 2]
        hip_dist      = torch.norm(hip_diff, p=2, dim=-1)    # [N]
        imitation_reward = imitation_reward + imitation_weight_hip_pos * torch.exp(-3.0 * hip_dist)

        # knees
        knee_joint_pos = self.robot.data.joint_pos[:, [20, 21]]         # [N, 2]
        knee_ref_pos   = current_ref_pos[:,[1,3]]
        knee_diff      = knee_joint_pos - knee_ref_pos
        knee_dist      = torch.norm(knee_diff, p=2, dim=-1)
        imitation_reward = imitation_reward + imitation_weight_knee_pos * torch.exp(-3.0 * knee_dist)

        # Body velocity
        vel_b = self.robot.data.root_com_lin_vel_b  # [N, 3]
        vel_ang = self.robot.data.root_com_ang_vel_b  # [N, 3]

        # Decompose
        forward_speed  = vel_b[:, 0]   # along robot-forward (x)
        lateral_speed  = vel_b[:, 1]   # sideways (y)
        yaw_speed = vel_ang[:, 2]   # yaw (w)

        vel_reward_fwd = torch.exp(-4.0 * torch.abs(forward_speed - (self.desired_fwd_speeds * 2.4)))
        vel_reward = fwd_vel_weight * vel_reward_fwd

        # Heading reward via yaw-rate P-controller (FIX 3)
        desired_yaw_rate = torch.clamp(0.5 * self.heading_error, -1.0, 1.0)
        yaw_tracking_err = torch.abs(desired_yaw_rate - yaw_speed)
        heading_reward   = heading_weight * torch.exp(-yaw_tracking_err / 0.25)

        # vel_reward_ang = torch.exp(-5.0 * torch.abs(yaw_speed - (self.desired_ang_speeds)))
        # vel_reward += yaw_vel_weight * vel_reward_ang

        # lat_vel_reward = lat_vel_weight * torch.exp(-4.0 * torch.abs(lateral_speed))
        # vel_reward += lat_vel_reward

        total_reward +=  imitation_reward + vel_reward + heading_reward

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

        i = 0
        for env_id in env_ids:

            if self.demo_mode and (self.test_fwd_speed is not None):
                self.episode_length_buf[env_ids] = 0
                self.desired_fwd_speeds[env_id] = self.test_fwd_speed / 2.4

                if self.test_heading is not None:
                    self.desired_headings[env_id] = self.test_heading
                    print(f"Heading is set to: {self.test_heading}")
                else:
                    self.desired_headings[env_id] = 0 

                self.reference_fwd_speed = self.test_fwd_speed / 2.4
                random_idx = None
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_fwd_speed
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:], self.periods[env_id] = self.findgait(encoder_vec) 
                
                self.phase_step[env_id] = 0             #fix this part it should not be 0 it should be closest index to 0 degree
                self.reference[env_id,[0,2]]  = torch.clamp(self.reference[env_id,[0,2]] , -np.pi/2, np.pi/2)     #Clip the hip angle
                self.reference[env_id,[1,3]]  = torch.clamp(self.reference[env_id,[1,3]] , -np.pi, 0)     
                root_state[i, 2] += 0.4

            elif self.demo_mode and (self.test_fwd_speed is None):
                raise ValueError("NO TEST SPEED IS DEFINED!!!!!")

            else:
                self.reference_fwd_speed = np.random.uniform(0.1, 2.4) / 2.4
                self.reference_heading = np.random.uniform(-1.0, 1.0)

                random_idx = np.random.randint(0,int(3/self.dt))

                self.desired_fwd_speeds[env_id] = self.reference_fwd_speed 
                self.desired_headings[env_id] = self.reference_heading
                
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_fwd_speed
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:], self.periods[env_id] = self.findgait(encoder_vec) 
                self.phase_step[env_id] = int(random_idx % self.periods[env_id])
                self.reference[env_id,[0,2]]  = torch.clamp(self.reference[env_id,[0,2]] , -np.pi/2, np.pi/2)     #Clip the hip angle
                self.reference[env_id,[1,3]]  = torch.clamp(self.reference[env_id,[1,3]] , -np.pi, 0)             #Clip the knee angle
            

            if random_idx:
                self.random_start_idx[env_id] = self.phase_step[env_id]

                rhip_pos = self.reference[env_id,self.phase_step[env_id],0]
                lhip_pos = self.reference[env_id,self.phase_step[env_id],2]

                rknee_pos = self.reference[env_id,self.phase_step[env_id],1]
                lknee_pos = self.reference[env_id,self.phase_step[env_id],3]

                hip_rad = lhip_pos if torch.abs(rhip_pos) > torch.abs(lhip_pos) else rhip_pos
                knee_rad = lknee_pos if torch.abs(rknee_pos) > torch.abs(lknee_pos) else rknee_pos

                hip_short = self.hip_knee - (torch.cos(hip_rad) * self.hip_knee)
                knee_short = self.knee_ankle - (torch.cos(knee_rad) * self.knee_ankle)
                total_short = hip_short + knee_short - 0.03 if (hip_short+knee_short) > 0.05 else 0
                # -------------------- RSI ON -----------------------------#
                joint_pos[i,12] = rhip_pos
                joint_pos[i,20] = rknee_pos

                joint_pos[i,15] = lhip_pos
                joint_pos[i,21] = lknee_pos

                root_state[i, 2] += 0.4
            i+=1
                
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

    def findgait(self, input_vec: torch.Tensor) -> torch.Tensor:
        if input_vec.ndim == 1:
            input_vec = input_vec.unsqueeze(0)        # [1, D]

        N = input_vec.shape[0]
        freqs = self.gaitgen_net(input_vec)
        predictions_np = freqs.detach().cpu().numpy()

        pred_list = []
        for n in range(N):
            single_out = predictions_np[n]                    
            single_pred = single_out[:136]
            single_period = single_out[136]
            single_pred = self.denormalize(single_pred,self.mean,self.std)      
            single_pred = self.recover_shape(single_pred)
            single_time = self.pred_ifft(single_pred)           
            pred_list.append(single_time)
        pred_np = np.stack(pred_list, axis=0)
        pred_torch = torch.from_numpy(pred_np).to(self.sim.device, dtype=torch.float32)
        single_period = int(single_period * self.episode_len)
        single_period = 12 if single_period < 12 else single_period
        return pred_torch, single_period # type: ignore

    def denormalize(self,pred_flat, mean, std):
        return (pred_flat * std) + mean

    def recover_shape(self,flat_data):
        recovered = flat_data.reshape(17, 4, 2)
        structured = recovered.transpose(1, 2, 0)
        return structured

    def pred_ifft(self,predictions):
        complex_pred = predictions[:, 0, :] + 1j * predictions[:, 1, :]
        org_rate = 10
        pred_time = np.fft.irfft(complex_pred, n=32, axis=1)
        pred_time = pred_time.transpose(1,0)
        if self.dt < 0.1:
            num_samples = int((pred_time.shape[0]) * (1/self.dt)/(org_rate)) 
            pred_time = resample(pred_time, num_samples, axis=0)
        return pred_time
    
    def find_closest_index(self,reference,period):
        if self.demo_mode and (self.test_fwd_speed is not None):
            distances = torch.abs(reference[:period,:]).sum(dim=-1)
            closest_index = torch.argmin(distances)
            active_phase_step = closest_index.item()

            return active_phase_step
        else:
            raise ValueError("No test speed provided")

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