# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .gait_generator_net import SimpleFCNN
import numpy as np
from scipy.signal import resample
from pxr import UsdGeom, Gf

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # # --------------------------    Gait Generator System    --------------------------#

        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load(rf'C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\locomotion\FINAL_BEST_MODEL.pth',weights_only=True))
        self.gaitgen_net.to(self.sim.device)
        self.gaitgen_net.eval()

        self.mean = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\locomotion\mean.npy")
        self.std = np.load(r"C:\Users\bates\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\locomotion\std.npy")
        self.leg_len = 0.83
        self.hip_knee = 0.4
        self.knee_ankle = 0.39
        self.dt = self.step_dt 
        self.reference = torch.empty(
            (self.num_envs, 1920, 4),
            device=self.sim.device,
            dtype=torch.float32,
        )

        # TESTING SETTING
        self.test_speed: float | None = None
        self.use_test_speed: bool = False
        self.random_start_idx = torch.zeros((self.cfg.terrain.num_envs,),device=self.sim.device,dtype=self.episode_length_buf.dtype) 
        self.desired_speeds = torch.zeros((self.cfg.terrain.num_envs,),device=self.sim.device,dtype=self.episode_length_buf.dtype) 
        
        # Demo mode settings
        self.demo_mode = getattr(self.cfg, 'demo_mode', False)
        if self.demo_mode:
            self.demo_type = getattr(self.cfg, 'demo_type', None)
        else:
            self.demo_type = None
        self.test_slope_deg = getattr(self.cfg, 'test_slope_deg', 0.0)
        self.ramp_demo = False
        if self.test_slope_deg != 0:
            self.ramp_demo = True
        self.avg_speed = 0

        # # --------------------------    Gait Generator System    --------------------------#

    def set_test_speed(self, speed: float | None):
        """If speed is not None, all envs will use this fixed speed at reset."""
        if speed is None:
            self.test_speed = None
            self.use_test_speed = False
        else:
            self.test_speed = float(speed)
            self.use_test_speed = True

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        if self.cfg.demo_mode and self.cfg.demo_type == "noise" and self.cfg.noise_amplitude > 0.0:
            from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg, HfWaveTerrainCfg
            from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
            # Create noisy terrain generator config
            amp = self.cfg.noise_amplitude
            noise_seed = self.cfg.noise_seed
            noise_type = self.cfg.noise_type
            
            if noise_type == "wave":
                sub_terrains = {
                    "wave_terrain": HfWaveTerrainCfg(
                        proportion=1.0,
                        amplitude_range=(amp, amp),
                        num_waves=4,
                        border_width=0.25,
                    ),
                }
            else:  # "random" (default)
                sub_terrains = {
                    "random_rough": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        noise_range=(-amp, amp),
                        noise_step=0.005,
                        border_width=0.25,
                    ),
                }
            
            # Create terrain generator config
            noisy_terrain_cfg = TerrainGeneratorCfg(
                seed=noise_seed,
                size=(20.0, 20.0),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains=sub_terrains,
            )
            
            # Update terrain config to use generator
            self.cfg.terrain.terrain_type = "generator"
            self.cfg.terrain.terrain_generator = noisy_terrain_cfg
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            # For ramp demo or regular mode, use ground plane or configured terrain
            if self.cfg.demo_mode and self.cfg.demo_type == "ramp":
                # For ramp demo, always use ground plane
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
                # Apply ramp rotation
                slope = float(self.test_slope_deg)
                if slope != 0.0:
                    stage = self.sim.stage
                    prim = stage.GetPrimAtPath("/World/ground")
                    if prim.IsValid():
                        xform = UsdGeom.Xformable(prim)
                        xform.ClearXformOpOrder()
                        rot_op = xform.AddRotateXYZOp()
                        # Rotate around Y axis for slope
                        rot_op.Set(Gf.Vec3f(0.0, slope, 0.0))
            else:
                # Use configured terrain (plane or generator)
                self.cfg.terrain.num_envs = self.scene.cfg.num_envs
                self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
                self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            terrain_prim_path = getattr(self.cfg.terrain, 'prim_path', "/World/ground")
            self.scene.filter_collisions(global_prim_paths=[terrain_prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions            ),
            dim=-1,
        )

        # # --------------------------    Gait Generator System    --------------------------#
        if self.reference is not None:
            T = self.reference.shape[1]
            # physics steps since reset ≈ episode_length * decimation
            self.step_idx = self.episode_length_buf + self.random_start_idx
            # clamp to leave room for t+10
            max_idx = T - 11
            env_ids = torch.arange(self.reference.shape[0], device=self.sim.device)
            self.current_ref = self.reference[env_ids,self.step_idx,:]
            future_reft1 = self.reference[env_ids,self.step_idx + 1,:]
            future_reft10 = self.reference[env_ids,self.step_idx + 10,:]
            obs = torch.cat(
                (
                    obs,
                    self.current_ref,
                    future_reft1,
                    future_reft10
                ),
                dim=-1
                )

        # # --------------------------    Gait Generator System    --------------------------#

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        total_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
            self.current_ref,
            self.dof_pos,
            self.vel_loc,
            self.desired_speeds,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # --------------------------------------------------------
        # DEMO MODE METRICS
        # --------------------------------------------------------
        if self.demo_mode:
            x_pos = self.torso_position[0, 0].item()
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
        if getattr(self.cfg, 'early_termination', True):
            # 1. Get Robot Root positions
            root_y = self.torso_position[:, 0] 
            root_z = self.torso_position[:, 2]

            # 2. Calculate Slope Angle in Radians
            # We create a tensor on the same device to ensure GPU compatibility
            slope_deg = torch.tensor(self.cfg.test_slope_deg, device=self.device)
            slope_rad = torch.deg2rad(slope_deg)

            # 3. Calculate Floor Z at the current Y position
            # Formula: z_floor = y * tan(theta)
            floor_z = root_y * torch.tan(-slope_rad)
            # 4. Calculate Relative Height
            # This is the height of the robot ABOVE the calculated inclined floor
            relative_height = root_z - floor_z
            # 5. Check Termination
            died = relative_height < self.cfg.termination_height
            
            # Handle Demo Mode Stats if died
            if self.cfg.demo_mode and died.any():
                x_pos = self.torso_position[0, 0].item()
                step = int(self.episode_length_buf.item())
                t = step * float(self.dt)
                if t > 0:
                    self.avg_speed = x_pos / t
        else:
            died = self.torso_position[:, 2] < self.cfg.termination_height

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # # --------------------------    Gait Generator System    --------------------------#
        i = 0
        for env_id in env_ids:
            if self.demo_mode and (self.test_speed is not None):
                self.episode_length_buf[env_ids] = 0
                self.desired_speeds[env_id] = self.test_speed / 2.4
                self.reference_speed = self.test_speed / 2.4
                random_idx = None
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_speed
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:] = self.findgait(encoder_vec) 
                self.reference[env_id,:]  = torch.clamp(self.reference[env_id,:] , -np.pi/2, np.pi/2)     #Clip the hip angle
                self.reference[env_id,[1,3]]  = torch.clamp(self.reference[env_id,[1,3]] , -np.pi, 0)     #Clip the knee angle
                
            elif self.demo_mode and (self.test_speed is None):
                raise ValueError("NO TEST SPEED IS DEFINED!!!!!")
                
            elif self.use_test_speed and (self.test_speed is not None):
                self.reference_speed = self.test_speed
                random_idx = None
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_speed/2.4
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:] = self.findgait(encoder_vec)                     #Find the gait
                self.reference[env_id,:]  = torch.clamp(self.reference[env_id,:] , -np.pi/2, np.pi/2)     #Clip the gait
                self.desired_speeds[env_id] = self.reference_speed
            else:
                self.reference_speed = np.random.uniform(0.2, 2.4)
                random_idx = np.random.randint(0,int(1/self.dt))
                encoder_vec = torch.empty((3),device=self.sim.device)   

                encoder_vec[0] = self.reference_speed/2.4
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:] = self.findgait(encoder_vec)                     #Find the gait
                self.reference[env_id,:]  = torch.clamp(self.reference[env_id,:] , -np.pi/2, np.pi/2)     #Clip the gait
                self.desired_speeds[env_id] = self.reference_speed
            
            if random_idx is not None:
                self.random_start_idx[env_id] = random_idx

                rhip_pos = self.reference[env_id,random_idx,0]
                lhip_pos = self.reference[env_id,random_idx,2]

                rknee_pos = self.reference[env_id,random_idx,1]
                lknee_pos = self.reference[env_id,random_idx,3]

                hip_rad = lhip_pos if torch.abs(rhip_pos) > torch.abs(lhip_pos) else rhip_pos
                knee_rad = lknee_pos if torch.abs(rknee_pos) > torch.abs(lknee_pos) else rknee_pos

                hip_short = self.hip_knee - (torch.cos(hip_rad) * self.hip_knee)
                knee_short = self.knee_ankle - (torch.cos(knee_rad) * self.knee_ankle)
                total_short = hip_short + knee_short - 0.03 if (hip_short+knee_short) > 0.05 else 0
                # -------------------- RSI ON -----------------------------#
                joint_pos[i,10] = rhip_pos
                joint_pos[i,15] = rknee_pos

                joint_pos[i,13] = lhip_pos
                joint_pos[i,16] = lknee_pos

                default_root_state[i, 2] -= (total_short)

                i+=1
                
                # -------------------- RSI ON -----------------------------#
            else:
                i+=1

        # # --------------------------    Gait Generator System    --------------------------#

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


    def findgait(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        input_vec: [N, D] or [D]
        returns:   [N, T, C] torch.Tensor
        """

        # Ensure batch dimension
        if input_vec.ndim == 1:
            input_vec = input_vec.unsqueeze(0)         # [1, D]

        N = input_vec.shape[0]

        # Forward network: [N, 204]
        freqs = self.gaitgen_net(input_vec)

        # → Convert to numpy for your existing post-processing code
        predictions_np = freqs.detach().cpu().numpy()

        # Process each environment individually
        pred_list = []
        for n in range(N):
            single_out = predictions_np[n]                     # Currently Flatten
            single_pred = single_out[:136]
            single_pred = self.denormalize(single_pred,self.mean,self.std)         # numpy [6, 2, 17]
            single_pred = self.recover_shape(single_pred)
            single_time = self.pred_ifft(single_pred)           # numpy [T, C]
            pred_list.append(single_time)

        # Stack: numpy [N, T, C]
        pred_np = np.stack(pred_list, axis=0)

        # Convert back to torch on device
        pred_torch = torch.from_numpy(pred_np).to(self.sim.device, dtype=torch.float32)

        return pred_torch

    def denormalize(self,pred_flat, mean, std):
        """Reverses the Standard Score normalization."""
        return (pred_flat * std) + mean

    def recover_shape(self,flat_data):
        """
        Reconstructs (4, 2, 17) from flat (136,) vector.
        """
        # 1. Reshape to creation shape: (Freqs=17, Joints=4, Real/Imag=2)
        recovered = flat_data.reshape(17, 4, 2)
        # 2. Transpose to IFFT shape: (Joints=4, Real/Imag=2, Freqs=17)
        structured = recovered.transpose(1, 2, 0)
        return structured

    def pred_ifft(self,predictions):
        """
        Performs Inverse FFT to get time-domain signals.
        """
        # Combine Real and Imaginary parts
        complex_pred = predictions[:, 0, :] + 1j * predictions[:, 1, :]
        org_rate = 10
        # Inverse FFT (n=32 points)
        pred_time = np.fft.irfft(complex_pred, n=32, axis=1)
        pred_time = pred_time.transpose(1,0)
        # Transpose for plotting: (Time, Joints)
        if self.dt < 0.1:
            num_samples = int((pred_time.shape[0]) * (1/self.dt)/(org_rate))  # resample with self.dt
            # Upsample using Fourier method
            pred_time = resample(pred_time, num_samples, axis=0)
            pred_time = np.tile(pred_time, (10,1))    # Create loop for reference movement
        return pred_time
    


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
    reference_state: torch.Tensor,
    dof_pos : torch.Tensor,
    vel_loc : torch.Tensor,
    desired_speeds : torch.Tensor,
):
    # ----------------- base humanoid rewards ----------------- #
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(
        heading_proj > 0.8,
        heading_weight_tensor,
        heading_weight * heading_proj / 0.8,
    )

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = (potentials - prev_potentials) * 0.5

    # ----------------- Gait Generator imitation ----------------- #

    imitation_reward = torch.zeros_like(heading_reward)
    imitation_weight_hip_pos = 0.75
    imitation_weight_knee_pos = 0.75
    imitation_weight_ankle_pos = 0.25
    vel_weight = 1.5

    # hips
    hip_joint_pos = dof_pos[:, [10, 13]]          # [N, 2]
    hip_ref_pos   = reference_state[:, [0, 2]]           # [N, 2]
    hip_diff      = hip_joint_pos - hip_ref_pos          # [N, 2]
    hip_dist      = torch.norm(hip_diff, p=2, dim=-1)    # [N]
    imitation_reward = imitation_reward + \
        imitation_weight_hip_pos * torch.exp(-5.0 * hip_dist)

    # knees
    knee_joint_pos = dof_pos[:, [15, 16]]         # [N, 2]
    knee_ref_pos   = reference_state[:, [1, 3]]          # [N, 2]
    knee_diff      = knee_joint_pos - knee_ref_pos
    knee_dist      = torch.norm(knee_diff, p=2, dim=-1)
    imitation_reward = imitation_reward + \
        imitation_weight_knee_pos * torch.exp(-5.0 * knee_dist)

    vel_error = torch.abs(vel_loc[:, 0] - desired_speeds)
    vel_reward = torch.exp(-2.0 * vel_error) * vel_weight
    # ----------------- Gait Generator imitation ----------------- #

    # ----------------- total reward ----------------- #
    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        + imitation_reward
        + vel_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        reset_terminated,
        torch.ones_like(total_reward) * death_cost,
        total_reward,
    )
    return total_reward



@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )

