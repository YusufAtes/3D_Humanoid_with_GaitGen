# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader


from .gait_generator_net import SimpleFCNN
import numpy as np
from scipy.signal import resample

class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        # The scale must be half the range to map [-1, 1] to the limits
        self.action_scale = 0.5 * (dof_upper_limits - dof_lower_limits)
        # self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
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

        self.desired_speeds = torch.zeros(
            (self.num_envs,),
            device=self.sim.device,
            dtype=torch.float32,
        )
        # TESTING SETTING
        self.test_speed: float | None = None
        self.demo_mode: bool = False
        self.random_start_idx = torch.zeros((self.num_envs,),device=self.sim.device,dtype=self.episode_length_buf.dtype) 
        self.avg_speed = 0

        # # --------------------------    Gait Generator System    --------------------------#

    def set_test_speed(self, speed: float | None):
        """If speed is not None, all envs will use this fixed speed at reset."""
        if speed is None:
            self.test_speed = None
            self.demo_mode = False
        else:
            self.test_speed = float(speed)
            self.demo_mode = True


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
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
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # # --------------------------    Gait Generator System    --------------------------#
        if self.reference is not None:

            # physics steps since reset ≈ episode_length * decimation
            self.phase_step = int(self.episode_length_buf + self.random_start_idx) % self.periods
            self.phase = self.phase_step / self.periods

            obs = torch.cat(
                (
                    obs,
                    self.phase.unsqueeze(1),
                    self.desired_speeds.unsqueeze(1)
                ),
                dim=-1
                )
        # # --------------------------    Gait Generator System    --------------------------#

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs[:,:81].clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        imitation_reward =  torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        imitation_weight_hip_pos = 0.5
        imitation_weight_knee_pos = 0.5
        fwd_vel_weight = 0.75
        lat_vel_weight = 0.2
        yaw_vel_weight = 0.2
        death_cost = -1.0

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

        # Reward forward speed and penalize lateral drift and yaw drift
        vel_reward_fwd = torch.exp(-3.0 * (forward_speed - (self.desired_speeds * 2.4))**2)
        vel_reward = fwd_vel_weight * vel_reward_fwd
        vel_reward += lat_vel_weight * torch.exp(-2.0 * torch.abs(lateral_speed))
        vel_reward += yaw_vel_weight * torch.exp(-2.0 * torch.abs(yaw_speed))

        total_reward = imitation_reward + vel_reward 

        died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        total_reward = torch.where(died, torch.ones_like(total_reward) * death_cost, total_reward)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.demo_mode:
            x_pos = self.robot.data.body_pos_w[0, self.ref_body_index, 0].item()
            step = int(self.episode_length_buf.item())
            t = step * float(self.dt)

            if step == 299:
                self.avg_speed = x_pos / t
                print("Average speed:", self.avg_speed)

        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            if self.demo_mode and died:
                x_pos = self.robot.data.body_pos_w[0, self.ref_body_index, 0].item()
                step = int(self.episode_length_buf.item())
                t = step * float(self.dt)
                self.avg_speed = x_pos / t
                print("Average speed:", self.avg_speed)
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)

        # if self.cfg.reset_strategy == "default":
        #     root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        # elif self.cfg.reset_strategy.startswith("random"):
        #     start = "start" in self.cfg.reset_strategy
        #     root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        # else:
        #     raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        # # --------------------------    Gait Generator System    --------------------------#

        i = 0
        for env_id in env_ids:
            if self.demo_mode and (self.test_speed is not None):
                self.desired_speeds[env_id] = self.test_speed / 2.4
                self.reference_speed = self.test_speed / 2.4
                random_idx = None
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_speed
                encoder_vec[1] = self.leg_len /1.0
                encoder_vec[2] = self.leg_len /1.0

                self.reference[env_id,:], self.periods[env_id] = self.findgait(encoder_vec) 
                
                self.phase_step[env_id] = 0             #fix this part it should not be 0 it should be closest index to 0 degree
                self.reference[env_id,[0,2]]  = torch.clamp(self.reference[env_id,[0,2]] , -np.pi/2, np.pi/2)     #Clip the hip angle
                self.reference[env_id,[1,3]]  = torch.clamp(self.reference[env_id,[1,3]] , -np.pi, 0)     
                root_state[i, 2] += 0.4

            else:
                self.reference_speed = np.random.uniform(0.1, 2.4) / 2.4
                random_idx = np.random.randint(0,int(3/self.dt))
                self.desired_speeds[env_id] = self.reference_speed 
                
                encoder_vec = torch.empty((3),device=self.sim.device)   
                encoder_vec[0] = self.reference_speed
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
                
                # -------------------- RSI ON -----------------------------#

        # # --------------------------    Gait Generator System    --------------------------#
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
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
            single_period = single_out[136]
            single_pred = self.denormalize(single_pred,self.mean,self.std)         # numpy [6, 2, 17]
            single_pred = self.recover_shape(single_pred)
            single_time = self.pred_ifft(single_pred)           # numpy [T, C]
            pred_list.append(single_time)
        # Stack: numpy [N, T, C]
        pred_np = np.stack(pred_list, axis=0)
        # Convert back to torch on device
        pred_torch = torch.from_numpy(pred_np).to(self.sim.device, dtype=torch.float32)
        single_period = int(single_period * self.episode_len)
        return pred_torch, single_period # type: ignore

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
        return pred_time
    

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
