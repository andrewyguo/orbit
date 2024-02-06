# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import Sequence, Tuple
import omni
from omni.isaac.cloner import GridCloner

from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.assets import Articulation
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit_assets.cartpole import CARTPOLE_CFG
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.sim import SimulationCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.utils.math import sample_uniform
from pxr import PhysxSchema

import gymnasium as gym

import numpy as np 

@configclass
class CartpoleEnvAndrewCfg:
    viewer: ViewerCfg = ViewerCfg(eye = (8.0, 0.0, 5.0))
    sim: SimulationCfg = SimulationCfg(dt = 1 / 120)
    num_observations = 4

    scene = None

class CartpoleEnvAndrew(gym.Env):
    cfg: CartpoleEnvAndrewCfg

    def __init__(self, cfg: CartpoleEnvAndrewCfg):
        self.cfg = cfg
        self.num_envs = 4096
        self.num_actions = 1
        self.decimation = 2

        # simulation
        self.sim = SimulationContext(self.cfg.sim)
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        self.num_observations = cfg.num_observations

        self.stage = omni.usd.get_context().get_stage()
        # lights
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

        # clone
        self.cloner = GridCloner(spacing=4.0)
        self.cloner.define_base_env("/World/envs")
        self.env_prim_paths = self.cloner.generate_paths("/World/envs/env", self.num_envs)
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        env_origins = self.cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=False,
            copy_from_source=True,
        )
        self.env_origins = torch.tensor(env_origins, device=self.sim.device, dtype=torch.float32)
        
        # robot
        cartpole_cfg = CARTPOLE_CFG
        cartpole_cfg.prim_path = "/World/envs/env_.*/Robot"
        self.cartpole = Articulation(cartpole_cfg)

        self.cloner.replicate_physics(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            base_env_path="/World/envs",
            root_path="/World/envs/env_",
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # disable collisions between the different environments
        physics_scene_prim_path = None
        for prim in self.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physics_scene_prim_path = prim.GetPrimPath()
                break
        self.cloner.filter_collisions(
            physics_scene_prim_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=[], # NOTE: Once we have envs that walk on the ground, global path will have to be set see self._global_prim_paths in interactive_scene.py
        )


        # the reset here will call the initialization callback of the articulation
        self.sim.reset()

        self.slider_to_cart_joint_idx, _ = self.cartpole.find_joints("slider_to_cart")
        self.cart_to_pole_joint_idx, _ = self.cartpole.find_joints("cart_to_pole")

        # RL specifics
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.long)
        self.max_episode_length = math.ceil(5.0 / (self.cfg.sim.dt * self.decimation))
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        print(f"in cartpole_env.py, shape of actions: {self.actions.shape}")
        self.action_scale = 100.0

        # Observation space 
        self.single_observation_space = gym.spaces.Dict()
        # self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.get_observations()["cartpole"]["obs_buf"].shape)
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        # reward scales (could be fetched from config)
        self.rew_scale_alive = 1.0
        self.rew_scale_terminated = -2.0
        self.rew_scale_pole_pos = -1.0
        self.rew_scale_cart_vel = -0.01
        self.rew_scale_pole_vel = -0.005

    def step(self, actions: torch.Tensor):
        self.actions = self.action_scale * actions.clone()
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self.slider_to_cart_joint_idx)

        # perform physics stepping
        for _ in range(self.decimation):
            # set actions into simulator and step the actuator network
            self.cartpole.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.cartpole.update(dt=self.cfg.sim.dt)
        # perform rendering if gui is enabled
        if self.sim.has_gui():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)

        self.reset_terminated, self.reset_time_outs = self.get_dones()
        self.reward_buf = self.get_rewards()
        self.obs_buf = self.get_observations()

        reset_buf = self.reset_terminated | self.reset_time_outs
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._reset_idx(reset_env_ids)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, {}

    def get_observations(self) -> dict:        
        obs = torch.cat(
            (
                self.cartpole.data.joint_pos,
                self.cartpole.data.joint_vel,
            ),
            dim=-1,
        )
        observations = {"cartpole": {"obs_buf": obs}}
        return observations

    def get_rewards(self) -> torch.Tensor:
        rew_alive = self.rew_scale_alive
        rew_termination = self.rew_scale_terminated * self.reset_terminated.float()
        rew_pole_pos = self.rew_scale_pole_pos * torch.sum(torch.square(self.cartpole.data.joint_pos[:, self.cart_to_pole_joint_idx]), dim=1)
        rew_cart_vel = self.rew_scale_cart_vel * torch.sum(torch.abs(self.cartpole.data.joint_vel[:, self.slider_to_cart_joint_idx]), dim=1)
        rew_pole_vel = self.rew_scale_pole_vel * torch.sum(torch.abs(self.cartpole.data.joint_vel[:, self.cart_to_pole_joint_idx]), dim=1)
        total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
        return total_reward

    def get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_upper_limits = torch.any(self.cartpole.data.joint_pos[:, self.cart_to_pole_joint_idx] > 3.0, dim=1)
        out_of_lower_limits = torch.any(self.cartpole.data.joint_pos[:, self.cart_to_pole_joint_idx] < -3.0, dim=1)
        out_of_limits = torch.logical_or(out_of_upper_limits, out_of_lower_limits)
        return out_of_limits, time_out

    def reset(self):
        return self._reset_idx(None)

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        # reset the robots
        self.cartpole.reset(env_ids)
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self.cart_to_pole_joint_idx] += sample_uniform(-0.25 * math.pi, 0.25 * math.pi, joint_pos[:, self.cart_to_pole_joint_idx].shape, joint_pos.device)
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.env_origins[env_ids]
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset the buffers
        self.episode_length_buf[env_ids] = 0.0
