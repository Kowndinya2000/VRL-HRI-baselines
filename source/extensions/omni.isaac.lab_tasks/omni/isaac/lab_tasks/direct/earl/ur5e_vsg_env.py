## Previous Author: Baichuan Huang
## Current Author: Kowndinya Boyalakuntla
## Date: 01 Jan, 2025
## Purpose: Migrate EARL from IsaacGym to Isaac Lab

import os 
import math
import glob
import random

import numpy as np
from PIL import Image
import cv2


import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, UR5e_CFG, AssetBaseCfg


@configclass
class UR5eVSGEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 37 # max_episode_length_s in seconds = dt * decimation * max_episode_length (in steps - from IsaacGym)
    decimation = 3
    # action_scale = 0.5
    action_space = 7
    observation_space = 49
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        device = "cuda:0",
        dt=1 / 60, 
        render_interval=2,
        # up_axis is always Z in Isaac Sim, so, no need to specify it
        # use_gpu_pipeline is deduced from the device
        gravity = (0.0, 0.0, -9.81),
        physx: PhysxCfg = PhysxCfg(
            # number of threads is no longer needed
            solver_type = 1,
            # use_gpu is deduced from the device
            max_position_iteration_count = 12,
            max_velocity_iteration_count = 4,
            # moved to actor config: contact_offset: 0.005  # default = 0.02
            # moved to actor config: rest_offset: 0.0  # default = 0.001
            bounce_threshold_velocity = 0.2,
            # moved to actor config: max_depentration_velocity: 10.0  # default = 100
            # default_buffer_size_multiplier is no longer needed
            gpu_max_rigid_contact_count=2**23,
            # num_subscenes is no longer needed
            # contact_collection is no longer needed
            
        ))
    ## [Comment by Kowndinya, 02 Jan, 2025]: add_damping: False - do not know how to migrate this from IsaacGym to Isaac Lab

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=4.0, replicate_physics=True)

    

class UR5eVSGEnv(DirectRLEnv):
    cfg: 