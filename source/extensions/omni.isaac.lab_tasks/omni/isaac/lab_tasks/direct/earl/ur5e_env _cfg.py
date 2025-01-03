## Previous Author: Baichuan Huang
## Current Author: Kowndinya Boyalakuntla
## Date: 02 Jan, 2025
## Purpose: Migrate EARL from IsaacGym to Isaac Lab

import os 
import math
import glob
import random

import numpy as np
from PIL import Image
import cv2


import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, UR5e_CFG, ISAAC_NUCLEUS_DIR


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
    
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/table_lab/table_lab.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, 0.0, -0.035), rot=(0.0, 0.0, 0.0, 0.0)),
    )
    # robot
    robot: ArticulationCfg = UR5e_CFG 

    # orange_juice
    orange_juice = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/OrangeJuice/OrangeJuice.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )

    # milk
    milk = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/Milk/Milk.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )
    
    # ketchup
    ketchup = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/Ketchup/Ketchup.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )
    # lights
    
    # cookies
    cookies = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/Cookies/Cookies.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )
    
    # BBQSauce
    bbq_sauce = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/BBQSauce/BBQSauce.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )

    # BBQSauce
    potted_meat_can = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/010_potted_meat_can",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )    
    
    # CrackerBox
    cracker_box = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/003_cracker_box/003_cracker_box.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )    

    # Spaghetti
    spaghetti = ArticulationCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/Spaghetti/Spaghetti.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                    activate_contact_sensors=False,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos = (0.65, 0.0, 0.23)
                ),
            )    
  
     
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
   
class UR5eVSGEnv(DirectRLEnv):
    cfg: UR5eVSGEnvCfg

    def __init__(self, cfg: UR5eVSGEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.headless = False
        self.max_episode_length = cfg.episode_length_s  / (cfg.decimation * 2 * cfg.sim.dt) 
        
        #[DOUBT@Kowndinya]: self.aggregate_mode is set to true in previous EARL
        # Reason: self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True) in _create_envs() in ur5_vsg_env.py