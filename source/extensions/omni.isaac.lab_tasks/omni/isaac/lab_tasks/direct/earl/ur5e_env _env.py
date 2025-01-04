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

import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, UR5e_CFG, AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg, ISAAC_NUCLEUS_DIR
import .ur5e_control as ur5e_control

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
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )
    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kowndi/Documents/Isaac_Sim_Learning/Dynamic_HRI/RL_Lab/table_lab/table_lab.usd",
            scale=(1, 1, 1)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, 0.0, -0.035), rot=(0.0, 0.0, 0.0, 0.0)),
    )
    # robot
    robot_cfg: ArticulationCfg = UR5e_CFG 

    # orange_juice
    orange_juice_cfg = ArticulationCfg(
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
    milk_cfg = ArticulationCfg(
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
    ketchup_cfg = ArticulationCfg(
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
    cookies_cfg = ArticulationCfg(
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
    bbq_sauce_cfg = ArticulationCfg(
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
    potted_meat_can_cfg = ArticulationCfg(
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
    cracker_box_cfg = ArticulationCfg(
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
    spaghetti_cfg = ArticulationCfg(
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
  
     
    
   
class UR5eVSGEnv(DirectRLEnv):
    cfg: UR5eVSGEnvCfg

    def __init__(self, cfg: UR5eVSGEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.headless = False
        self.max_episode_length = self.cfg.episode_length_s  / (self.cfg.decimation * 2 * self.cfg.sim.dt) 
        
        # [DOUBT@Kowndinya]: self.aggregate_mode is set to true in previous EARL
        # Reason: self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True) in _create_envs() in ur5_vsg_env.py

        self.gripper_lift_pos = torch.tensor([0.65, 0.15, 0.33], device=self.cfg.sim.device).repeat((self.cfg.scene.num_envs, 1))
        self.gripper_lift_quat = torch.tensor([-0.048, 0.029, -0.002, -0.998], device=self.cfg.sim.device).repeat((self.cfg.scene.num_envs, 1))
        
        self.cfg_ctrl = {}
        self.cfg_ctrl["num_envs"] = cfg.scene.num_envs
        self.cfg_ctrl["jacobian_type"] = "geometric"
        self.cfg_ctrl["motor_ctrl_mode"] = "velocity"
        self.cfg_ctrl["gain_space"] = "joint"
        self.cfg_ctrl["ik_method"] = "dls"
        self.cfg_ctrl["do_force_ctrl"] = False
        self.cfg_ctrl["do_inertial_comp"] = False

        # [PENDING@Kowndinya]: Add color to gripper fingers as in previous EARL
        self.acquire_env_tensors()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._object = Articulation(self.cfg.orange_juice_cfg)
        self._table = RigidObject(self.cfg.table_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["object"] = self._object
        self.scene.rigid_objects["table"] = self._table
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
        light_cfg.func("/World/light", light_cfg)
        
    def acquire_env_tensors(self):
        self._jacobian = self._robot.root_physx_view.get_jacobians()
        self._robot_link_positions = self._robot.data.body_state_w()


        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()


    def _apply_action(self) -> None:
        pass 

    def _get_observations(self) -> dict:
        pass

    def is_static(self, prev_body_pos, prev_body_quat, body_pos, body_quat, pos_threshold=0.001, quat_threshold=0.01):
        pos_static = torch.all((prev_body_pos - body_pos).abs() < pos_threshold)
        quat_static = torch.all((prev_body_quat - body_quat).abs() < quat_threshold)
        return pos_static & quat_static
    

    def _get_keypoint_offsets(self, offset_distance):
        """Get four cornor points offsets, centered at 0."""

        keypoint_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
        keypoint_offsets[:, 0, 0:2] = offset_distance
        keypoint_offsets[:, 1, 0] = -offset_distance
        keypoint_offsets[:, 1, 1] = offset_distance
        keypoint_offsets[:, 2, 0:2] = -offset_distance
        keypoint_offsets[:, 3, 0] = offset_distance
        keypoint_offsets[:, 3, 1] = -offset_distance
        return keypoint_offsets

    def _set_dof_velocity(self):
        """Set robot DOF velocity to move fingertips towards target pose."""

        self.ctrl_robot_dof_velocity[:] = ur5e_control.compute_dof_velocity(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            arm_dof_vel=self.arm_dof_vel,
            gripper_dof_pos=self.gripper_dof_pos,
            fingertip_midpoint_pos=self.gripper_grasp_pose_pos,
            fingertip_midpoint_quat=self.gripper_grasp_pose_quat,
            jacobian=self.gripper_grasp_pose_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_gripper_dof_pos_target,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_gripper_grasp_pose_pos_target,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_gripper_grasp_pose_quat_target,
            device=self.device,
            dof_pos_default=self.robot_dof_pos_default,
        )

        # TODO: only for ur5, if it is a different robot, this should be re-designed
        arm_dof_idx = self.arm_dof_pos.shape[1]
        max_vel = 0.2
        ratio = max_vel / self.ctrl_robot_dof_velocity[:, 0 : arm_dof_idx].abs().max(dim=1).values
        ratio[ratio > 1] = 1
        self.ctrl_robot_dof_velocity[:, 0 : arm_dof_idx] = ratio.view(-1, 1) * self.ctrl_robot_dof_velocity[:, 0 : arm_dof_idx]
        # print(self.ctrl_robot_dof_velocity[:, 0 : arm_dof_idx].max(dim=1).values.max())

    def _set_dof_torque(self):
        """Set robot DOF torque to move fingertips towards target pose."""

        self.ctrl_robot_dof_torque[:] = ur5e_control.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            arm_dof_vel=self.arm_dof_vel,
            gripper_dof_pos=self.gripper_dof_pos,
            gripper_dof_vel=self.gripper_dof_vel,
            fingertip_midpoint_pos=self.gripper_grasp_pose_pos,
            fingertip_midpoint_quat=self.gripper_grasp_pose_quat,
            fingertip_midpoint_linvel=self.gripper_grasp_pose_linvel,
            fingertip_midpoint_angvel=self.gripper_grasp_pose_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.gripper_grasp_pose_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_gripper_dof_pos_target,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_gripper_grasp_pose_pos_target,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_gripper_grasp_pose_quat_target,
            ctrl_target_fingertip_contact_wrench=self.ctrl_fingertip_contact_wrench_target,
            device=self.device,
        )
