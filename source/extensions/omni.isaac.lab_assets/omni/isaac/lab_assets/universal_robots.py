"""
Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR5e_CFG`: The UR5e arm calibrated with the gripper.
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_ASSETS_DATA_DIR
import os

##
# Configuration for the UR5e arm + gripper.
##

UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
     usd_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "ur5e_simplified_gripper.usd"), 
     rigid_props=sim_utils.RigidBodyPropertiesCfg(
         disable_gravity=True,
         contact_offset=0.005,  # default = 0.02
         rest_offset=0.0,  # default = 0.001
         max_depentration_velocity=10.0,  # default = 100
         linear_damping=0.0,
         angular_damping=0.5,
         max_linear_velocity=1000.0, # m/s in both IssacGym and Isaac Lab
         max_angular_velocity=64*(3.14/180), # rad/s in Isaac Lab and deg/s in IsaacGym
     ),
     activate_contact_sensors=False, # [Comment by Kowndinya, 02 Jan, 2025]: Might need to set to True to avoid collisions, read more and verify
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.0),
        joint_pos={
            'shoulder_pan_joint': -0.662, 
            'shoulder_lift_joint': -2.312,
            'elbow_joint': 1.822, 
            'wrist_1_joint': -1.543, 
            'wrist_2_joint': -1.228, 
            'wrist_3_joint': 2.562, 
            'left_inner_finger_pad_joint': 0.0425, 
            'right_inner_finger_pad_joint': 0.0425
        }
    ),
    actuators={
        # [Comment by Kowndinya, 02 Jan 2025] Might need to separately configure the gripper actuator
        # zero passive stiffness for the arm
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], # [Comment by Kowndinya, 02 Jan 2025] All joints except I guess
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=0.0, # p-gain
            damping=20.0, # d-gain
        ),
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_pad_joint", "right_inner_finger_pad_joint"], # [Comment by Kowndinya, 02 Jan 2025] Gripper Joints I guess
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=1000.0,
            damping=80.0,
        )
    }
)

""" Configuration of UR5e arm + gripper using implicit PD actuator models. """