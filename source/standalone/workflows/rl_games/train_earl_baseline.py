## Author: Kowndinya Boyalakuntla
## Date: 01 Jan, 2025
## Purpose: Migrate EARL from IsaacGym to Isaac Lab

"""Script to train RL Agent with RL-Games"""


"""Launch Isaac Sim Simulator first.""""

import argparse
import sys 

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="UR5VSGTaskPPO", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Append AppLauncher CLI arguments
AppLauncher.add_app_launcher_args(parser)

#parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video 

if args_cli.video:
    args_cli.enable_cameras = True

# clear out the sys.argv so that hydra doesn't get confused
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app 

import gymnasium as gym
import math 
import os 
import random
from datetime import datetime
from omni.isaac.lab.utils.dict import print_dict, omegaconf_to_dict