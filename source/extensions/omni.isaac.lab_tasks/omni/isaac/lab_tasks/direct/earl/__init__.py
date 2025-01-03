"""
EARL Gym Environment Registration
"""

import gymnasium as gym

from . import agents

gym.register(
    id="EARL-UR5e-Direct-v0",
    entry_point=f"{__name__}.earl"