tomato🍅


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.evaluation import evaluate_policy

from custom_env import ProductionLineEnv
import pandas as pd
import numpy as np


initial_equipments = pd.DataFrame({"EQP_ID": ["E1", "E2", "E3"]})
initial_lots = pd.DataFrame({"LOT_ID": ["L1", "L2", "L3", "L4", "L5", "L6"]})
initial_st = pd.DataFrame({
    "EQP_ID": ["E1", "E1", "E1", "E1", "E1", "E1",
               "E2", "E2", "E2", "E2", "E2", "E2",
               "E3", "E3", "E3", "E3", "E3", "E3"],
    "LOT_ID": ["L1", "L2", "L3", "L4", "L5", "L6",
               "L1", "L2", "L3", "L4", "L5", "L6",
               "L1", "L2", "L3", "L4", "L5", "L6"],
    "ST": [10, 10, 20, 20, 20, 20,
           20, 20, 10, 30, 30, 30,
           30, 30, 30, 10, 10, 10]
})


env = ProductionLineEnv(initial_equipments, initial_lots, initial_st)
# check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

evaluate_policy(model, env, n_eval_episodes=1)

model.save("custom")
