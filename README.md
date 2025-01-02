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



import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Equipment:
    def __init__(self, id, status="run"):
        self.id = id
        self.status = status
        self.end_time = 0


class Lot:
    def __init__(self, id, status="wait"):
        self.id = id
        self.status = status


class ProductionLineEnv(gym.Env):
    def __init__(self, initial_equipments, initial_lots, initial_st):
        super(ProductionLineEnv, self).__init__()

        # 장비 초기화
        self.equipments = [Equipment(id=eq_id) for eq_id in initial_equipments["EQP_ID"]]

        # 재공 초기화
        self.lots = [Lot(id=lot_id) for lot_id in initial_lots["LOT_ID"]]

        # 생산 시간 DataFrame 초기화
        self.st = initial_st.set_index(["EQP_ID", "LOT_ID"])

        # 현재 시간 초기화
        self.step_cnt = 0

        # Action space: 장비와 재공 선택
        self.action_space = spaces.MultiDiscrete([len(self.equipments), len(self.lots)])

        # Observation space: 각 장비 상태, 각 장비의 종료 시간, 각 재공의 상태
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.equipments) * 2 + len(self.lots),), dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_cnt = 0

        for eq in self.equipments:
            eq.status = "run"
            eq.end_time = 0

        for lot in self.lots:
            lot.status = "wait"

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        self.step_cnt += 1
        equipment_index, lot_index = action
        equipment = self.equipments[equipment_index]
        lot = self.lots[lot_index]
        print("action : " + equipment.id + ", " + lot.id)
        reward = 0.0  # 보상 초기화

        if (equipment.id, lot.id) in self.st.index:
            production_time = self.st.loc[
                (equipment.id, lot.id), "ST"
            ]
            equipment.end_time += production_time
            lot.status = "done"
            reward -= float(production_time)  # 생산 시간에 대한 패널티
        else:
            reward = -20.0  # 생산 시간 정보가 없음

        self.render()

        terminated = all(lot.status == "done" for lot in self.lots)
        if terminated:
            print("All lots are done-----------------------------------------------------")
        truncated = terminated
        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def __action_masks(self):
        equipment_mask = np.array(
            [eqp.id == "E2" for eqp in self.equipments], dtype=bool
        )
        lot_mask = np.array([lot.status == "wait" for lot in self.lots], dtype=bool)
        action_mask = np.outer(equipment_mask, lot_mask).flatten()
        print("Action Mask Shape:", action_mask.shape)
        print("Action Mask:\n", action_mask)  # 추가된 print 문
        return action_mask

    def render(self, mode="human"):
        print(f"현재 step: {self.step_cnt}")
        for eqp in self.equipments:
            print(f"  장비 {eqp.id}: {eqp.status}, 종료 시간: {eqp.end_time}")
        for lot in self.lots:
            print(f"  재공 {lot.id}: 상태 {lot.status}")

    def _get_observation(self):
        Equipment_status = np.array(
            [1 if eqp.status == "run" else 0 for eqp in self.equipments],
            dtype=np.float32,
        )
        equipment_end_times = np.array(
            [eqp.end_time for eqp in self.equipments], dtype=np.float32
        )
        lot_status = np.array(
            [1 if lot.status == "done" else 0 for lot in self.lots], dtype=np.float32
        )
        return np.concatenate(
            (Equipment_status, equipment_end_times, lot_status), dtype=np.float32
        )
