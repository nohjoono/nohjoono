```
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

from equipment import Equipment
from lot import Lot

class CustomEnv(gym.Env):
    def __init__(self, init_data=None, num_eqp=10, num_lot=10, dim_eqp=10, dim_lot=10):
        super(CustomEnv, self).__init__()

        # Number fo equipments and lots
        self.init_data = init_data
        self.dim_eqp = dim_eqp
        self.dim_lot = dim_lot
        self.num_eqp = num_eqp if num_eqp != 0 else len(self.initial_equipments)
        self.num_lot = num_lot if num_lot != 0 else len(self.initial_lots)
        self.equipments = []
        self.lots = []
        self.st = []

        self.episode_cnt = 0
        self.step_cnt = 0

        # Action space
        self.action_space = Discrete(dim_eqp * dim_lot)

        # Observation space
        self.observation_space = Dict(
            # eqp_status=Box(0, 1, shape=(dim_eqp,), dtype=bool),
            eqp_end_time=Box(0, 1, shape=(dim_eqp,), dtype=float),
            # lot_status=Box(0, 1, shape=(dim_lot,), dtype=bool),
            st=Box(20, 40, shape=(dim_eqp * dim_lot,), dtype=np.int32),
        )

        # Initialize
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # print("reset")
        # self.episode_cnt = 0
        self.step_cnt = 0
        self.cmax = 0.0
        self.reward = 0.0

        # 초기화
        self.equipments.clear()
        for i in range(1, self.num_eqp + 1):
            self.equipments.append(Equipment(f"E{i}"))

        self.lots.clear()
        for i in range(1, self.num_lot + 1):
            self.lots.append(Lot(f"L{i}"))

        self.st = np.zeros(self.dim_eqp * self.dim_lot, dtype=np.int8)
        self.st[:self.num_eqp * self.num_lot] = np.random.randint(20, 41, self.num_eqp * self.num_lot)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        self.step_cnt += 1
        equipment_index = action // self.num_lot
        lot_index = action % self.num_lot

        equipment = self.equipments[equipment_index]
        lot = self.lots[lot_index]

        # print(f"action no : {action}, eqp : {equipment.id}, lot : {lot.id}")

        production_time = self.st[action]
        equipment.end_time += production_time
        equipment.history.append(lot.id)
        
        lot.status = "done"
        lot.eqp = equipment.id

        observation = self._get_observation()
        reward = 0
        # reward += float(production_time * -1.0)
        # reward -= float(self._get_reward_std())
        reward += float(self._get_reward_cmax())
        terminated = all(lot.status == "done" for lot in self.lots)
        truncated = False

        self.reward += reward

        if terminated:
            self.episode_cnt += 1
            # self.render()

        return observation, reward, terminated, truncated, {}

    def action_masks(self):
        action_mask = np.full(self.dim_eqp * self.dim_lot, False, dtype=bool)

        equipment_mask = np.array([True for eqp in self.equipments], dtype=bool)
        lot_mask = np.array([lot.status == "wait" for lot in self.lots], dtype=bool)
        run_action_mask = np.outer(equipment_mask, lot_mask).flatten()

        action_mask[:len(run_action_mask)] = run_action_mask
        
        return action_mask

    def _get_observation(self):
        # 장비 상태
        eqp_status = np.zeros(self.dim_eqp, dtype=bool)
        run_eqp_status = np.array([True if eqp.status == "run" else False for eqp in self.equipments], dtype=bool)
        eqp_status[:self.num_eqp] = run_eqp_status

        # 장비의 최대 종료 시간을 기준으로 정규화
        max_end_time = max(eqp.end_time for eqp in self.equipments)
        if max_end_time == 0:
            max_end_time = 1
        
        # 장비 종료 시간
        eqp_end_times = np.zeros(self.dim_eqp, dtype=float)
        run_eqp_end_times  = np.array([eqp.end_time / max_end_time for eqp in self.equipments], dtype=float)
        eqp_end_times[:self.num_eqp] = run_eqp_end_times

        # 재공 상태
        lot_status = np.zeros(self.dim_lot, dtype=bool)
        run_lot_status = np.array([True if lot.status == "wait" else False for lot in self.lots], dtype=bool)
        lot_status[:self.num_lot] = run_lot_status

        observation = {
            # "eqp_status" : eqp_status,
            "eqp_end_time": eqp_end_times,
            # "lot_status": lot_status,
            "st": self.st,
        }
        
        return observation

    def _get_reward_cmax(self):
        new_cmax = max(eqp.end_time for eqp in self.equipments)
        reward = self.cmax - new_cmax
        self.cmax = new_cmax
        return float(reward)
    
    def _get_reward_std(self):
        eqp_end_times = np.array([eqp.end_time for eqp in self.equipments])
        reward = np.std(eqp_end_times)
        return reward
        

    

    def render(self):
        print(f"------------------------------------------------------------------------")
        print(f"@ episode: {self.episode_cnt}, step: {self.step_cnt}")
        print(f"@ EQP Info")
        for eqp in self.equipments:
            print(f"{eqp.id}: 종료시간={eqp.end_time}, History={eqp.history}")
        # print("\n@ Lot Info")
        # for lot in self.lots:
        #     print(f"{lot.id}: 상태={lot.status}, eqp={lot.eqp}")
        # print(self.st)
        print(f"@ reward: {self.reward}")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from custom_env import CustomEnv
from custom_vec_env import CustomVecEnv

num_eqp = 3
num_lot = 7
dim_eqp = 3
dim_lot = 10
tb_log_name = f'({num_eqp},{num_lot})-({dim_eqp},{dim_lot})'

env = CustomEnv(init_data=None, num_eqp=num_eqp, num_lot=num_lot, dim_eqp=dim_eqp, dim_lot=dim_lot)
env = Monitor(env, filename=None)

model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/")
total_timesteps = 1000 * 1000

model.learn(total_timesteps, tb_log_name=tb_log_name)
model.save("./model/MaskablePPO")
```
