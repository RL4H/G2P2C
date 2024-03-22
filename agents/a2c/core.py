import math
import numpy as np
import torch
from collections import deque
from utils import core
from utils.core import custom_reward, custom_reward_traj


class Memory:
    def __init__(self, args, device):
        self.size = args.n_step
        self.device = device
        self.feature_hist = args.feature_history
        self.features = args.n_features
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.handcrafted_features = args.n_handcrafted_features
        self.observation = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def store(self, obs, act, rew, val, logp, counter):
        assert self.ptr < self.max_size
        self.observation[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = True if counter == 0 else False
        self.ptr += 1

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.observation, act=self.actions, v_pred=self.state_values,
                    logp=self.logprobs, first_flag=self.first_flag, reward=self.rewards)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}


class StateSpace:
    def __init__(self, args):
        self.feature_history = args.feature_history
        self.glucose = deque(self.feature_history*[0], self.feature_history)
        self.insulin = deque(self.feature_history*[0], self.feature_history)
        self.glucose_max = args.glucose_max
        self.glucose_min = args.glucose_min
        self.insulin_max = args.insulin_max
        self.insulin_min = args.insulin_min
        self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)

    def update(self, cgm=0, ins=0):
        cgm = core.linear_scaling(x=cgm, x_min=self.glucose_min, x_max=self.glucose_max)
        ins = core.linear_scaling(x=ins, x_min=self.insulin_min, x_max=self.insulin_max)
        self.glucose.append(cgm)  # self.glucose.appendleft(cgm)
        self.insulin.append(ins)
        self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)
        return self.state


def get_IS_Rew(state, IS):
    REWARD_FACTOR = 0.1
    EXP_FACTOR = 1
    reward = custom_reward([state])
    IS_val = (1 -(math.exp(EXP_FACTOR * (IS - 0.5)))) if IS < 0 else (1 + IS)
    reward = (reward * IS_val) if state > 112.5 else (reward * (1 / max(IS_val, 1e-5)))
    reward = reward + (REWARD_FACTOR * custom_reward([state]))
    return reward


def composite_reward(args, state=None, reward=None):
    MAX_GLUCOSE = 600
    if reward == None:
        reward = custom_reward([state])
    x_max, x_min = 0, custom_reward([MAX_GLUCOSE]) #get_IS_Rew(MAX_GLUCOSE, 4) # custom_reward([MAX_GLUCOSE])
    reward = ((reward - x_min) / (x_max - x_min))
    if state <= 40:
        reward = -15
    elif state >= MAX_GLUCOSE:
        reward = 0
    else:
        reward = reward
    return reward
