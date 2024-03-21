import math
import torch
import numpy as np
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
        self.state_features = np.zeros(core.combined_shape(self.size, (1, self.handcrafted_features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.cgm_target = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def store(self, obs, features, act, rew, val, logp, cgm_target, counter):
        assert self.ptr < self.max_size
        self.observation[self.ptr] = obs
        self.state_features[self.ptr] = features
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = True if counter == 0 else False
        self.cgm_target[self.ptr] = cgm_target
        self.ptr += 1

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.observation, feat=self.state_features, act=self.actions, v_pred=self.state_values,
                    logp=self.logprobs, first_flag=self.first_flag, reward=self.rewards, cgm_target=self.cgm_target)
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
        self.t_meal = args.t_meal
        self.use_carb_announcement = args.use_carb_announcement
        self.mealAnnounce = args.use_meal_announcement
        self.todAnnounce = args.use_tod_announcement

        if not self.mealAnnounce and not self.todAnnounce:  # only ins and glucose
            self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)
        elif self.todAnnounce:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.tod_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_20 = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_60 = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_120 = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.tod_announcement_arr,
                                   self.hc_iob_20, self.hc_iob_60, self.hc_iob_120), axis=-1).astype(np.float32)
        elif self.use_carb_announcement:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.carb_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.carb_announcement_arr), axis=-1).astype(np.float32)
        else:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            # self.meal_type_arr = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr), axis=-1).astype(np.float32)

    def update(self, cgm=0, ins=0, meal=0, hour=0, meal_type=0, carbs=0):
        cgm = core.linear_scaling(x=cgm, x_min=self.glucose_min, x_max=self.glucose_max)
        ins = core.linear_scaling(x=ins, x_min=self.insulin_min, x_max=self.insulin_max)
        hour = core.linear_scaling(x=hour, x_min=0, x_max=312)  # hour is given 0-23
        t_to_meal = core.linear_scaling(x=meal, x_min=0, x_max=self.t_meal)  #self.t_meal * -2
        snack, main_meal = 0, 0
        if meal_type == 0.3:
            snack = 1
        elif meal_type == 1:
            main_meal = 1

        meal_type = core.linear_scaling(x=meal_type, x_min=0, x_max=1)
        carbs = core.linear_scaling(x=carbs, x_min=0, x_max=120)
        self.glucose.append(cgm)  # self.glucose.appendleft(cgm)
        self.insulin.append(ins)

        # handcrafted features
        ins_20 = sum(self.state[-4:, 1])  # last 20 minreward
        ins_60 = sum(self.state[-12:, 1])  # last 60 min
        ins_120 = sum(self.state[-24:, 1])  # last 120 min

        if not self.mealAnnounce and not self.todAnnounce:
            self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)
        elif self.todAnnounce:
            self.meal_announcement_arr.append(t_to_meal)
            self.tod_announcement_arr.append(hour)
            self.hc_iob_20.append(ins_20)
            self.hc_iob_60.append(ins_60)
            self.hc_iob_120.append(ins_120)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.tod_announcement_arr,
                                   self.hc_iob_20, self.hc_iob_60, self.hc_iob_120), axis=-1).astype(np.float32)
        elif self.use_carb_announcement:
            self.meal_announcement_arr.append(t_to_meal)
            self.carb_announcement_arr.append(carbs)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.carb_announcement_arr), axis=-1).astype(np.float32)
        else:
            self.meal_announcement_arr.append(t_to_meal)
            # self.meal_type_arr.append(meal_type)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr), axis=-1).astype(np.float32)

        #handcraft_features = [cgm, ins, ins_20, ins_60, ins_120, hour, t_to_meal, snack, main_meal]
        handcraft_features = [hour]
        return self.state, handcraft_features


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


