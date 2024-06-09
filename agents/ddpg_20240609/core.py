import math
import torch
import numpy as np
from collections import deque
from utils import core
from utils.core import custom_reward, custom_reward_traj


class CGPredHorizon:
    def __init__(self, args):
        self.horizon = args.planning_n_step
        self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
        self.actions = np.zeros(self.horizon, dtype=np.float32)
        self.cur_state = 0
        self.feat = 0
        self.counter = 0

    def update(self, cur_state, feat, action, real_glucose, policy):
        done, err = False, 0
        if self.counter == 0:
            self.cur_state = cur_state
            self.feat = feat
        self.actions[self.counter] = action
        self.real_glucose[self.counter] = real_glucose
        self.counter += 1
        if self.counter == self.horizon:
            done = True
            err = policy.Actor.horizon_error(self.cur_state, self.feat, self.actions, self.real_glucose,
                                                            mode='forward')
            self.actions = np.zeros(self.horizon, dtype=np.float32)
            self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
            self.counter = 0
        return done, err

    def reset(self):
        self.actions = np.zeros(self.horizon, dtype=np.float32)
        self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
        self.counter = 0


class BGPredBuffer:
    def __init__(self, args):
        self.n_bgp_steps = args.n_bgp_steps
        self.args = args
        self.actor_predictions = []
        self.critic_predictions = []
        self.real_glucose = []

    def clear(self):
        self.actor_predictions = []
        self.critic_predictions = []
        self.real_glucose = []

    def update(self, act_pred, critic_pred, real):
        act_pred = act_pred.flatten()
        critic_pred = critic_pred.flatten()
        for i in range(0, len(act_pred)):
            act_pred[i] = core.inverse_linear_scaling(y=act_pred[i], x_min=self.args.glucose_min, x_max=self.args.glucose_max)
            critic_pred[i] = core.inverse_linear_scaling(y=critic_pred[i], x_min=self.args.glucose_min, x_max=self.args.glucose_max)
        self.actor_predictions.append(act_pred)
        self.critic_predictions.append(critic_pred)
        self.real_glucose.append(real)

    def calc_accuracy(self):
        a_rmse_error, c_rmse_error, count = 0, 0, 0
        if len(self.real_glucose) > self.n_bgp_steps:  # check if atleast bg steps more than pred horizon
            for i in range(0, len(self.real_glucose)-self.n_bgp_steps):
                count += 1
                a_rmse_error += np.sum(np.square(self.actor_predictions[i] - self.real_glucose[i:i+self.n_bgp_steps]))
                c_rmse_error += np.sum(np.square(self.critic_predictions[i] - self.real_glucose[i:i + self.n_bgp_steps]))
        else:
            a_rmse_error, c_rmse_error, count = 0, 0, 1
        return np.sqrt(a_rmse_error/(self.n_bgp_steps*count)), np.sqrt(c_rmse_error/(self.n_bgp_steps*count))

    def calc_simple_rmse(self):
        a_rmse_error, c_rmse_error, count = 0, 0, len(self.real_glucose)
        if len(self.real_glucose) > 0:
            for i in range(0, len(self.real_glucose)):
                a_rmse_error += (np.square(self.actor_predictions[i] - self.real_glucose[i]))
                c_rmse_error += (np.square(self.critic_predictions[i] - self.real_glucose[i]))
        else:
            a_rmse_error, c_rmse_error = 0, 0
            return a_rmse_error, c_rmse_error
        return np.sqrt(a_rmse_error / (count))[0], np.sqrt(c_rmse_error / (count))[0]


class AuxiliaryBuffer:
    def __init__(self, args, device):
        self.size = args.aux_buffer_max
        self.n_bgp_steps = args.n_bgp_steps
        self.bgp_pred_mode = args.bgp_pred_mode
        self.old_states = torch.zeros(self.size, args.feature_history, args.n_features, device=device, dtype=torch.float32)
        self.handcraft_feat = torch.zeros(self.size, 1, args.n_handcrafted_features, device=device, dtype=torch.float32)
        self.actions = torch.zeros(self.size, 1, device=device, dtype=torch.float32)
        self.logprob = torch.zeros(self.size, 1, device=device, dtype=torch.float32)
        self.value_target = torch.zeros(self.size, device=device, dtype=torch.float32)
        self.aux_batch_size = args.aux_batch_size
        self.device = device
        self.buffer_filled = False
        self.buffer_level = 0

        if self.bgp_pred_mode:
            self.cgm_target = torch.zeros(self.size, 1, self.n_bgp_steps, device=device ,dtype=torch.float32)
        else:
            self.cgm_target = torch.zeros(self.size, 1, device=device, dtype=torch.float32)

    def update(self, s, hand_feat, cgm_target, actions, first_flag):
        if self.bgp_pred_mode:
            s, hand_feat, actions, cgm_target = self.prepare_bgp_prediction(s, hand_feat, cgm_target, actions, first_flag)
            update_size = actions.shape[0]  # this size is lesser then rollout samples.
            self.old_states = torch.cat((self.old_states[update_size:, :, :], s), dim=0)
            self.handcraft_feat = torch.cat((self.handcraft_feat[update_size:, :, :], hand_feat), dim=0)
            self.cgm_target = torch.cat((self.cgm_target[update_size:, :, :], cgm_target), dim=0)
            self.actions = torch.cat((self.actions[update_size:], actions), dim=0)
        else:  # normal buffer updating approach.
            cgm_target = cgm_target.view(-1, 1)
            update_size = actions.shape[0]
            self.old_states = torch.cat((self.old_states[update_size:, :, :], s), dim=0)
            self.handcraft_feat = torch.cat((self.handcraft_feat[update_size:, :, :], hand_feat), dim=0)
            self.cgm_target = torch.cat((self.cgm_target[update_size:, :], cgm_target), dim=0)
            self.actions = torch.cat((self.actions[update_size:], actions), dim=0)

        if not self.buffer_filled:
            self.buffer_level += update_size
            if self.buffer_level >= self.size:
                self.buffer_filled = True

        if update_size > self.size:
            print('The auxilliary update at rollout is larger than MAX buffer size!')
            exit()

        assert self.old_states.shape[0] == self.size
        assert self.handcraft_feat.shape[0] == self.size
        assert self.cgm_target.shape[0] == self.size
        assert self.actions.shape[0] == self.size

    def prepare_bgp_prediction(self, s_hist, s_handcraft, cgm_target, act, first_flag):
        buffer_len = s_hist.shape[0]
        bgp_first_flag = first_flag.view(-1).cpu().numpy()
        bgp_cgm_target = cgm_target.cpu().numpy()
        bgp_s_hist = s_hist.cpu().numpy()
        bgp_s_handcraft = s_handcraft.cpu().numpy()
        bgp_act = act.cpu().numpy()
        new_cgm_target = np.zeros(core.combined_shape(buffer_len, (1, self.n_bgp_steps)), dtype=np.float32)
        delete_arr = list(range((buffer_len - self.n_bgp_steps + 1), buffer_len))
        for ii in range(0, buffer_len - (self.n_bgp_steps - 1)):
            flag_status = np.sum(bgp_first_flag[ii + 1:ii + self.n_bgp_steps])  # future steps cant have flag = 1
            new_cgm_target[ii] = bgp_cgm_target[ii:ii + self.n_bgp_steps]
            if flag_status >= 1:
                delete_arr.append(ii)
        bgp_s_hist = torch.from_numpy(np.delete(bgp_s_hist, delete_arr, axis=0)).to(self.device)
        bgp_s_handcraft = torch.from_numpy(np.delete(bgp_s_handcraft, delete_arr, axis=0)).to(self.device)
        bgp_act = torch.from_numpy(np.delete(bgp_act, delete_arr, axis=0)).to(self.device)
        new_cgm_target = torch.from_numpy(np.delete(new_cgm_target, delete_arr, axis=0)).to(self.device)
        return bgp_s_hist, bgp_s_handcraft, bgp_act, new_cgm_target

    def update_targets(self, policy):
        # calculate the new targets for value and log prob.
        # done batch wise to reduce memory, aux batch size is used.
        start_idx = 0
        while start_idx < self.size:
            end_idx = min(start_idx + self.aux_batch_size, self.size)
            state_batch = self.old_states[start_idx:end_idx, :, :]
            handcraft_feat_batch = self.handcraft_feat[start_idx:end_idx, :, :]
            actions_old_batch = self.actions[start_idx:end_idx]
            value_predict, _, _ = policy.evaluate_critic(state_batch, handcraft_feat_batch, action=None, cgm_pred=False)
            logprobs, _, _, _ = policy.evaluate_actor(state_batch, actions_old_batch, handcraft_feat_batch)
            self.logprob[start_idx:end_idx, :] = logprobs.detach()
            self.value_target[start_idx:end_idx] = value_predict.detach()
            start_idx += self.aux_batch_size
        assert self.value_target.shape[0] == self.size
        assert self.logprob.shape[0] == self.size


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
        self.next_observation = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.next_state_features = np.zeros(core.combined_shape(self.size, (1, self.handcrafted_features)), dtype=np.float32)
        self.done = np.zeros(self.size, dtype=np.bool_)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def store(self, obs, features, act, rew, next_obs, next_features, done):
        assert self.ptr < self.max_size
        self.observation[self.ptr] = obs
        self.state_features[self.ptr] = features
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_observation[self.ptr] = next_obs
        self.next_state_features[self.ptr] = next_features
        self.done[self.ptr] = True if done == 1 else False
        self.ptr += 1

    # def finish_path(self, final_v):
    #     self.next_observation[self.ptr] = final_v
    #     self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.observation, feat=self.state_features, act=self.actions, next_state=self.next_observation,
                    next_feat=self.next_state_features, done=self.done, reward=self.rewards)
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


def exponential_reward(args, prev_state=None, new_state=None):
    prev_sum, new_sum = 0, 0
    size = len(new_state)
    for t in range(0, size):
        x = t+1
        factor = 1  # math.exp((x-size)/10)
        prev_sum += (factor * composite_reward(args, state=prev_state[t]))
        new_sum += (factor * composite_reward(args, state=new_state[t]))
    return new_sum - prev_sum

def differential_reward(args, latest_cgm=None, start_cgm=None, reward=None):
    latest_reward = composite_reward(args, state=latest_cgm, reward=reward)
    start_rew = composite_reward(args, state=start_cgm, reward=reward)
    return latest_reward - start_rew

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

def traj_reward(args, bg_hist, k):
    # reward = custom_reward_traj(bg_hist, k)
    reward = 0
    for i in bg_hist:
        r = custom_reward([i])
        reward += composite_reward(args, state=i, reward=r, steps=None)
    return reward


