import torch
import numpy as np
from utils import core


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

