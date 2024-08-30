import csv
import gym
import torch
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import deque
from utils.pumpAction import Pump
from utils.core import get_env, time_in_range, custom_reward, combined_shape
from agents.td3.core import Memory, StateSpace, composite_reward
from agents.std_bb.BBController import BasalBolusController
from utils.carb_counting import carb_estimate


class Worker:
    def __init__(self, args, mode, patients, env_ids, seed, worker_id, device):
        self.worker_id = worker_id
        self.worker_mode = mode
        self.args = args
        self.device = device
        self.episode = 0
        self.update_timestep = args.n_step
        self.max_test_epi_len = args.max_test_epi_len
        self.max_epi_length = args.max_epi_length
        self.calibration = args.calibration
        self.simulation_seed = seed + 100
        self.patient_name = patients[args.patient_id]
        self.env_id = str(worker_id) + '_' + env_ids[args.patient_id]
        self.env = get_env(self.args, patient_name=self.patient_name, env_id=self.env_id,
                           custom_reward=custom_reward, seed=self.simulation_seed)
        self.state_space = StateSpace(self.args)
        self.pump = Pump(self.args, patient_name=self.patient_name)
        self.std_basal = self.pump.get_basal()
        self.memory = Memory(self.args, device)
        self.episode_history = np.zeros(combined_shape(self.max_epi_length, 14), dtype=np.float32)
        self.reinit_flag = False
        self.init_env()
        self.log1_columns = ['epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins', 'mu', 'sigma',
                             'prob', 'state_val', 'day_hour', 'day_min', 'IS']
        self.log2_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
        self.save_log([self.log1_columns], '/'+self.worker_mode+'/data/logs_worker_')
        self.save_log([self.log2_columns], '/'+self.worker_mode+'/data/'+self.worker_mode+'_episode_summary_')

    def init_env(self):
        if not self.reinit_flag:
            self.episode += 1
        self.counter = 0
        self.init_state = self.env.reset()
        self.cur_state, self.feat = self.state_space.update(cgm=self.init_state.CGM, ins=0, meal=0)
        self.cgm_hist = deque(self.calibration * [0], self.calibration)
        self.ins_hist = deque(self.calibration * [0], self.calibration)
        self.cgm_hist.append(self.init_state.CGM)
        self.pump.calibrate(self.init_state)
        self.calibration_process()

    def calibration_process(self):
        self.reinit_flag, cur_cgm = False, 0
        for t in range(0, self.calibration):  # open-loop simulation for calibration period.
            state, reward, is_done, info = self.env.step(self.std_basal)
            cur_cgm = state.CGM
            self.cgm_hist.append(state.CGM)
            self.ins_hist.append(self.std_basal)
            self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=self.std_basal,
                                                                meal=info['remaining_time'], hour=self.counter,
                                                                meal_type=info['meal_type']) #info['day_hour']
            self.reinit_flag = True if info['meal_type'] != 0 else False  # meal_type zero -> no meal
        if (cur_cgm < 110 or 130 < cur_cgm) and self.worker_mode != 'training':  # checking simulation start within normo
            self.reinit_flag = True
        if self.reinit_flag:
            self.init_env()

    def rollout(self, td3, replay_memory):
        ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper = 0, 0, 0, 0, 0, 0, 0, 0, 0
        if self.worker_mode != 'training':  # fresh env for testing
            self.init_env()
        rollout_steps = self.update_timestep if self.worker_mode == 'training' else self.max_test_epi_len

        for n_steps in range(0, rollout_steps):
            policy_step, mu, sigma = td3.get_action(self.cur_state, self.feat, worker_mode=self.worker_mode)
            selected_action = policy_step[0]
            rl_action, pump_action = self.pump.action(agent_action=selected_action, prev_state=self.init_state, prev_info=None)
            state, reward, is_done, info = self.env.step(pump_action)
            reward = composite_reward(self.args, state=state.CGM, reward=reward)
            this_state = deepcopy(self.cur_state)
            this_feat = deepcopy(self.feat)
            done_flag = 1 if state.CGM <= 40 or state.CGM >= 600 else 0
            # update -> state.
            self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=pump_action,
                                                                meal=info['remaining_time'], hour=(self.counter+1),
                                                                meal_type=info['meal_type'], carbs=info['future_carb'])

            if self.worker_mode == 'training':
                replay_memory.push(torch.as_tensor(this_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                                   torch.as_tensor(this_feat, dtype=torch.float32, device=self.device).unsqueeze(0),
                                   torch.as_tensor([selected_action], dtype=torch.float32, device=self.device),
                                   torch.as_tensor([reward], dtype=torch.float32, device=self.device),
                                   torch.as_tensor(self.cur_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                                   torch.as_tensor(self.feat, dtype=torch.float32, device=self.device).unsqueeze(0),
                                   torch.as_tensor([done_flag], dtype=torch.float32, device=self.device))


            # store -> rollout for training
            # if self.worker_mode == 'training':
            #     self.memory.store(this_state, this_feat, selected_action, reward, self.cur_state, self.feat, done_flag)

            self.episode_history[self.counter] = [self.episode, self.counter, state.CGM, info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, mu[0], sigma[0], 0, 0, info['day_hour'],
                                                  info['day_min'], 0]
            self.counter += 1
            stop_factor = (self.max_epi_length - 1) if self.worker_mode == 'training' else (self.max_test_epi_len - 1)
            criteria = state.CGM <= 40 or state.CGM >= 600 or self.counter > stop_factor  # training or state.CGM >= 400
            if criteria:  # episode termination criteria.
                df = pd.DataFrame(self.episode_history[0:self.counter], columns=self.log1_columns)
                df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                          mode='a', header=False, index=False)
                alive_steps = self.counter
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             self.episode, self.counter, display=False)
                self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                hgbi, ri, sev_hyper, 0, 0]],
                              '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')

                if self.worker_mode == 'training':
                    self.init_env()
                else:
                    break  # stop rollout if this is a testing worker!
        data = [ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper]
        return data

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + str(self.worker_id) + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

