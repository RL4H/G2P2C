import csv
import numpy as np
import pandas as pd
from collections import deque
from utils.pumpAction import Pump
from utils.core import get_env, time_in_range, custom_reward, combined_shape, linear_scaling, inverse_linear_scaling
from agents.g2p2c.core import Memory, BGPredBuffer, CGPredHorizon
from utils.statespace import StateSpace
from utils.reward_func import composite_reward


class Worker:
    def __init__(self, args, mode, patients, env_ids, seed, worker_id, device):
        self.args = args
        self.episode = 0
        self.worker_mode = mode
        self.worker_id = worker_id
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
        self.bgp_buffer = BGPredBuffer(self.args)
        self.CGPredHorizon = CGPredHorizon(self.args)
        self.episode_history = np.zeros(combined_shape(self.max_epi_length, 13), dtype=np.float32)
        self.reinit_flag = False
        self.init_env()
        self.log1_columns = ['epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins', 'mu', 'sigma',
                             'prob', 'state_val', 'day_hour', 'day_min']
        self.log2_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
        self.save_log([self.log1_columns], '/'+self.worker_mode+'/data/logs_worker_')
        self.save_log([self.log2_columns], '/'+self.worker_mode+'/data/'+self.worker_mode+'_episode_summary_')

    def init_env(self):
        if not self.reinit_flag:
            self.episode += 1
        self.counter = 0
        self.init_state = self.env.reset()
        init_cgm = self.init_state.observation.CGM
        self.cur_state, self.feat = self.state_space.update(cgm=init_cgm, ins=0, meal=0)
        self.pump.calibrate(self.init_state)
        self.calibration_process()

    def calibration_process(self):
        self.reinit_flag, cur_cgm = False, 0
        for t in range(0, self.calibration):  # open-loop simulation for calibration period.
            state, reward, is_done, info = self.env.step(self.std_basal)
            cur_cgm = state.CGM
            self.cur_state, self.feat = self.state_space.update(cgm=cur_cgm, ins=self.std_basal,
                                                                meal=info['remaining_time'], hour=self.counter,
                                                                meal_type=info['meal_type'])  # info['day_hour']
            self.reinit_flag = True if info['meal_type'] != 0 else False  # meal_type zero -> no meal
        if (cur_cgm < 110 or 130 < cur_cgm) and self.worker_mode != 'training':  # checking simulati start within normo
            self.reinit_flag = True
        if self.reinit_flag:
            self.init_env()

    def rollout(self, policy):
        ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper = 0, 0, 0, 0, 0, 0, 0, 0, 0
        aBGpred_rmse, a_horizonBG_rmse, horizon_rmse_count = -1, 0, 0
        self.bgp_buffer.clear()
        self.CGPredHorizon.reset()
        if self.worker_mode != 'training':  # fresh env for testing
            self.init_env()
        rollout_steps = self.update_timestep if self.worker_mode == 'training' else self.max_test_epi_len

        for n_steps in range(0, rollout_steps):
            policy_step = policy.get_action(self.cur_state, self.feat)
            selected_action = policy_step['action'][0]
            rl_action, pump_action = self.pump.action(agent_action=selected_action,
                                                      prev_state=self.init_state, prev_info=None)
            state, _reward, is_done, info = self.env.step(pump_action)
            cur_cgm = state.CGM
            reward = composite_reward(self.args, state=cur_cgm, reward=_reward)
            self.bgp_buffer.update(policy_step['a_cgm'], policy_step['c_cgm'], cur_cgm)
            # calulate the horison pred error rmse
            horizon_calc_done, err = self.CGPredHorizon.update(self.cur_state, self.feat, policy_step['action'][0],
                                                               cur_cgm, policy)
            if horizon_calc_done:
                a_horizonBG_rmse += err[0]
                horizon_rmse_count += 1

            if self.worker_mode == 'training':   # store -> rollout for training
                scaled_cgm = linear_scaling(x=cur_cgm, x_min=self.args.glucose_min, x_max=self.args.glucose_max)
                self.memory.store(self.cur_state, self.feat, policy_step['action'][0],
                                  reward, policy_step['state_value'], policy_step['log_prob'], scaled_cgm, self.counter)
            # update -> state.
            self.cur_state, self.feat = self.state_space.update(cgm=cur_cgm, ins=pump_action,
                                                                meal=info['remaining_time'], hour=(self.counter+1),
                                                                meal_type=info['meal_type'], carbs=info['future_carb']) #info['day_hour']
            self.episode_history[self.counter] = [self.episode, self.counter, cur_cgm, info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0], policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0], info['day_hour'],
                                                  info['day_min']]
            self.counter += 1
            stop_factor = (self.max_epi_length - 1) if self.worker_mode == 'training' else (self.max_test_epi_len - 1)

            criteria = cur_cgm <= 40 or cur_cgm >= 600 or self.counter > stop_factor
            if criteria:  # episode termination criteria.
                if self.worker_mode == 'training':
                    final_val = policy.get_final_value(self.cur_state, self.feat)
                    self.memory.finish_path(final_val)

                df = pd.DataFrame(self.episode_history[0:self.counter], columns=self.log1_columns)
                df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                          mode='a', header=False, index=False)
                alive_steps = self.counter
                aBGpred_rmse, cBGpred_rmse = self.bgp_buffer.calc_simple_rmse()
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             self.episode, self.counter, display=False)
                self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                hgbi, ri, sev_hyper, aBGpred_rmse, cBGpred_rmse]],
                              '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')
                
                # === 행동 통계 로깅 추가 시작 ===
                actions_this_episode = df['rl_ins'].to_numpy() # 'rl_ins' 컬럼 사용 (에이전트의 최종 RL 행동)
                action_mean = np.mean(actions_this_episode) if len(actions_this_episode) > 0 else np.nan
                action_std = np.std(actions_this_episode) if len(actions_this_episode) > 0 else np.nan
                action_min = np.min(actions_this_episode) if len(actions_this_episode) > 0 else np.nan
                action_max = np.max(actions_this_episode) if len(actions_this_episode) > 0 else np.nan

                print(f"DEBUG: [Worker {self.worker_id}, Epi {self.episode}] Action Stats: Mean={action_mean:.3f}, Std={action_std:.3f}, Min={action_min:.3f}, Max={action_max:.3f}")

                # 기존 요약 로그에 행동 통계 추가 (log2_columns 및 save_log 호출 수정 필요)
                # 예: self.log2_columns에 'action_mean', 'action_std', 'action_min', 'action_max' 추가
                # self.save_log 호출 시 해당 값들 전달
                # current_summary_data = [self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                #                         hgbi, ri, sev_hyper, aBGpred_rmse, cBGpred_rmse,
                #                         action_mean, action_std, action_min, action_max] # 예시
                # self.save_log([current_summary_data], 
                #               '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')
                # === 행동 통계 로깅 추가 끝 ===

                # 기존 save_log 호출은 위에서 행동 통계를 포함하여 한 번만 하도록 수정하거나,
                # 혹은 행동 통계는 별도의 로그 파일/방식으로 기록
                if not np.isnan(action_mean): # 행동 통계가 유효할 때만 기존 로그에 추가한다고 가정
                     self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                   hgbi, ri, sev_hyper, aBGpred_rmse, cBGpred_rmse,
                                   action_mean, action_std, action_min, action_max]], # 새 컬럼 추가
                                 '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')
                     # self.log2_columns도 이에 맞게 수정 필요:
                     # self.log2_columns = ['epi', 't', ..., 'cBGP_rmse', 'act_mean', 'act_std', 'act_min', 'act_max']
                else: # 행동 통계가 NaN이면 기존 방식대로 로그 (컬럼 수 불일치 방지)
                     self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                   hgbi, ri, sev_hyper, aBGpred_rmse, cBGpred_rmse]],
                                 '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')


                if self.worker_mode == 'training':
                    self.init_env()
                else:
                    break  # stop rollout if this is a testing worker!

        aBGpred_rmse, _ = self.bgp_buffer.calc_simple_rmse()
        a_horizonBG_rmse = np.sqrt(a_horizonBG_rmse / horizon_rmse_count) if horizon_rmse_count != 0 else 0
        if self.worker_mode == 'training':
            data = self.memory.get()
        else:
            data = [ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper]
        return data, aBGpred_rmse, a_horizonBG_rmse

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + str(self.worker_id) + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()
