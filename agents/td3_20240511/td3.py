import gc
import gym
import random
import csv
import time
import math
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from utils.reward_normalizer import RewardNormalizer, update_mean_var_count_from_moments
from agents.td3.worker import Worker
from agents.td3.models import ActorCritic
from collections import namedtuple, deque

#python run_RL_agent.py --agent ddpg --folder_id test --patient_id 0 --return_type average --action_type exponential --device cuda --noise_std 0.001 --noise_model normal_dist --seed 1 --debug 1

#python run_RL_agent.py --agent ddpg --folder_id LR/1e-5/DDPG0_1 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-5 --vf_lr 1e-5 --seed 1 --debug 0
#python run_RL_agent.py --agent ddpg --folder_id troubleshoot_20kBuffer_OUNoise/DDPG0_1 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-4 --vf_lr 1e-4 --noise_model ou_noise --noise_std 1e-3  --seed 1 --debug 0


Transition = namedtuple('Transition',
                        ('state', 'feat', 'action', 'reward', 'next_state', 'next_feat', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        # if len(self.memory) + len(Transition(*args)) >= self.capacity:
        #     # Randomly remove one element if memory is full
        #     for i in range(len(Transition(*args))):
        #         remove_index = random.randint(0, len(self.memory) - 1)
        #         self.memory.remove(self.memory[remove_index])
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TD3:
    def __init__(self, args, device, load, path1, path2):
        self.args = args
        self.n_step = args.n_step
        self.feature_history = args.feature_history
        self.n_handcrafted_features = args.n_handcrafted_features
        self.n_features = args.n_features
        self.grad_clip = args.grad_clip

        self.gamma = args.gamma
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.device = device

        self.replay_buffer_size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        self.target_update_interval = 1  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.train_pi_iters = args.n_pi_epochs
        self.shuffle_rollout = args.shuffle_rollout
        # self.soft_q_lr = args.vf_lr
        self.value_lr = args.vf_lr
        self.policy_lr = args.pi_lr
        self.grad_clip = args.grad_clip

        # self.entropy_coef = 0.1  #0.001  # args.entropy_coef
        # self.target_entropy = -1  # 0.001
        # self.entropy_lr = 1e-4 * 3
        # self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * self.entropy_coef).requires_grad_(True)
        # self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.entropy_lr)

        # self.sac_v2 = args.sac_v2

        self.weight_decay = 0

        ### DDPG networks:
        self.td3 = ActorCritic(args, load, path1, path2, device).to(self.device)
        # self.soft_q_criterion1 = nn.MSELoss()
        # self.soft_q_criterion2 = nn.MSELoss()
        self.value_criterion1 = nn.MSELoss()
        self.value_criterion2 = nn.MSELoss()

        # self.soft_q_optimizer1 = torch.optim.Adam(self.ddpg.soft_q_net1.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        # self.soft_q_optimizer2 = torch.optim.Adam(self.ddpg.soft_q_net2.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        self.value_optimizer1 = torch.optim.Adam(self.td3.value_net1.parameters(), lr=self.value_lr,
                                                weight_decay=self.weight_decay)
        self.value_optimizer2 = torch.optim.Adam(self.td3.value_net2.parameters(), lr=self.value_lr,
                                                weight_decay=self.weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.td3.policy_net.parameters(), lr=self.policy_lr,
                                                 weight_decay=self.weight_decay)

        # if self.sac_v2:
        #     for target_param, param in zip(self.ddpg.target_q_net1.parameters(), self.ddpg.soft_q_net1.parameters()):
        #         target_param.data.copy_(param.data)
        #     for target_param, param in zip(self.ddpg.target_q_net2.parameters(), self.ddpg.soft_q_net2.parameters()):
        #         target_param.data.copy_(param.data)
        #     for p in self.ddpg.target_q_net1.parameters():
        #         p.requires_grad = False
        #     for p in self.ddpg.target_q_net2.parameters():
        #         p.requires_grad = False
        # else:
        #     self.value_criterion = nn.MSELoss()
        #     self.value_optimizer = torch.optim.Adam(self.ddpg.value_net.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        #     for target_param, param in zip(self.ddpg.value_net_target.parameters(), self.ddpg.value_net.parameters()):
        #         target_param.data.copy_(param.data)
        #     for p in self.ddpg.value_net_target.parameters():
        #         p.requires_grad = False

        for target_param, param in zip(self.td3.policy_net.parameters(), self.td3.policy_net_target.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.td3.value_net1.parameters(), self.td3.value_net_target1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.td3.value_net2.parameters(), self.td3.value_net_target2.parameters()):
            target_param.data.copy_(param.data)


        for p in self.td3.policy_net_target.parameters():
            p.requires_grad = False
        for p in self.td3.value_net_target1.parameters():
            p.requires_grad = False
        for p in self.td3.value_net_target2.parameters():
            p.requires_grad = False

        self.replay_memory = ReplayMemory(self.replay_buffer_size)

        print('Policy Parameters: {}'.format(
            sum(p.numel() for p in self.td3.policy_net.parameters() if p.requires_grad)))
        # print('Q1 Parameters: {}'.format(sum(p.numel() for p in self.ddpg.soft_q_net1.parameters() if p.requires_grad)))
        # print('Q2 Parameters: {}'.format(sum(p.numel() for p in self.ddpg.soft_q_net2.parameters() if p.requires_grad)))
        print(
            'Value Parameters: {}'.format(sum(p.numel() for p in self.td3.value_net1.parameters() if p.requires_grad)))

        self.save_log([['policy_loss', 'value_loss', 'pi_grad', 'val_grad']], '/model_log')
        self.model_logs = torch.zeros(4, device=self.device)
        self.save_log([['ri', 'alive_steps', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi',
                        'sev_hyper', 'rollout', 'trial']], '/evaluation_log')
        self.save_log([['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')
        self.save_log([[1, 0, 0, 0, 0]], '/experiment_summary')
        self.completed_interactions = 0

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def update(self):
        if len(self.replay_memory) < self.sample_size * 10:
            return

        print('Running network update...')
        cl, pl, ql1, ql2, count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, val_grad, q2_grad, coeff_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        for i in range(self.train_pi_iters):
            # sample from buffer
            transitions = self.replay_memory.sample(self.sample_size)
            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            cur_feat_batch = torch.cat(batch.feat)
            actions_batch = torch.cat(batch.action).unsqueeze(1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            next_feat_batch = torch.cat(batch.next_feat)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            #actions_pi, log_prob = self.ddpg.evaluate_policy(cur_state_batch, cur_feat_batch)
            # self.entropy_coef = torch.exp(self.log_ent_coef.detach()) if self.sac_v2 else 0.001

            # value network update

            new_action, next_log_prob = self.td3.evaluate_target_policy_noise(next_state_batch, next_feat_batch)
            next_values = torch.min(self.td3.value_net_target1(next_state_batch, next_feat_batch, new_action),
                                    self.td3.value_net_target2(next_state_batch, next_feat_batch, new_action))
            target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            predicted_value1 = self.td3.value_net1(cur_state_batch, cur_feat_batch, actions_batch)
            predicted_value2 = self.td3.value_net2(cur_state_batch, cur_feat_batch, actions_batch)

            value_loss1 = self.value_criterion1(target_value.detach(), predicted_value1)
            value_loss2 = self.value_criterion1(target_value.detach(), predicted_value2)

            self.value_optimizer1.zero_grad()
            self.value_optimizer2.zero_grad()

            value_loss1.backward()
            value_loss2.backward()

            self.value_optimizer1.step()
            self.value_optimizer2.step()
            # self.ddpg.policy_net.ActionModule.policy_noise.reset()
            # self.ddpg.policy_net_target.ActionModule.policy_noise.reset()
            cl += value_loss1.detach()

            for param in self.td3.value_net1.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            for param in self.td3.value_net2.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            # if not self.sac_v2:
            #     self.value_optimizer.zero_grad()
            #     with torch.no_grad():
            #         min_qf_val = torch.min(self.ddpg.soft_q_net1(cur_state_batch, cur_feat_batch, actions_pi),
            #                                self.ddpg.soft_q_net2(cur_state_batch, cur_feat_batch, actions_pi))
            #     predicted_value = self.ddpg.value_net(cur_state_batch, cur_feat_batch)
            #     value_func_estimate = min_qf_val - (self.entropy_coef * log_prob)  # todo the temperature paramter
            #     value_loss = 0.5 * self.value_criterion(predicted_value, value_func_estimate.detach())
            #     value_loss.backward()
            #     coeff_grad += torch.nn.utils.clip_grad_norm_(self.ddpg.value_net.parameters(), self.grad_clip)
            #     self.value_optimizer.step()
            #     cl += value_loss.detach()
            #
            # # q network update
            # self.soft_q_optimizer1.zero_grad()
            # self.soft_q_optimizer2.zero_grad()
            # with torch.no_grad():  # calculate the target q vals here.
            #     if self.sac_v2:
            #         new_action, next_log_prob = self.ddpg.evaluate_policy(next_state_batch, next_feat_batch)
            #         next_q_values = torch.min(self.ddpg.target_q_net1(next_state_batch, next_feat_batch, new_action),
            #                                   self.ddpg.target_q_net2(next_state_batch, next_feat_batch, new_action))
            #         next_q_values = next_q_values - self.entropy_coef * next_log_prob
            #         target_q_values = (reward_batch + (self.gamma * (1 - done_batch) * next_q_values))
            #     else:
            #         target_value = self.ddpg.value_net_target(next_state_batch, next_feat_batch)
            #         target_q_values = (reward_batch + self.gamma * (1 - done_batch) * target_value)
            #
            # predicted_q_value1 = self.ddpg.soft_q_net1(cur_state_batch, cur_feat_batch, actions_batch)
            # predicted_q_value2 = self.ddpg.soft_q_net2(cur_state_batch, cur_feat_batch, actions_batch)
            # q_value_loss1 = 0.5 * self.soft_q_criterion1(predicted_q_value1, target_q_values)
            # q_value_loss2 = 0.5 * self.soft_q_criterion2(predicted_q_value2, target_q_values)
            # q_value_loss1.backward()
            # q1_grad += torch.nn.utils.clip_grad_norm_(self.ddpg.soft_q_net1.parameters(), self.grad_clip)
            # self.soft_q_optimizer1.step()
            # q_value_loss2.backward()
            # q2_grad += torch.nn.utils.clip_grad_norm_(self.ddpg.soft_q_net2.parameters(), self.grad_clip)
            # self.soft_q_optimizer2.step()

            # actor update : next q values
            # freeze value networks save compute: ref: openai:




            for p in self.td3.value_net1.parameters():
                p.requires_grad = False
            for p in self.td3.value_net2.parameters():
                p.requires_grad = False

            for p in self.td3.value_net_target1.parameters():
                p.requires_grad = False
            for p in self.td3.value_net_target2.parameters():
                p.requires_grad = False

            # min_qf_pi = torch.min(self.ddpg.soft_q_net1(cur_state_batch, cur_feat_batch, actions_pi),
            #                       self.ddpg.soft_q_net2(cur_state_batch, cur_feat_batch, actions_pi))

            policy_action, _ = self.td3.evaluate_policy_no_noise(cur_state_batch, cur_feat_batch)

            policy_loss = torch.min(self.td3.value_net1(cur_state_batch, cur_feat_batch, policy_action),
                                    self.td3.value_net2(cur_state_batch, cur_feat_batch, policy_action))
            policy_loss = (-1 * policy_loss).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            pl += policy_loss.detach()
            pi_grad += torch.nn.utils.clip_grad_norm_(self.td3.policy_net.parameters(), 10)

            # save compute: ref: openai:
            for p in self.td3.value_net1.parameters():
                p.requires_grad = True
            for p in self.td3.value_net2.parameters():
                p.requires_grad = True

            for p in self.td3.value_net_target1.parameters():
                p.requires_grad = True
            for p in self.td3.value_net_target2.parameters():
                p.requires_grad = True

            # entropy coeff update
            # if self.sac_v2:
            #     self.ent_coef_optimizer.zero_grad()
            #     _, log_prob = self.ddpg.evaluate_policy(cur_state_batch, cur_feat_batch)
            #     ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            #     ent_coef_loss.backward()
            #     coeff_grad += torch.nn.utils.clip_grad_norm_([self.log_ent_coef], self.grad_clip)
            #     self.ent_coef_optimizer.step()
            #     cl += ent_coef_loss.detach()

            self.n_updates += 1

            #Actor Update
            if self.n_updates % 2 * self.target_update_interval == 0:

                for p in self.td3.value_net1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net2.parameters():
                    p.requires_grad = False

                for p in self.td3.value_net_target1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net_target2.parameters():
                    p.requires_grad = False

                # min_qf_pi = torch.min(self.ddpg.soft_q_net1(cur_state_batch, cur_feat_batch, actions_pi),
                #                       self.ddpg.soft_q_net2(cur_state_batch, cur_feat_batch, actions_pi))

                policy_action, _ = self.td3.evaluate_policy_no_noise(cur_state_batch, cur_feat_batch)

                policy_loss = torch.min(self.td3.value_net1(cur_state_batch, cur_feat_batch, policy_action),
                                        self.td3.value_net2(cur_state_batch, cur_feat_batch, policy_action))
                policy_loss = (-1 * policy_loss).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                pl += policy_loss.detach()
                pi_grad += torch.nn.utils.clip_grad_norm_(self.td3.policy_net.parameters(), 10)

                # save compute: ref: openai:
                for p in self.td3.value_net1.parameters():
                    p.requires_grad = True
                for p in self.td3.value_net2.parameters():
                    p.requires_grad = True

                for p in self.td3.value_net_target1.parameters():
                    p.requires_grad = True
                for p in self.td3.value_net_target2.parameters():
                    p.requires_grad = True

                with torch.no_grad():
                    print("################updated target")
                    for param, target_param in zip(self.td3.policy_net.parameters(),
                                                   self.td3.policy_net_target.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)


            #Critic Update
                if self.n_updates % self.target_update_interval == 0:
                    with torch.no_grad():
                        print("################updated target")
                        for param, target_param in zip(self.td3.value_net1.parameters(),
                                                       self.td3.value_net_target1.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)
                        for param, target_param in zip(self.td3.value_net2.parameters(),
                                                       self.td3.value_net_target2.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)

                    # if self.sac_v2:
                    #     for param, target_param in zip(self.ddpg.soft_q_net1.parameters(),
                    #                                    self.ddpg.target_q_net1.parameters()):
                    #         target_param.data.mul_((1 - self.soft_tau))
                    #         target_param.data.add_(self.soft_tau * param.data)
                    #     for param, target_param in zip(self.ddpg.soft_q_net2.parameters(),
                    #                                    self.ddpg.target_q_net2.parameters()):
                    #         target_param.data.mul_((1 - self.soft_tau))
                    #         target_param.data.add_(self.soft_tau * param.data)
                    # else:
                    #     for param, target_param in zip(self.ddpg.value_net.parameters(),
                    #                                    self.ddpg.value_net_target.parameters()):
                    #         target_param.data.mul_((1 - self.soft_tau))
                    #         target_param.data.add_(self.soft_tau * param.data)

            # pl += policy_loss.detach()
            # ql1 += q_value_loss1.detach()
            # ql2 += q_value_loss2.detach()

        self.model_logs[0] = cl  # value loss or coeff loss
        self.model_logs[1] = pl
        self.model_logs[2] = pi_grad
        self.model_logs[3] = val_grad

        self.save_log([self.model_logs.detach().cpu().flatten().numpy()], '/model_log')
        print('success')

    def decay_lr(self):
        self.policy_lr = self.policy_lr / 10
        self.value_lr = self.value_lr / 10
        # self.entropy_lr = self.entropy_lr / 10
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = self.policy_lr
        for param_group in self.value_optimizer1.param_groups:
            param_group['lr'] = self.value_lr
        for param_group in self.value_optimizer2.param_groups:
            param_group['lr'] = self.value_lr
        # for param_group in self.soft_q_optimizer2.param_groups:
        #     param_group['lr'] = self.soft_q_lr

    def decay_noise_std(self):
        self.td3.policy_net.ActionModule.noise_std = self.td3.policy_net.ActionModule.noise_std / 10
        self.td3.policy_net_target.ActionModule.noise_std = self.td3.policy_net_target.ActionModule.noise_std / 10


    def run(self, args, patients, env_ids, seed):
        MAX_INTERACTIONS = 4000 if args.debug == 1 else 800000
        LR_DECAY_INTERACTIONS = 2000 if args.debug == 1 else 600000
        experiment_done, job_status, last_lr_update = False, 1, 0
        stop_criteria_len, stop_criteria_threshold = 10, 5
        ri_arr = np.ones(stop_criteria_len, dtype=np.float32) * 1000

        # setting up the testing arguments
        testing_args = deepcopy(args)
        testing_args.meal_amount = [40, 20, 80, 10, 60, 30]
        testing_args.meal_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        testing_args.time_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        testing_args.meal_prob = [1, -1, 1, -1, 1, -1]

        worker_agents = [Worker(args, 'training', patients, env_ids, i + 5, i, self.device) for i in
                         range(self.n_training_workers)]
        testing_agents = [Worker(testing_args, 'testing', patients, env_ids, i + 5000, i + 5000, self.device) for i in
                          range(self.n_testing_workers)]

        # sac learning
        for rollout in range(0, 30000):  # steps * n_workers * epochs
            print("rollout: ", rollout)
            t1 = time.time()
            for i in range(self.n_training_workers):
                data = worker_agents[i].rollout(self.td3, self.replay_memory)
                self.update()
            t2 = time.time()
            self.td3.save(rollout)
            self.td3.policy_net.ActionModule.policy_noise.reset()

            if rollout % 20 == 0:
                self.td3.policy_net.ActionModule.policy_noise.sigma = self.td3.policy_net.ActionModule.policy_noise.sigma/3

            # testing
            t5 = time.time()
            ri = 0
            if self.completed_interactions > 200000:
                self.td3.is_testing_worker = True
            for i in range(self.n_testing_workers):
                res = testing_agents[i].rollout(self.td3, self.replay_memory)
                ri += res[0]
            ri_arr[rollout % stop_criteria_len] = ri / self.n_testing_workers  # mean ri of that rollout.
            t6 = time.time()
            self.td3.is_testing_worker = False
            gc.collect()

            # decay lr
            self.completed_interactions += (self.n_step * self.n_training_workers)
            if (self.completed_interactions - last_lr_update) > LR_DECAY_INTERACTIONS:
                self.decay_lr()
                # self.decay_noise_std()
                last_lr_update = self.completed_interactions

            if self.completed_interactions > MAX_INTERACTIONS:
                experiment_done = True
                job_status = 2

            # logging and termination
            if self.args.verbose:
                print('\nExperiment: {}, Rollout {}: Time for rollout: {}, update: {}, '
                      'testing: {}'.format(self.args.folder_id, rollout, (t2 - t1), 0, (t6 - t5)))
            self.save_log([[job_status, rollout, (t2 - t1), 0, (t6 - t5)]], '/experiment_summary')

            if experiment_done:
                print('################## starting the validation trials #######################')
                n_val_trials = 3 if args.debug == 1 else 500
                validation_agents = [Worker(testing_args, 'testing', patients, env_ids, i + 6000, i + 6000, self.device)
                                     for i in range(n_val_trials)]
                for i in range(n_val_trials):
                    res = validation_agents[i].rollout(self.td3, self.replay_memory)
                print('Algo RAN Successfully')
                exit()
