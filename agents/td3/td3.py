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

# python run_RL_agent.py --agent td3 --folder_id TEMPORAL_PERBuffer/TD3/NormDist_Sigma_2e-1/TD30_1 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_model normal_dist --noise_std 2e-1  --mu_penalty 1 --replay_buffer_type per_rank --replay_buffer_temporal_decay 0.99 --seed 1 --debug 0

#python run_RL_agent.py --agent td3 --folder_id SJ53Experiments/NoCutOff/NoiseApplication/TD3/Model2/NormDist/sigma_2e-1/TD30_1 --patient_id 0 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_application 2 --noise_model normal_dist --mu_penalty 1 --noise_std 2e-1 --action_penalty_limit 1.0 --action_penalty_coef 1e-2 --seed 1 --debug 0
#python run_RL_agent.py --agent td3 --folder_id LargeNoiseStudy/TD3/NormDist/sigma_5e0/TD32_1 --patient_id 2 --return_type average --action_type exponential --device cuda --pi_lr 1e-3 --vf_lr 1e-3 --soft_tau 0.005 --noise_model normal_dist --mu_penalty 0 --noise_std 5e0 --action_penalty_coef 0 --seed 1 --debug 0

Transition = namedtuple('Transition',
                        ('state', 'feat', 'action', 'reward', 'next_state', 'next_feat', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritisedExperienceReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, temporal_decay = 1):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.temporal_decay = temporal_decay  # Decay factor for temporal weighting
        self.timestamps = np.zeros((capacity,), dtype=np.float32)  # Track when each sample was added
        self.current_time = 0  # Incremental time counter to simulate timestamps

    def push(self, *args):
        '''Save a transition with a maximum priority initially'''
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority if max_priority > 0 else 1.0
        self.timestamps[self.position] = self.current_time  # Set the timestamp
        self.position = (self.position + 1) % self.capacity
        self.current_time += 1  # Increment the time counter

    def sample(self, batch_size, beta=0.4, buffer_type="per_proportional"):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
            timestamps = self.timestamps
        else:
            priorities = self.priorities[:self.position]
            timestamps = self.timestamps[:self.position]

        # Rank the priorities
        ranked_indices = np.argsort(priorities)
        ranks = np.empty_like(ranked_indices)
        ranks[ranked_indices] = np.arange(len(priorities))

        if buffer_type == "per_proportional":
            probabilities = priorities ** self.alpha  # Temporal difference based probability
        elif buffer_type == "per_rank":
            probabilities = (1 / (ranks + 1)) ** self.alpha  # Rank based probability

        # Apply temporal decay factor to the probabilities
        # decay_weights = np.exp(-self.temporal_decay * (self.current_time - timestamps))
        decay_weights = self.temporal_decay ** (self.current_time - timestamps)
        adjusted_probabilities = probabilities * decay_weights

        # Normalize the probabilities
        sum_probabilities = adjusted_probabilities.sum()
        if sum_probabilities == 0:
            probabilities = np.ones_like(adjusted_probabilities) / len(adjusted_probabilities)
        else:
            probabilities = adjusted_probabilities / sum_probabilities

        if np.isnan(probabilities).any():
            print('Nan probabilities found')
            print(probabilities)
            probabilities = np.ones_like(probabilities) / len(probabilities)

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + 1e-8

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

        self.target_update_interval = 2  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.train_pi_iters = args.n_pi_epochs
        self.shuffle_rollout = args.shuffle_rollout
        # self.soft_q_lr = args.vf_lr
        self.value_lr = args.vf_lr
        self.policy_lr = args.pi_lr
        self.grad_clip = args.grad_clip

        self.mu_penalty = args.mu_penalty
        self.action_penalty_limit = args.action_penalty_limit
        self.action_penalty_coef = args.action_penalty_coef

        self.replay_buffer_type = args.replay_buffer_type
        self.replay_buffer_alpha = args.replay_buffer_alpha
        self.replay_buffer_beta = args.replay_buffer_beta
        self.replay_buffer_temporal_decay = args.replay_buffer_temporal_decay


        self.weight_decay = 0

        ### TD3 networks:
        self.td3 = ActorCritic(args, load, path1, path2, device).to(self.device)
        self.value_criterion1 = nn.MSELoss()
        self.value_criterion2 = nn.MSELoss()
        self.value_optimizer1 = torch.optim.Adam(self.td3.value_net1.parameters(), lr=self.value_lr,
                                                 weight_decay=self.weight_decay)
        self.value_optimizer2 = torch.optim.Adam(self.td3.value_net2.parameters(), lr=self.value_lr,
                                                 weight_decay=self.weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.td3.policy_net.parameters(), lr=self.policy_lr,
                                                 weight_decay=self.weight_decay)
        for target_param, param in zip(self.td3.policy_net.parameters(), self.td3.policy_net_target.parameters()):
            target_param.data.copy_(param.data)

        for p in self.td3.policy_net_target.parameters():
            p.requires_grad = False

        for target_param, param in zip(self.td3.value_net1.parameters(), self.td3.value_net_target1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.td3.value_net2.parameters(), self.td3.value_net_target2.parameters()):
            target_param.data.copy_(param.data)

        for p in self.td3.value_net_target1.parameters():
            p.requires_grad = False
        for p in self.td3.value_net_target2.parameters():
            p.requires_grad = False

        if self.replay_buffer_type == "random":
            self.replay_memory = ReplayMemory(self.replay_buffer_size)
        elif self.replay_buffer_type == "per_proportional" or self.replay_buffer_type == "per_rank":
            self.replay_memory = PrioritisedExperienceReplayMemory(self.replay_buffer_size,
                                                                   alpha=self.replay_buffer_alpha, temporal_decay=self.replay_buffer_temporal_decay)
        else:
            print("Incorrect replay buffer type")

        print('Policy Parameters: {}'.format(
            sum(p.numel() for p in self.td3.policy_net.parameters() if p.requires_grad)))
        print(
            'Value network 1 Parameters: {}'.format(
                sum(p.numel() for p in self.td3.value_net1.parameters() if p.requires_grad)))
        print(
            'Value network 2 Parameters: {}'.format(
                sum(p.numel() for p in self.td3.value_net2.parameters() if p.requires_grad)))

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
            if self.replay_buffer_type == "random":
                transitions = self.replay_memory.sample(self.sample_size)
            elif self.replay_buffer_type == "per_proportional" or self.replay_buffer_type == "per_rank":
                transitions, indices, weights = self.replay_memory.sample(self.sample_size,
                                                                          beta=self.replay_buffer_beta,
                                                                          buffer_type=self.replay_buffer_type)
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            cur_feat_batch = torch.cat(batch.feat)
            actions_batch = torch.cat(batch.action).unsqueeze(1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            next_feat_batch = torch.cat(batch.next_feat)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            # value network update
            new_action, next_log_prob = self.td3.evaluate_target_policy_noise(next_state_batch, next_feat_batch)
            next_values = torch.min(self.td3.value_net_target1(next_state_batch, next_feat_batch, new_action),
                                    self.td3.value_net_target2(next_state_batch, next_feat_batch, new_action))
            target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            predicted_value1 = self.td3.value_net1(cur_state_batch, cur_feat_batch, actions_batch)
            predicted_value2 = self.td3.value_net2(cur_state_batch, cur_feat_batch, actions_batch)

            value_loss1 = self.value_criterion1(target_value.detach(), predicted_value1)
            value_loss2 = self.value_criterion2(target_value.detach(), predicted_value2)
            # td_error = predicted_value - target_value
            # value_loss = (td_error.pow(2) * weights).mean()
            # self.replay_memory.update_priorities(indices, np.abs(td_error.cpu().detach().numpy()))

            self.value_optimizer1.zero_grad()
            self.value_optimizer2.zero_grad()

            value_loss1.backward()
            value_loss2.backward()

            self.value_optimizer1.step()
            self.value_optimizer2.step()

            cl += value_loss1.detach()

            for param in self.td3.value_net1.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            for param in self.td3.value_net2.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            self.n_updates += 1

            # actor update
            if self.n_updates % self.target_update_interval == 0:
                # freeze value networks save compute: ref: openai:
                for p in self.td3.value_net1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net2.parameters():
                    p.requires_grad = False

                for p in self.td3.value_net_target1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net_target2.parameters():
                    p.requires_grad = False

                policy_action, _ = self.td3.evaluate_policy_no_noise(cur_state_batch, cur_feat_batch)
                policy_loss = torch.min(self.td3.value_net1(cur_state_batch, cur_feat_batch, policy_action),
                                        self.td3.value_net2(cur_state_batch, cur_feat_batch, policy_action))

                if self.replay_buffer_type == "random":
                    policy_loss = (-1 * policy_loss).mean()
                elif self.replay_buffer_type == "per_proportional" or self.replay_buffer_type == "per_rank":
                    policy_loss = (-1 * policy_loss * weights).mean()

                #TODO remove the mu_penalty as no longer needed as can be turned off and on by the action_penalty_coef
                policy_loss += self.mu_penalty * action_penalty(policy_action, lower_bound=-self.action_penalty_limit,
                                                                upper_bound=self.action_penalty_limit,
                                                                penalty_factor=self.action_penalty_coef)

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

                # Update target networks
                with torch.no_grad():
                    print("################updated target networks")
                    for param, target_param in zip(self.td3.value_net1.parameters(),
                                                   self.td3.value_net_target1.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)
                    for param, target_param in zip(self.td3.value_net2.parameters(),
                                                   self.td3.value_net_target2.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

                    for param, target_param in zip(self.td3.policy_net.parameters(),
                                                   self.td3.policy_net_target.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

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

        # DDPG learning
        for rollout in range(0, 30000):  # steps * n_workers * epochs
            print("rollout: ", rollout)
            t1 = time.time()
            for i in range(self.n_training_workers):
                data = worker_agents[i].rollout(self.td3, self.replay_memory)
                self.update()
            t2 = time.time()
            self.td3.save(rollout)
            self.td3.policy_net.ActionModule.policy_noise.reset()

            # if rollout >= 80 and rollout % 40 == 0:
            # if rollout == 80:
            #     self.td3.policy_net.ActionModule.policy_noise.sigma = self.td3.policy_net.ActionModule.policy_noise.sigma / 2
            #     self.td3.policy_net.ActionModule.policy_noise.reset()
            #     print("Reduced noise --> updated noise sigma is {}".format(self.td3.policy_net.ActionModule.policy_noise.sigma))

            # if rollout == 40:
            #     self.ddpg.policy_net.ActionModule.policy_noise.sigma = 2e-1
            #     self.ddpg.policy_net.ActionModule.policy_noise.reset()
            #
            # if rollout == 150:
            #     self.ddpg.policy_net.ActionModule.policy_noise.sigma = 1e-1
            #     self.ddpg.policy_net.ActionModule.policy_noise.reset()
            #
            # if rollout == 185:
            #     self.ddpg.policy_net.ActionModule.policy_noise.sigma = 5e-2
            #     self.ddpg.policy_net.ActionModule.policy_noise.reset()

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


def action_penalty(action, lower_bound=-1.0, upper_bound=1.0, penalty_factor=0.1):
    mu = torch.atanh(action)
    high_penalty = torch.clamp(mu - upper_bound, min=0.0)
    low_penalty = torch.clamp(lower_bound - mu, min=0.0)
    total_penalty = high_penalty + low_penalty
    # return penalty_factor * torch.mean(total_penalty ** 2)
    return penalty_factor * torch.mean(mu ** 2)
    # return penalty_factor * torch.mean(torch.abs(mu))

