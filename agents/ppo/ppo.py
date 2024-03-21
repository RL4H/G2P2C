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
from utils.reward_normalizer import RewardNormalizer
from agents.ppo.worker import Worker
from agents.ppo.models import ActorCritic


class PPO:
    def __init__(self, args, device, load, path1, path2):
        self.args = args
        self.n_step = args.n_step
        self.feature_history = args.feature_history
        self.n_handcrafted_features = args.n_handcrafted_features
        self.n_features = args.n_features
        self.grad_clip = args.grad_clip
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.target_kl = args.target_kl
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.batch_size = args.batch_size
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.device = device

        self.policy = ActorCritic(args, load, path1, path2, device).to(self.device)
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)

        self.value_criterion = nn.MSELoss()
        self.shuffle_rollout = args.shuffle_rollout
        self.normalize_reward = args.normalize_reward
        self.reward_normaliser = RewardNormalizer(num_envs=self.n_training_workers, cliprew=10.0,
                                                  gamma=self.gamma, epsilon=1e-8, per_env=False)
        self.return_type = args.return_type
        self.rollout_buffer = {}
        self.old_states = torch.rand(self.n_training_workers, self.n_step, self.feature_history, self.n_features,
                                     device=self.device)
        self.feat = torch.rand(self.n_training_workers, self.n_step, 1, self.n_handcrafted_features, device=self.device)
        self.old_actions = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.old_logprobs = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.reward = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_targ = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.adv = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_pred = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)
        self.first_flag = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)

        self.save_log([['policy_grad', 'value_grad', 'val_loss', 'exp_var', 'true_var', 'pi_loss', 'avg_rew']], '/model_log')
        self.save_log([['ri', 'alive_steps', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi',
                        'sev_hyper', 'rollout', 'trial']], '/evaluation_log')
        self.model_logs = torch.zeros(7, device=self.device)
        self.save_log([['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')
        self.save_log([[1, 0, 0, 0, 0]], '/experiment_summary')
        self.completed_interactions = 0
        self.avg_rew = 0

        if self.args.verbose:
            print('Policy Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('Value Network Parameters: {}'.format(
                sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def compute_gae(self):
        orig_device = self.v_pred.device
        assert orig_device == self.reward.device == self.first_flag.device
        vpred, reward, first = (x.cpu() for x in (self.v_pred, self.reward, self.first_flag))
        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = reward.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)
        adv = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = reward[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            adv[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        vtarg = vpred[:, :-1] + adv
        return adv.to(device=orig_device), vtarg.to(device=orig_device)

    def prepare_rollout_buffer(self):
        '''concat data from different workers'''
        s_hist = self.old_states.view(-1, self.feature_history, self.n_features)
        s_handcraft = self.feat.view(-1, 1, self.n_handcrafted_features)
        act = self.old_actions.view(-1, 1)
        logp = self.old_logprobs.view(-1, 1)
        v_targ = self.v_targ.view(-1)
        adv = self.adv.view(-1)
        first_flag = self.first_flag.view(-1)
        buffer_len = s_hist.shape[0]

        if self.shuffle_rollout:
            rand_perm = torch.randperm(buffer_len)
            s_hist = s_hist[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
            s_handcraft = s_handcraft[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
            act = act[rand_perm, :]  # torch.Size([batch, 1])
            logp = logp[rand_perm, :]  # torch.Size([batch, 1])
            v_targ = v_targ[rand_perm]  # torch.Size([batch])
            adv = adv[rand_perm]  # torch.Size([batch])

        self.rollout_buffer = dict(s_hist=s_hist, s_handcraft=s_handcraft, act=act, logp=logp, ret=v_targ,
                                   adv=adv, len=buffer_len)

    def train_pi(self):
        print('Running pi update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training, buffer_len = True, self.rollout_buffer['len']
        for i in range(self.train_pi_iters):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = self.rollout_buffer['s_hist'][start_idx:end_idx, :, :]
                feat_batch = self.rollout_buffer['s_handcraft'][start_idx:end_idx, :, :]
                old_actions_batch = self.rollout_buffer['act'][start_idx:end_idx, :]
                old_logprobs_batch = self.rollout_buffer['logp'][start_idx:end_idx, :]
                advantages_batch = self.rollout_buffer['adv'][start_idx:end_idx]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
                self.optimizer_Actor.zero_grad()
                logprobs, dist_entropy = self.policy.evaluate_actor(old_states_batch, old_actions_batch, feat_batch)
                ratios = torch.exp(logprobs - old_logprobs_batch)
                ratios = ratios.squeeze()
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()
                # print('\nPPO debug ratio: {}, adv_mean {}, adv_sigma {}'.format(ratios.mean().detach().cpu().numpy(),
                #       advantages_batch.mean().detach().cpu().numpy(), advantages_batch.std().detach().cpu().numpy()))

                # early stop: approx kl calculation
                log_ratio = logprobs - old_logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
                if approx_kl > 1.5 * self.target_kl:
                    if self.args.verbose:
                        print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, n_batch, approx_kl))
                    continue_pi_training = False
                    break
                if torch.isnan(policy_loss):  # for debugging only!
                    print('policy loss: {}'.format(policy_loss))
                    exit()
                temp_loss_log += policy_loss.detach()
                policy_loss.backward()
                policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                pol_count += 1
                self.optimizer_Actor.step()
                start_idx += self.batch_size
            if not continue_pi_training:
                break
        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running vf update...')
        explained_var = torch.zeros(1, device=self.device)
        val_loss_log = torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)
        value_grad = torch.zeros(1, device=self.device)
        true_var = torch.zeros(1, device=self.device)
        buffer_len = self.rollout_buffer['len']
        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < buffer_len:
                end_idx = min(start_idx + self.batch_size, buffer_len)
                old_states_batch = self.rollout_buffer['s_hist'][start_idx:end_idx, :, :]
                feat_batch = self.rollout_buffer['s_handcraft'][start_idx:end_idx, :, :]
                returns_batch = self.rollout_buffer['ret'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                state_values = self.policy.evaluate_critic(old_states_batch, feat_batch)
                value_loss = self.value_criterion(state_values, returns_batch)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = state_values.detach().flatten()
                y_true = returns_batch.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)
        #print('\nvalue update: explained varience is {} true variance is {}'.format(explained_var / val_count, true_var / val_count))
        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def update(self, rollout):
        if self.return_type == 'discount':
            if self.normalize_reward:  # reward normalisation
                self.reward = self.reward_normaliser(self.reward, self.first_flag)
            self.adv, self.v_targ = self.compute_gae()  # # calc returns

        if self.return_type == 'average':
            self.reward = self.reward_normaliser(self.reward, self.first_flag, type='average')
            self.adv, self.v_targ = self.compute_gae()

        self.prepare_rollout_buffer()
        self.model_logs[6] = self.avg_rew
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4]  = self.train_vf()

        self.save_log([self.model_logs.detach().cpu().flatten().numpy()], '/model_log')

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr

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

        worker_agents = [Worker(args, 'training', patients, env_ids, i+5, i, self.device) for i in range(self.n_training_workers)]
        testing_agents = [Worker(testing_args, 'testing', patients, env_ids, i+5000, i+5000, self.device) for i in range(self.n_testing_workers)]

        # ppo learning
        for rollout in range(0, 30000):  # steps * n_workers * epochs
            t1 = time.time()
            for i in range(self.n_training_workers):
                data = worker_agents[i].rollout(self.policy)
                self.old_states[i] = data['obs']
                self.feat[i] = data['feat']
                self.old_actions[i] = data['act']
                self.old_logprobs[i] = data['logp']
                self.v_pred[i] = data['v_pred']
                self.reward[i] = data['reward']
                self.first_flag[i] = data['first_flag']

            t2 = time.time()
            t3 = time.time()
            self.update(rollout)
            self.policy.save(rollout)
            t4 = time.time()

            t5 = time.time()
            ri = 0
            # testing
            if self.completed_interactions > 200000:
                self.policy.is_testing_worker = True
            for i in range(self.n_testing_workers):
                res = testing_agents[i].rollout(self.policy)
                ri += res[0]
            ri_arr[rollout % stop_criteria_len] = ri / self.n_testing_workers  # mean ri of that rollout.
            t6 = time.time()
            self.policy.is_testing_worker = False
            gc.collect()

            # decay lr
            self.completed_interactions += (self.n_step * self.n_training_workers)
            if (self.completed_interactions - last_lr_update) > LR_DECAY_INTERACTIONS:
                self.decay_lr()
                last_lr_update = self.completed_interactions

            if self.completed_interactions > MAX_INTERACTIONS:
                experiment_done = True
                job_status = 2

            # logging and termination
            if self.args.verbose:
                print('\nExperiment: {}, Rollout {}: Time for rollout: {}, update: {}, '
                      'testing: {}'.format(self.args.folder_id, rollout, (t2 - t1), (t4 - t3), (t6 - t5)))
            self.save_log([[job_status, rollout, (t2 - t1), (t4 - t3), (t6 - t5)]], '/experiment_summary')

            if experiment_done:
                print('################## starting the validation trials #######################')
                n_val_trials = 3 if args.debug == 1 else 500
                validation_agents = [Worker(testing_args, 'testing', patients, env_ids, i + 6000, i + 6000, self.device) for i in range(n_val_trials)]
                for i in range(n_val_trials):
                    res = validation_agents[i].rollout(self.policy)
                print('Algo RAN Successfully')
                exit()
