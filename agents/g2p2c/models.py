import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import core
from utils.reward_func import composite_reward


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.n_features = args.n_features
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.LSTM = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, bidirectional=self.bidirectional)  # (seq_len, batch, input_size)

    def forward(self, s, feat, mode):
        if mode == 'batch':
            output, (hid, cell) = self.LSTM(s)
            lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
            feat = feat.squeeze(1)
        else:
            s = s.unsqueeze(0)  # add batch dimension
            output, (hid, cell) = self.LSTM(s)  # hid = layers * dir, batch, hidden
            lstm_output = hid.squeeze(1)  # remove batch dimension
            lstm_output = torch.flatten(lstm_output)  # hid = layers * hidden_size

        if self.use_handcraft == 1:  # concat_features = torch.cat((lstm_output, feat), dim=1)
            if mode == 'batch':
                extract_states = torch.cat((lstm_output, feat), dim=1)  # ==>torch.size[256 + 5]
            else:
                extract_states = torch.cat((lstm_output, feat), dim=0)
        else:
            extract_states = lstm_output
        return extract_states, lstm_output


class GlucoseModel(nn.Module):
    def __init__(self, args, device):
        super(GlucoseModel, self).__init__()
        self.n_features = args.n_features
        self.device = device
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        # self.feature_extractor = self.n_hidden * self.n_layers * self.directions + \
        #                          (self.n_handcrafted_features * self.use_handcraft)
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor #* 2
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        # self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        # self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.cgm_mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.cgm_sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, extract_state, action, mode):
        concat_dim = 1 if (mode == 'batch') else 0
        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output = fc_output1
        # fc_output2 = F.relu(self.fc_layer2(fc_output1))
        # fc_output = F.relu(self.fc_layer3(fc_output2))
        cgm_mu = F.tanh(self.cgm_mu(fc_output))
        # deterministic
        # cgm_sigma = torch.zeros(1, device=self.device, dtype=torch.float32)
        # cgm = cgm_mu
        # probabilistic
        cgm_sigma = F.softplus(self.cgm_sigma(fc_output) + 1e-5)
        z = self.normal.sample()
        cgm = cgm_mu + cgm_sigma * z
        cgm = torch.clamp(cgm, -1, 1)
        return cgm_mu, cgm_sigma, cgm


class ActionModule(nn.Module):
    def __init__(self, args, device):
        super(ActionModule, self).__init__()
        self.device = device
        self.args = args
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions + \
                                 (self.n_handcrafted_features * self.use_handcraft)
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, action_type='N'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = F.tanh(self.mu(fc_output))
        sigma = F.sigmoid(self.sigma(fc_output) + 1e-5)
        z = self.normalDistribution(0, 1).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)
        try:
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])
        except ValueError:
            print('\nCurrent mu: {}, sigma: {}'.format(mu, sigma))
            print('shape: {}. {}'.format(mu.shape, sigma.shape))
            print(extract_states.shape)
            print(extract_states)
            log_prob = torch.ones(2, 1, device=self.device, dtype=torch.float32) * self.glucose_target
        # log_prob = dst.log_prob(action[0])
        # sigma = torch.ones(1, device=self.device, dtype=torch.float32) * 0.01
        # clamp the sigma
        # sigma = F.softplus(self.sigma(fc_output) + 1e-5)
        # sigma = torch.clamp(sigma, 1e-5, 0.33)
        return mu, sigma, action, log_prob


class ValueModule(nn.Module):
    def __init__(self, args, device):
        super(ValueModule, self).__init__()
        self.device = device
        self.output = args.n_action
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions + \
                                 (self.n_handcrafted_features * self.use_handcraft)
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.value = NormedLinear(self.last_hidden, self.output, scale=0.1)

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        value = (self.value(fc_output))
        return value


class ActorNetwork(nn.Module):
    def __init__(self, args, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.args = args
        self.FeatureExtractor = FeatureExtractor(args)
        self.GlucoseModel = GlucoseModel(args, self.device)
        self.ActionModule = ActionModule(args, self.device)
        self.distribution = torch.distributions.Normal
        self.planning_n_step = args.planning_n_step
        self.n_planning_simulations = args.n_planning_simulations
        self.glucose_target = core.linear_scaling(x=112.5, x_min=self.args.glucose_min, x_max=self.args.glucose_max)
        self.t_to_meal = core.linear_scaling(x=0, x_min=0, x_max=self.args.t_meal)

    def forward(self, s, feat, old_action, mode, is_training=False):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
        if mode == 'forward':
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, action.detach(), mode)
        else:
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, old_action.detach(), mode)
        return mu, sigma, action, log_prob, cgm_mu, cgm_sigma, cgm

    def update_state(self, s, cgm_pred, action, batch_size):
        if batch_size == 1:
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=0)
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(1, device=self.device)), dim=0)
            s_new = s_new.unsqueeze(0)
            s = torch.cat((s[1:self.args.feature_history, :], s_new), dim=0)
        else:
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(batch_size, 1, device=self.device)), dim=1)
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=1)
            s_new = s_new.unsqueeze(1)
            s = torch.cat((s[:, 1:self.args.feature_history, :], s_new), dim=1)
        return s

    def expert_search(self, s, feat, rew_norm_var, mode):
        pi, mu, sigma, s_e, f_e, r = self.expert_MCTS_rollout(s, feat, mode, rew_norm_var)
        return pi, mu, sigma, s_e, f_e, r

    def expert_MCTS_rollout(self, s, feat, mode, rew_norm_var=1):
        batch_size = s.shape[0]
        first_action, first_mu, first_sigma, cum_reward, mu, sigma = 0, 0, 0, 0, 0, 0
        for i in range(self.planning_n_step):
            extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode) #.detach()# todo: fix handcraft features
            extract_states, lstmOut = extract_states.detach(), lstmOut.detach()
            mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
            if i == 0:
                first_action = action
                first_mu = mu
                first_sigma = sigma
            _, _, cgm_pred = self.GlucoseModel.forward(lstmOut, action, mode)
            bg = core.inverse_linear_scaling(y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min, x_max=self.args.glucose_max)
            reward = np.array([[composite_reward(self.args, state=xi, reward=None)] for xi in bg])
            reward = reward / (math.sqrt(rew_norm_var + 1e-8))
            reward = np.clip(reward, 10, 10)
            discount = (self.args.gamma ** i)
            cum_reward += (reward * discount)

            # todo: fix - this is a hardcoded to pump action exponential!!!
            action = action.detach()
            cgm_pred = cgm_pred.detach()
            pump_action = self.args.action_scale * (torch.exp((action - 1) * 4))
            action = core.linear_scaling(x=pump_action, x_min=self.args.insulin_min, x_max=self.args.insulin_max)
            ### #todo

            s = self.update_state(s, cgm_pred, action, batch_size)
            feat[0] += 1  #todo: quick fix for integrating the 'n', wont work other handcraft feat
        cum_reward = torch.as_tensor(cum_reward, dtype=torch.float32, device=self.device)
        return first_action, first_mu, first_sigma, s, feat, cum_reward

    def horizon_error(self, s, feat, actions, real_glucose, mode):
        horizon_error = 0
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        for i in range(0, len(actions)):
            cur_action = torch.as_tensor(actions[i], dtype=torch.float32, device=self.device).reshape(1)
            extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode) #.detach()
            extract_states, lstmOut = extract_states.detach(), lstmOut.detach()

            cgm_mu, cgm_sigma, cgm_pred = self.GlucoseModel.forward(lstmOut, cur_action, mode)
            pred = core.inverse_linear_scaling(y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min,
                                               x_max=self.args.glucose_max)
            horizon_error += ((pred - real_glucose[i])**2)
            s = self.update_state(s, cgm_pred, cur_action, batch_size=1)
        return horizon_error / len(actions)


class CriticNetwork(nn.Module):
    def __init__(self, args, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)
        self.aux_mode = args.aux_mode
        self.GlucoseModel = GlucoseModel(args, device)
    def forward(self, s, feat, action, cgm_pred=True, mode='forward'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        value = self.ValueModule.forward(extract_states)
        cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(lstmOut, action.detach(), mode) if cgm_pred else (None, None, None)
        return value, cgm_mu, cgm_sigma, cgm


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args, device)
        self.Critic = CriticNetwork(args, device)
        if load:
            self.Actor = torch.load(actor_path, map_location=device)
            self.Critic = torch.load(critic_path, map_location=device)
        self.distribution = torch.distributions.Normal
        self.is_testing_worker = False

    def predict(self, s, feat):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm = self.Actor(s, feat, None, mode='forward',
                                                                               is_training=self.is_testing_worker)
        state_value, c_cgm_mu,  c_cgm_sigma, c_cgm = self.Critic(s, feat, action, cgm_pred=True, mode='forward' )
        return (mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm), (state_value, c_cgm_mu,  c_cgm_sigma, c_cgm)

    def get_action(self, s, feat):
        (mu, std, act, log_prob, a_cgm_mu, a_cgm_sig, a_cgm), (s_val, c_cgm_mu, c_cgm_sig, c_cgm) = self.predict(s, feat)
        data = dict(mu=mu, std=std, action=act, log_prob=log_prob, state_value=s_val, a_cgm_mu=a_cgm_mu,
                    a_cgm_sigma=a_cgm_sig, c_cgm_mu=c_cgm_mu, c_cgm_sigma=c_cgm_sig, a_cgm=a_cgm, c_cgm=c_cgm)
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s, feat):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        state_value, _,  _, _ = self.Critic(s, feat, action=None, cgm_pred=False, mode='forward' )
        return state_value.detach().cpu().numpy()

    def evaluate_actor(self, state, action, feat):  # evaluate batch
        action_mean, action_std, _, _, a_cgm_mu, a_cgm_sigma, _ = self.Actor(state, feat, action, mode='batch')
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy, a_cgm_mu, a_cgm_sigma

    def evaluate_critic(self, state, feat, action, cgm_pred):  # evaluate batch
        state_value, c_cgm_mu,  c_cgm_sigma, _ = self.Critic(state, feat, action, cgm_pred, mode='batch')
        return torch.squeeze(state_value), c_cgm_mu,  c_cgm_sigma

    def save(self, episode):
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)

            
def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
