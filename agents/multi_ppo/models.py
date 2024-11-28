import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import core


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
        # note: extract_states and lstm_output is only different when handcraft features are used.
        return extract_states, lstm_output


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
        self.ActionModule = ActionModule(args, self.device)

    def forward(self, s, feat, old_action, mode, is_training=False):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
        return mu, sigma, action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, args, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)

    def forward(self, s, feat, mode='forward'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        value = self.ValueModule.forward(extract_states)
        return value


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

    def predict(self, s, feat):  # forward func for networks
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mean, std, action, log_prob = self.Actor(s, feat, None, mode='forward', is_training=self.is_testing_worker)
        state_value = self.Critic(s, feat, mode='forward' )
        return mean, std, action, log_prob, state_value

    def get_action(self, s, feat):  # pass values to worker for simulation on cpu.
        mu, std, act, log_prob, s_val = self.predict(s, feat)
        data = dict(mu=mu, std=std, action=act, log_prob=log_prob, state_value=s_val)
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s, feat):  # terminating V(s) of traj
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        state_value = self.Critic(s, feat, mode='forward' )
        return state_value.detach().cpu().numpy()

    def evaluate_actor(self, state, action, feat):  # evaluate actor <batch>
        action_mean, action_std, _, _ = self.Actor(state, feat, action, mode='batch')
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def evaluate_critic(self, state, feat):  # evaluate critic <batch>
        state_value = self.Critic(state, feat, mode='batch')
        return torch.squeeze(state_value)

    def save(self, episode):  # save checkpoints for networks.
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)

            
def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
