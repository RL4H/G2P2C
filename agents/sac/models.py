import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20


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


class QvalModel(nn.Module):
    def __init__(self, args, device):
        super(QvalModel, self).__init__()
        self.n_features = args.n_features
        self.device = device
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions + \
                                 (self.n_handcrafted_features * self.use_handcraft)
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        self.q = nn.Linear(self.last_hidden, self.output)

    def forward(self, extract_state, action, mode):
        concat_dim = 1 if (mode == 'batch') else 0
        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        qval = self.q(fc_output)
        return qval


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
        self.mu = nn.Linear(self.last_hidden, self.output)
        self.sigma = nn.Linear(self.last_hidden, self.output)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, worker_mode='training'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = self.mu(fc_output)
        sigma = self.sigma(fc_output)  # * 0.66, + 1e-5
        log_std = torch.clamp(sigma, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(log_std)

        dst = self.normalDistribution(mu, action_std)
        if worker_mode == 'training':
            gaussian_action = dst.rsample()
        else:
            gaussian_action = mu

        action = torch.tanh(gaussian_action)

        # calc log_prob
        # openai implementation
        logp_pi = dst.log_prob(gaussian_action[0])  #.sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - gaussian_action[0] - F.softplus(-2 * gaussian_action[0])))  #.sum(axis=1)
        # SAC paper implementation
        # log_prob = dst.log_prob(gaussian_action[0]) - torch.log(1 - action[0] ** 2 + 1e-6)

        return mu, action_std, action, logp_pi


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
        self.value = nn.Linear(self.last_hidden, self.output)

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        value = (self.value(fc_output))
        return value


class PolicyNetwork(nn.Module):
    def __init__(self, args, device):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.args = args
        self.FeatureExtractor = FeatureExtractor(args)
        self.ActionModule = ActionModule(args, self.device)
        self.distribution = torch.distributions.Normal

    def forward(self, s, feat, mode='forward', worker_mode='training'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states, worker_mode='training')
        return mu, sigma, action, log_prob


class ValueNetwork(nn.Module):
    def __init__(self, args, device):
        super(ValueNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)

    def forward(self, s, feat, mode='batch'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        value = self.ValueModule.forward(extract_states)
        return value


class QNetwork(nn.Module):
    def __init__(self, args, device):
        super(QNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.QvalModel = QvalModel(args, device)

    def forward(self, s, feat, action, mode='batch'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode)
        qvalue = self.QvalModel.forward(extract_states, action, mode)
        return qvalue


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.experiment_dir = args.experiment_dir
        self.is_testing_worker = False
        self.sac_v2 = args.sac_v2
        self.policy_net = PolicyNetwork(args, device)
        self.soft_q_net1 = QNetwork(args, device)
        self.soft_q_net2 = QNetwork(args, device)

        if self.sac_v2:
            self.target_q_net1 = QNetwork(args, device)
            self.target_q_net2 = QNetwork(args, device)
        else:
            self.value_net = ValueNetwork(args, device)
            self.value_net_target = ValueNetwork(args, device)

    def get_action(self, s, feat, mode='forward', worker_mode='training'):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mu, sigma, action, log_prob = self.policy_net.forward(s, feat, mode='forward', worker_mode='training')
        return action.detach().cpu().numpy(), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def evaluate_policy(self, state, feat):  # evaluate policy <batch>
        mu, sigma, action, log_prob = self.policy_net.forward(state, feat, mode='batch')
        return action, log_prob

    def save(self, episode):  # save checkpoints
        if self.sac_v2:
            policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
            soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
            soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
            torch.save(self.policy_net, policy_net_path)
            torch.save(self.soft_q_net1, soft_q_net1_path)
            torch.save(self.soft_q_net2, soft_q_net2_path)
        else:
            policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
            soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
            soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
            value_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_value_net.pth'
            torch.save(self.policy_net, policy_net_path)
            torch.save(self.soft_q_net1, soft_q_net1_path)
            torch.save(self.soft_q_net2, soft_q_net2_path)
            torch.save(self.value_net, value_net_path)


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
