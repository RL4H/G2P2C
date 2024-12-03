import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import core
from agents.ddpg.core import composite_reward
import math

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
        # self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)

        self.sigma = nn.Linear(self.last_hidden, self.output)
        self.normalDistribution = torch.distributions.Normal
        self.noise_model = args.noise_model
        self.noise_std = args.noise_std
        self.policy_noise = ExploratoryNoise(0, self.noise_std, noise_model=self.noise_model)
        self.noise_application = args.noise_application
        self.target_action_std = args.target_action_std
        self.target_action_lim = args.target_action_lim




    def forward(self, extract_states, worker_mode='training'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = self.mu(fc_output)
        sigma = self.sigma(fc_output)  # * 0.66, + 1e-5
        log_std = torch.clamp(sigma, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(log_std)
        # dst = self.normalDistribution(mu, action_std)

        if worker_mode == 'training':
            noise_value = self.policy_noise.get_noise()

            # Noise Model 0 - Same as paper
            if self.noise_application == 0:
                action = torch.tanh(mu + (torch.randn_like(mu) * 0.1))# Value from Fujimoto et al (2018) paper

            # Noise Model 1 - Additive to mu
            elif self.noise_application == 1:
                action = torch.tanh(mu + noise_value)

            # Noise Model 2 - Multiplicative to mu
            elif self.noise_application == 2:
                action = torch.tanh((1 + noise_value) * mu)

            # Noise Model 3 - Additive to tanh(mu) with clamp [-1,1]
            elif self.noise_application == 3:
                action = torch.tanh(mu) + noise_value
                action = torch.clamp(action, min=-1, max=1)

            # Noise Model 4 - Multiplicative to tanh(mu) with clamp [-1,1]
            elif self.noise_application == 4:
                action = torch.tanh(mu) * (1 + noise_value)
                action = torch.clamp(action, min=-1, max=1)

            # Noise Model 5 - Additive to scaled tanh(mu) then passed through tanh
            elif self.noise_application == 5:
                mu_scale = 2
                action = mu_scale * torch.tanh(mu) + noise_value
                action = torch.tanh(action)

            # Noise Model 6 - Multiplicative to scaled tanh(mu) then passed through tanh
            elif self.noise_application == 6:
                mu_scale = 2
                action = mu_scale * torch.tanh(mu) * (1 + noise_value)
                action = torch.tanh(action)

            else:
                print("Invalid noise application code")

        elif worker_mode == 'target':
            # gaussian_action = mu + self.normalDistribution(0, self.noise_std).rsample()  # dst.rsample()

            # action = torch.tanh(mu) + self.policy_noise.get_noise()
            # action = torch.clamp(action, min=-1, max=1)

            # action = torch.tanh((torch.clamp(1+self.policy_noise.get_noise(), min=-1.05, max=1.05))*mu)
            # action = torch.tanh((torch.clamp(torch.from_numpy(np.array([1 + self.policy_noise.get_noise()])).float().to(mu.device), min=-1.05, max=1.05)) * mu)
            # action_no_noise = torch.tanh(mu)
            # action_with_noise = torch.tanh(mu) + self.policy_noise.get_noise()
            # action = torch.clamp(action_with_noise,min=action_no_noise*0.99, max=action_no_noise*1.01)

            policy_noise = self.target_action_std #0.2 # Value from Fujimoto et al (2018) paper
            noise_clip = self.target_action_lim #0.5 # Value from Fujimoto et al (2018) paper

            noise = (torch.randn_like(mu) * policy_noise).clamp(-noise_clip, noise_clip)
            action = torch.tanh(mu + noise)


        else:
            action = torch.tanh(mu)

        # action = torch.tanh(gaussian_action)

        # calc log_prob
        # openai implementation
        logp_pi = 0#dst.log_prob(gaussian_action[0])  # .sum(axis=-1)
        # logp_pi -= (2 * (np.log(2) - gaussian_action[0] - F.softplus(-2 * gaussian_action[0])))  # .sum(axis=1)
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
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states, worker_mode=worker_mode)
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
        # self.sac_v2 = args.sac_v2
        self.policy_net = PolicyNetwork(args, device)

        self.value_net1 = QNetwork(args, device)
        self.value_net2 = QNetwork(args, device)

        if load:
            self.policy_net = torch.load(actor_path, map_location=device)
            self.value_net1 = torch.load(critic_path, map_location=device)
            self.value_net2 = torch.load(critic_path, map_location=device)

        # Copy for target networks
        self.policy_net_target = deepcopy(self.policy_net)  # PolicyNetwork(args, device)
        self.value_net_target1 = deepcopy(self.value_net1)#QNetwork(args, device)
        self.value_net_target2 = deepcopy(self.value_net2)  # QNetwork(args, device)

    def get_action(self, s, feat, mode='forward', worker_mode='training'):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mu, sigma, action, log_prob = self.policy_net.forward(s, feat, mode=mode, worker_mode=worker_mode)
        return action.detach().cpu().numpy(), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def get_action_no_noise(self, s, feat, mode='forward', worker_mode='training'):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mu, sigma, action, log_prob = self.policy_net.forward(s, feat, mode=mode, worker_mode='no noise')
        return action.detach().cpu().numpy(), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def evaluate_policy(self, state, feat):  # evaluate batch
        mu, sigma, action, log_prob = self.policy_net.forward(state, feat, mode='batch')
        return action, log_prob

    def evaluate_policy_no_noise(self, state, feat):  # evaluate batch
        mu, sigma, action, log_prob = self.policy_net.forward(state, feat, mode='batch', worker_mode='no noise')
        return action, log_prob

    def evaluate_target_policy_noise(self, state, feat):  # evaluate batch
        mu, sigma, action, log_prob = self.policy_net_target.forward(state, feat, mode='batch', worker_mode='target')
        return action, log_prob

    def save(self, episode):
        # if self.sac_v2:
        #     policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
        #     soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
        #     soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
        #     torch.save(self.policy_net, policy_net_path)
        #     torch.save(self.soft_q_net1, soft_q_net1_path)
        #     torch.save(self.soft_q_net2, soft_q_net2_path)
        # else:
        #     policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
        #     soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
        #     soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
        #     value_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_value_net.pth'
        #     torch.save(self.policy_net, policy_net_path)
        #     torch.save(self.soft_q_net1, soft_q_net1_path)
        #     torch.save(self.soft_q_net2, soft_q_net2_path)
        #     torch.save(self.value_net, value_net_path)
        policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
        policy_net_target_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net_target.pth'
        # soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
        # soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
        value_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_value_net.pth'
        value_net_target_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_value_net_target.pth'

        torch.save(self.policy_net, policy_net_path)
        torch.save(self.policy_net_target, policy_net_target_path)
        # torch.save(self.soft_q_net1, soft_q_net1_path)
        # torch.save(self.soft_q_net2, soft_q_net2_path)
        torch.save(self.value_net1, value_net_path)#TODO: update to include network 2
        torch.save(self.value_net_target1, value_net_target_path)


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out


# class PolicyNoise():
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma
#
#     def __call__(self):
#         return torch.distributions.Normal(self.mu, self.sigma)


class ExploratoryNoise:
    # OU noise applied same as default paramaters as Lillicrap et al. (2016)
    def __init__(self, mu, sigma, noise_model='normal_dist', theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.noise_model = noise_model
        self.reset()
        self.noise = torch.distributions.Normal(0, self.sigma).rsample()

    def get_noise(self):
        if self.noise_model == 'normal_dist':
            return torch.distributions.Normal(0, self.sigma).rsample()

        elif self.noise_model == 'ou_noise':
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
                + self.sigma * np.sqrt(self.dt) * np.random.normal()
            self.x_prev = x
            return x

        else:
            raise ValueError(self.noise_model + " not valid noise type for policy exploration")

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.noise = torch.distributions.Normal(0, self.sigma).rsample()


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out