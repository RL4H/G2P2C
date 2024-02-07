import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.n_features = args.n_features
        self.use_handcraft = args.use_handcraft
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.LSTM = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                            batch_first=True, bidirectional=self.bidirectional)  # (seq_len, batch, input_size)

    def forward(self, s, mode):
        if mode == 'batch':
            output, (hid, cell) = self.LSTM(s)
            lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
        else:
            s = s.unsqueeze(0)  # add batch dimension
            output, (hid, cell) = self.LSTM(s)  # hid = layers * dir, batch, hidden
            lstm_output = hid.squeeze(1)  # remove batch dimension
            lstm_output = torch.flatten(lstm_output)  # hid = layers * hidden_size
        extract_states = lstm_output
        return extract_states, lstm_output


class QvalModel(nn.Module):
    def __init__(self, args, device):
        super(QvalModel, self).__init__()
        self.n_features = args.n_features
        self.device = device
        self.use_handcraft = args.use_handcraft
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.q = NormedLinear(self.last_hidden, self.output, scale=0.1)

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
        self.use_handcraft = args.use_handcraft

        self.discrete_actions = args.discrete_actions #args.n_discrete_actions if self.discrete_actions else a
        self.output = args.n_action

        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        if self.discrete_actions:
            self.mu = nn.Linear(self.last_hidden, self.output)
            self.categoricalDistribution = torch.distributions.Categorical
        else:
            self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
            self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
            self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, softmax_dim=0):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        fc_output_end = self.mu(fc_output)

        if self.discrete_actions:
            # print('\nnew')
            probs = F.softmax(fc_output_end, dim=softmax_dim)
            # print(fc_output_end)
            # print(probs)

            dst = self.categoricalDistribution(probs)
            discrete_action = dst.sample()  # [0, k-1]

            prob_action = probs[discrete_action]
            log_prob = dst.log_prob(discrete_action)

            action = -1.0 + discrete_action * (2/(self.output-1))  # convert to corresponding continuous space,

            best_action = torch.argmax(probs)
            mu = -1.0 + best_action * (2/(self.output-1))  # this is for deterministic policy in testing
            sigma = probs  # not needed todo: refactor

            log_prob = log_prob.unsqueeze(0)
            mu = mu.unsqueeze(0)
            sigma = sigma
            action = action.unsqueeze(0)

            # print(best_action)

        else:
            mu = F.tanh(fc_output_end)
            sigma = F.sigmoid(self.sigma(fc_output) + 1e-5)
            z = self.normalDistribution(0, 1).sample()
            action = mu + sigma * z
            action = torch.clamp(action, -1, 1)
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])

        return mu, sigma, action, log_prob


class ValueModule(nn.Module):
    def __init__(self, args, device):
        super(ValueModule, self).__init__()
        self.device = device
        self.output = args.n_action
        self.use_handcraft = args.use_handcraft
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
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
        self.distribution = torch.distributions.Normal

    def forward(self, s, old_action, mode, softmax_dim=0):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, mode)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states, softmax_dim=softmax_dim)
        return mu, sigma, action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, args, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)

    def forward(self, s, mode='forward'):
        extract_states, lstmOut = self.FeatureExtractor.forward(s, mode)
        value = self.ValueModule.forward(extract_states)
        return value


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device

        self.discrete_actions = args.discrete_actions  #args.n_discrete_actions if self.discrete_actions else
        self.output = args.n_action

        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args, device)
        self.Critic = CriticNetwork(args, device)
        if load:
            self.Actor = torch.load(actor_path, map_location=device)
            self.Critic = torch.load(critic_path, map_location=device)
        self.distribution = torch.distributions.Normal
        self.is_testing_worker = False

    def predict(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        mean, std, action, log_prob = self.Actor(s, None, mode='forward', softmax_dim=0)
        state_value = self.Critic(s, mode='forward' )
        return (mean, std, action, log_prob), state_value

    def get_action(self, s):
        (mu, std, act, log_prob), (s_val) = self.predict(s)
        data = dict(mu=mu, std=std, action=act, log_prob=log_prob, state_value=s_val)
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        state_value = self.Critic(s, mode='forward')
        return state_value.detach().cpu().numpy()

    def evaluate_actor(self, state, action):  # evaluate batch
        action_mean, action_std, _, _, = self.Actor(state, action, mode='batch', softmax_dim=1)

        # discrete
        if self.discrete_actions:
            probs = action_std  # batch x K
            dist = torch.distributions.Categorical(probs)
            dist_entropy = dist.entropy() #.sum(0, keepdim=True)
            discrete_action = ((action + 1) * (self.output-1)) / 2
            discrete_action = discrete_action.long()  # convert to torch.int64, batch x 1
            #prob_discrete = probs.gather(1, discrete_action)  # batch x 1
            action_logprobs = dist.log_prob(discrete_action.squeeze(1))
            action_logprobs = action_logprobs.unsqueeze(1)
            dist_entropy = dist_entropy.unsqueeze(1)
            # print(action_logprobs.shape)
            # print(dist_entropy.shape)
            # exit()
        else:
            # continuous
            dist = self.distribution(action_mean, action_std)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy

    def evaluate_critic(self, state):  # evaluate batch
        state_value = self.Critic(state, mode='batch')
        return torch.squeeze(state_value)

    def save(self, episode):
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)

            
def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
