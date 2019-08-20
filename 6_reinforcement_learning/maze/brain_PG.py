import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.out = nn.Linear(10, n_actions)

        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.out.weight, 0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.out(x)
        return F.softmax(action_values, dim=-1)


class PolicyGradient():
    def __init__(self, n_actions,
                 n_features,
                 learning_rate=0.01,
                 gamma=0.9
                 ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy = Net(n_features, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.losses = []

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values = self.policy(state)
        m = Categorical(action_values)
        action = m.sample()
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        ''' learn from one episode.'''
        ep_obs = torch.FloatTensor(self.ep_obs)
        ep_as = torch.LongTensor(self.ep_as)

        # calculate every V_t
        R = 0
        rewards = []
        for r in self.ep_rs[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        ep_rs = torch.FloatTensor(rewards)
        # normalize rewards
        ep_rs = (ep_rs - ep_rs.mean()) / (rewards.std() + eps)

        out = self.policy(ep_obs)
        loss = self.loss_func(out, ep_as) * ep_rs
        loss = loss.mean()
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
