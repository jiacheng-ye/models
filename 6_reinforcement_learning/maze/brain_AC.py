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
        self.action_head = nn.Linear(10, n_actions)
        self.value_head = nn.Linear(10, 1)

        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.action_head.weight, 0, 0.1)
        nn.init.normal_(self.value_head.weight, 0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_scores = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_value


class AC():
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
        self.losses = []

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_values, state_value = self.policy(state)
        m = Categorical(action_values)
        action = m.sample()
        return action

    def learn(self, s, a, r, s_):
        ''' learn from one step'''
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        a = torch.LongTensor(a)

        prob_as, v = self.policy(s)
        _, v_ = self.policy(s_)

        td_error = r + self.gamma * v_ - v

        actor_loss = (self.loss_func(prob_as.unsqueeze(0), a) * td_error.detach()).mean()
        critic_loss = (td_error.pow(2)).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
