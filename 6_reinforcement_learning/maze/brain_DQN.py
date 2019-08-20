import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        return action_values


class DQN():
    def __init__(self, n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, n_features * 2 + 2))

        self.eval_net, self.target_net = Net(n_features, n_actions), Net(n_features, n_actions)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.losses = []

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= self.epsilon:
            action_values = self.eval_net(state)
            action = torch.max(action_values, 1)[1].cpu().numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_a = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, self.n_features + 1:self.n_features + 2])
        batch_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(batch_s).gather(1, batch_a)  # [batch, 1], estimate Q
        q_next = self.target_net(batch_s_).detach()
        q_target = batch_r + self.gamma * q_next.max(1)[0]  # `real` Q
        loss = self.loss_func(q_eval, q_target)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
