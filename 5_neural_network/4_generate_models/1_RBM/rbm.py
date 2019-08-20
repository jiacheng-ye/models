import torch


class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 device="cpu"):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.device = device

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        self.weights = self.weights.to(device)
        self.visible_bias = self.visible_bias.to(device)
        self.hidden_bias = self.hidden_bias.to(device)

        self.weights_momentum = self.weights_momentum.to(device)
        self.visible_bias_momentum = self.visible_bias_momentum.to(device)
        self.hidden_bias_momentum = self.hidden_bias_momentum.to(device)

    def infer(self, v):
        return self._sigmoid(torch.matmul(v, self.weights) + self.hidden_bias)

    def generate(self, h):
        return self._sigmoid(torch.matmul(h, self.weights.t()) + self.visible_bias)

    def sample_h(self, v):
        return (self.infer(v) >= torch.rand(self.num_hidden).to(self.device)).float()

    def sample_v(self, h):
        return (self.generate(h) >= torch.rand(self.num_visible).to(self.device)).float()

    def contrastive_divergence(self, input_data):
        # Positive phase
        pos_h_prob = self.infer(input_data)

        # Negative phase
        v_pre = input_data
        for step in range(self.k):
            h_pre = self.sample_h(v_pre)
            v_pre = self.sample_v(h_pre)

        neg_h_prob = self.infer(v_pre)

        grad_w = torch.matmul(input_data.t(), pos_h_prob) - torch.matmul(v_pre.t(), neg_h_prob)
        grad_visiable_bias = (input_data - v_pre).sum(0)
        grad_hidden_bias = (pos_h_prob - neg_h_prob).sum(0)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += grad_w

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += grad_visiable_bias

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += grad_hidden_bias

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - v_pre) ** 2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
