import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

h_dim = 200
z_dim = 100
image_size = 784
num_epochs = 200
batch_size = 128
learning_rate = 1e-3
c_num = 10

sample_dir = 'cvae_samples'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../../data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

# (60000, 28, 28)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


class CVAE(nn.Module):
    def __init__(self, image_size, h_dim, z_dim, c_num):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(image_size + c_num, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim + c_num, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        self.c_num = c_num

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if len(p.size()) > 1:
                nn.init.xavier_normal_(p)

    def encode(self, x, c):
        x = torch.cat((x,c), dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, c):
        c_onehot = torch.zeros(x.size(0), self.c_num).scatter_(1, c.unsqueeze(-1),1).type_as(x)
        mu, log_var = self.encode(x, c_onehot)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z, c_onehot)
        return x_reconst, mu, log_var


model = CVAE(image_size, h_dim, z_dim, c_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
for epoch in range(num_epochs):
    total_loss = 0.
    total_reconst_loss = 0.
    total_kl_loss = 0.
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x, y)

        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reconst_loss += reconst_loss.item()
        total_kl_loss += kl_div.item()
        total_loss += loss.item()
    print("Epoch[{}/{}]], Total Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}"
          .format(epoch + 1, num_epochs, total_loss / len(mnist), total_reconst_loss / len(mnist),
                  total_kl_loss / len(mnist)))

with torch.no_grad():
        # Save the sampled images
        z = torch.randn(c_num, z_dim).to(device)
        c = torch.arange(0, c_num)
        c_onehot = torch.zeros(c_num, c_num).scatter_(1, c.unsqueeze(-1), 1).type_as(z)

        out = model.decode(z, c_onehot).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

        # Save the reconstructed images
        save_image(x_reconst.view(-1, 1, 28, 28), os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))
