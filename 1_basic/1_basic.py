# -*- coding:utf8 -*-
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)


def simple_clip_grad(grad):
    return grad / 2 if grad >= 1 else grad


d = a * b
h = d.register_hook(simple_clip_grad)
e = b * c
f = d + e
f.backward()
h.remove()
# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #
x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print('dl/dw: ', linear.weight.grad)
print('dl/db: ', linear.bias.grad)

optimizer.step()
# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

pred = linear(x)
loss = criterion(pred, y)
print('w after 1 step optimization: ', linear.weight)
print('b after 1 step optimization: ', linear.bias)
print('loss after 1 step optimization: ', loss.item())

# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
print(y)  # 变成tensor类型
z = y.numpy()
print(z)  # 变成numpy array类型

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

train_dataset = torchvision.datasets.CIFAR10(root=r'../data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
# get one
image, label = train_dataset[0]
print(image.size())  # [3, 32, 32]
print(label)
image = image.numpy()

image = image[:, ..., np.newaxis]
print(image.shape)
image = np.concatenate([image[0], image[1], image[2]], -1)
print(image.shape)
plt.imshow(image)
plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

data_iter = iter(train_loader)

# get a batch
images, labels = data_iter.next()

for images, labels in train_loader:
    pass


# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1. Initialize file paths or a list of file names.
        pass

    def __getitem__(self, item):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0


# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

resnet = torchvision.models.resnet18(pretrained=True)
# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer(fully connection) for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())  # (64, 100)

# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
