import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm, trange
import os
from rbm import RBM

########## CONFIGURATION ##########
BATCH_SIZE = 100
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = 200

DATA_FOLDER = '../../../data/'
OUTPUT_DIR = 'output'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

########## LOADING DATASET ##########
print('Loading dataset...')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),  # 3 for RGB channels
                         std=(0.5, 0.5, 0.5))])

train_dataset = datasets.MNIST(root=DATA_FOLDER, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = datasets.MNIST(root=DATA_FOLDER, train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, device=device)

for epoch in trange(EPOCHS, desc="Epoch"):
    epoch_error = 0.0

    for batch, _ in tqdm(train_loader, desc="Iteration"):
        batch = batch.view(BATCH_SIZE, -1).to(device)  # flatten input data

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error.item()

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

    # Save real images
    if (epoch + 1) == 1:
        images = batch.reshape(batch.size(0), 1, 28, 28)
        save_image(images, os.path.join(OUTPUT_DIR, 'real_images.png'))

    # Save sampled images
    h = torch.randn(BATCH_SIZE, HIDDEN_UNITS).to(device)
    fake_images = rbm.sample_v(h)
    fake_images = fake_images.reshape(BATCH_SIZE, 1, 28, 28)
    save_image(fake_images, os.path.join(OUTPUT_DIR, 'fake_images-{}.png'.format(epoch + 1)))
