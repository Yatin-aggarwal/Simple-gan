import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    def __init__(self , z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(256, 784)
        self.fc4 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(784, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.tanh(self.fc4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc3(x))
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 128
image_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 100


generator = Generator(z_dim).to(device)
discriminator = Discriminator(z_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = torchvision.datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.BCELoss()
optimizer_gen = optim.Adam(generator.parameters(), lr=lr)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()

        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        lossG.backward()
        optimizer_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                step += 1




