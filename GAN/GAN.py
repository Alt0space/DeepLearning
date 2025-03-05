#%% [code]
import os
import sys
import signal
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =============================================================================
# Mount Google Drive (if your dataset is stored on Drive)
# =============================================================================
#from google.colab import drive
#drive.mount('/content/drive')

# =============================================================================
# Flags and Hyperparameters
# =============================================================================
TEST_MODE = False     # True => Load generator and generate an image (no training)
LOAD_MODEL = False    # True => Load pre-trained weights from "generator.pth"/"discriminator.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
batch_size = 16         # For 256x256 images
image_size = 256        # Image dimensions
latent_dim = 100        # Size of the latent vector
num_epochs = 25         # Number of training epochs
learning_rate = 0.0002  # Learning rate for optimizers
beta1 = 0.5             # Beta1 hyperparameter for Adam

# Dataset directory
data_dir = "./datasets/"

print(torch.cuda.is_available())

# =============================================================================
# Model Definitions
# =============================================================================


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_map_size=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 16),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_map_size * 16, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(feature_map_size, feature_map_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size // 2),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(feature_map_size // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(img_channels, feature_map_size // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 64x64
            nn.Conv2d(feature_map_size // 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_map_size * 8, feature_map_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(feature_map_size * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability in [0, 1]
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# =============================================================================
# Custom weights initialization
# =============================================================================

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# =============================================================================
# Utility: Image Visualization
# =============================================================================

def imshow(img, title="Image"):
    """Displays a single image or a grid of images."""
    img = img / 2 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.show()

# =============================================================================
# Testing / Inference Function
# =============================================================================

def test_inference(loaded_netG=None):
    """
    If loaded_netG is provided, use it directly.
    Otherwise, load from 'generator.pth' on disk.
    Then generate and display a single sample image.
    """
    if loaded_netG is None:
        if not os.path.exists("generator.pth"):
            print("No trained generator found! Exiting test mode.")
            return
        netG = Generator(latent_dim=latent_dim).to(device)
        netG.load_state_dict(torch.load("generator.pth", map_location=device))
        print("Loaded generator from 'generator.pth' for testing.")
    else:
        netG = loaded_netG

    netG.eval()
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_img = netG(noise).cpu()
    grid = torchvision.utils.make_grid(generated_img, normalize=True)
    imshow(grid, title="Generated Image (Test Mode)")

# =============================================================================
# Training Function
# =============================================================================

def train():
    # -------------------------------
    # Data Loading
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1] normalization
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )

    # -------------------------------
    # Initialize Models
    # -------------------------------
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # -------------------------------
    # Optionally Load Pre-trained Weights
    # -------------------------------
    if LOAD_MODEL:
        if os.path.exists("generator.pth"):
            netG.load_state_dict(torch.load("generator.pth", map_location=device))
            print("Loaded pre-trained generator weights from 'generator.pth'.")
        else:
            print("No pre-trained generator weights found. Training from scratch.")

        if os.path.exists("discriminator.pth"):
            netD.load_state_dict(torch.load("discriminator.pth", map_location=device))
            print("Loaded pre-trained discriminator weights from 'discriminator.pth'.")
        else:
            print("No pre-trained discriminator weights found. Training from scratch.")

    # -------------------------------
    # Loss and Optimizers
    # -------------------------------
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Lists for tracking losses
    losses_G = []
    losses_D = []

    # -------------------------------
    # Graceful Interruption Handling
    # -------------------------------
    def save_checkpoint_and_exit(signum, frame):
        print(f"\nReceived signal {signum}. Saving interrupt checkpoint and exiting...")
        torch.save(netG.state_dict(), "generator_interrupt.pth")
        torch.save(netD.state_dict(), "discriminator_interrupt.pth")
        print("Saved 'generator_interrupt.pth' & 'discriminator_interrupt.pth'. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_checkpoint_and_exit)
    signal.signal(signal.SIGTERM, save_checkpoint_and_exit)

    # -------------------------------
    # Training Loop
    # -------------------------------
    print("Starting GAN training...")
    try:
        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"), 0):

                # -----------------
                # 1. Train Discriminator
                # -----------------
                netD.zero_grad()
                real_images = real_images.to(device)
                b_size = real_images.size(0)

                # Labels
                real_label = torch.full((b_size,), 0.9, device=device)  # one-sided label smoothing
                fake_label = torch.full((b_size,), 0.0, device=device)

                # Forward real batch through D
                output_real = netD(real_images)
                lossD_real = criterion(output_real, real_label)
                lossD_real.backward()

                # Generate fake images
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_images = netG(noise)

                # Forward fake batch through D
                output_fake = netD(fake_images.detach())
                lossD_fake = criterion(output_fake, fake_label)
                lossD_fake.backward()
                optimizerD.step()

                lossD_total = lossD_real.item() + lossD_fake.item()
                losses_D.append(lossD_total)

                # -----------------
                # 2. Train Generator
                # -----------------
                netG.zero_grad()
                # We want the generator to fool D into believing fake images are real
                output_fake_for_G = netD(fake_images)
                lossG = criterion(output_fake_for_G, real_label)  # pretend they're real
                lossG.backward()
                optimizerG.step()
                losses_G.append(lossG.item())

                if i % 100 == 0:
                    tqdm.write(
                        f"[Epoch {epoch+1}/{num_epochs}][Batch {i}/{len(dataloader)}] "
                        f"Loss_D: {lossD_total:.4f} Loss_G: {lossG.item():.4f}"
                    )

            # Save checkpoint at the end of each epoch
            torch.save(netG.state_dict(), f"generator_checkpoint_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"discriminator_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}.")

        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by KeyboardInterrupt. Saving interrupt checkpoint...")
        torch.save(netG.state_dict(), "generator_interrupt.pth")
        torch.save(netD.state_dict(), "discriminator_interrupt.pth")
        print("Saved 'generator_interrupt.pth' & 'discriminator_interrupt.pth'. Exiting.")
        return

    # -------------------------------
    # Save Final Models
    # -------------------------------
    torch.save(netG.state_dict(), "generator.pth")
    torch.save(netD.state_dict(), "discriminator.pth")
    print("Final models saved as 'generator.pth' and 'discriminator.pth'.")

    # -------------------------------
    # Plot Training Losses
    # -------------------------------
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_G, label="Generator Loss")
    plt.plot(losses_D, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # -------------------------------
    # Generate a Sample After Training
    # -------------------------------
    test_inference(netG)

# =============================================================================
# Main execution
# =============================================================================


if __name__ == '__main__':
    if TEST_MODE:
        for i in range(5):
            test_inference()
            time.sleep(5)
    else:
        train()