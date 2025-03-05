import os
import random
import sys
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# =============================================================================
# Hyperparameters and Configuration
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODE = "mix"  # "train" or "mix"
# =============================================================================
# Mount Google Drive (if your dataset is stored on Drive)
# =============================================================================


# Hyperparameters
batch_size = 8  # Reduced to 4 for 256x256 images to fit 12 GB VRAM
image_size = 256  # Changed to 256x256
latent_dim = 256  # Latent space size
num_epochs = 100  # Number of epochs for convergence
learning_rate = 0.0001  # Learning rate for Adam
data_dir = "datasets/"  # Adjust to your dataset path

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Normalizes to [0, 1]
])

dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)


# =============================================================================
# VAE Model Definition
# =============================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=256, img_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Adjusted for 256x256 input
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 4, 2, 1, bias=False),  # 256x256 -> 128x128
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # 128x128 -> 64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8 -> 4x4 (added layer)
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)  # Mean (adjusted for 512 channels)
        self.fc_log_var = nn.Linear(512 * 4 * 4, latent_dim)  # Log variance

        # Decoder: Adjusted for 256x256 output
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)  # Adjusted for 512 channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # 64x64 -> 128x128
            nn.ReLU(True),
            nn.ConvTranspose2d(16, img_channels, 4, 2, 1, bias=False),  # 128x128 -> 256x256
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 512 * 4 * 4)  # Adjusted for 512 channels
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 4, 4)  # Adjusted for 512 channels
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# =============================================================================
# Loss Function
# =============================================================================
def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


# =============================================================================
# Training Function
# =============================================================================
def train_vae():
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision

    # Checkpoint handling
    def save_checkpoint_and_exit(signum, frame):
        print(f"\nReceived signal {signum}. Saving checkpoint and exiting...")
        torch.save(vae.state_dict(), "vae_interrupt.pth")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_checkpoint_and_exit)
    signal.signal(signal.SIGTERM, save_checkpoint_and_exit)

    print("Starting VAE training...")
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images = images.to(device)
            optimizer.zero_grad()

            with autocast():
                recon_images, mu, log_var = vae(images)
                loss = vae_loss(recon_images, images, mu, log_var)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), f"vae_checkpoint_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved at epoch {epoch + 1}.")

    torch.save(vae.state_dict(), "vae_final.pth")
    print("Training completed. Model saved as 'vae_final.pth'.")
    return vae


# =============================================================================
# Face Mixing Function
# =============================================================================
def mix_faces(image_path1, image_path2, model_path="vae_final.pth"):
    # Load and preprocess images
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    img1_tensor = preprocess(img1).unsqueeze(0).to(device)
    img2_tensor = preprocess(img2).unsqueeze(0).to(device)

    # Load trained VAE
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # Encode both images
    with torch.no_grad():
        mu1, log_var1 = vae.encode(img1_tensor)
        mu2, log_var2 = vae.encode(img2_tensor)

    # Mix latent representations (average the means)
    mu_mixed = (mu1 + mu2) / 2

    # Decode the mixed latent vector
    with torch.no_grad():
        mixed_image = vae.decode(mu_mixed)

    # Display original and mixed images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img1_tensor.squeeze().cpu().permute(1, 2, 0).numpy())
    axs[0].set_title("Face 1")
    axs[0].axis("off")
    axs[1].imshow(img2_tensor.squeeze().cpu().permute(1, 2, 0).numpy())
    axs[1].set_title("Face 2")
    axs[1].axis("off")
    axs[2].imshow(mixed_image.squeeze().cpu().permute(1, 2, 0).numpy())
    axs[2].set_title("Mixed Face")
    axs[2].axis("off")
    plt.show()

    return mixed_image


# =============================================================================
# Utility: Visualize Reconstruction
# =============================================================================
def visualize_reconstruction(vae, dataloader):
    vae.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        recon_images, _, _ = vae(images)

        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(min(4, images.size(0))):
            axs[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axs[0, i].axis("off")
            axs[1, i].imshow(recon_images[i].cpu().permute(1, 2, 0).numpy())
            axs[1, i].axis("off")
        axs[0, 0].set_title("Original")
        axs[1, 0].set_title("Reconstructed")
        plt.show()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":

    if MODE == "train":

        # Train the VAE
        vae = train_vae()

    elif MODE == "mix":

        # Load trained VAE
        vae = VAE(latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load("vae_final.pth", map_location=device))

        # Example usage of face mixing
        image_path1, image_path2 = [f"datasets/celeba_hq_256/{random.randint(0, 30000):05}.jpg" for _ in range(2)]
        mixed_image = mix_faces(image_path1, image_path2)

