# Variational Autoencoder (VAE) Image Generation Project

## Overview
This project implements a Variational Autoencoder (VAE) for image generation and face mixing using PyTorch. The implementation supports training on custom image datasets, image reconstruction, and face blending capabilities.

## Features
- Custom VAE neural network architecture
- 256x256 pixel image processing
- Variational autoencoding with KL divergence loss
- Face mixing functionality
- Mixed precision training
- Graceful training interruption
- Image reconstruction visualization

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- Pillow (PIL)

## Hyperparameters
- Batch Size: 8
- Image Size: 256x256
- Latent Dimension: 256
- Number of Epochs: 100
- Learning Rate: 0.0001

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm pillow
```

## Dataset Preparation
- Place your training images in a directory structure:
```
datasets/
└── class_name/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Modes of Operation
The script supports two primary modes:

### Training Mode
- Set `MODE = "train"`
- Trains the VAE on your dataset
- Saves model checkpoints and final model

### Face Mixing Mode
- Set `MODE = "mix"`
- Loads a pre-trained model
- Demonstrates face blending between two images

## Training
```bash
python VAE.py
```
- Model checkpoints saved every 10 epochs
- Final model saved as `vae_final.pth`

## Face Mixing
- Requires pre-trained model
- Automatically selects random images from dataset
- Visualizes original and mixed images

## Outputs
- Trained models:
  - Periodic checkpoints: `vae_checkpoint_epoch_X.pth`
  - Final model: `vae_final.pth`
  - Interrupt checkpoint: `vae_interrupt.pth`
- Visualization of reconstructed images
- Mixed face images

## Device Support
- Automatically detects and uses CUDA if available
- Falls back to CPU if no GPU detected
- Supports mixed precision training

## Customization
- Modify hyperparameters in the script
- Adjust network architecture as needed
- Change dataset directory path

## Main Functions
- `train_vae()`: Train the Variational Autoencoder
- `mix_faces()`: Blend faces using latent space interpolation
- `visualize_reconstruction()`: Show original vs reconstructed images

## Interruption Handling
- Can safely interrupt training (Ctrl+C)
- Saves checkpoint before exiting

## Limitations
- Requires GPU for efficient training
- Performance depends on dataset quality
- Face mixing works best with similar face datasets

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Ivan Murzin

## Acknowledgments
- Inspired by VAE and generative modeling research
- Uses PyTorch for deep learning implementation