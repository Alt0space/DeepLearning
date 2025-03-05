# Image Generation GAN Project

## Overview
This project implements a Generative Adversarial Network (GAN) for generating images using PyTorch. The implementation supports training on custom image datasets and includes functionality for model checkpointing, interruption handling, and image generation.

## Features
- Custom Generator and Discriminator neural network architectures
- Support for 256x256 pixel image generation
- Configurable hyperparameters
- One-sided label smoothing
- Graceful training interruption with checkpoint saving
- Loss visualization
- Test mode for generating sample images

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

## Hyperparameters
- Batch Size: 16
- Image Size: 256x256
- Latent Dimension: 100
- Number of Epochs: 25
- Learning Rate: 0.0002
- Beta1: 0.5

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm
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

## Training
- Set `TEST_MODE = False`
- Set `LOAD_MODEL = False` for training from scratch
- Run the script:
```bash
python GAN.py
```

## Testing/Inference
- Set `TEST_MODE = True`
- Run the script to generate sample images

## Output
- Trained models are saved as:
  - `generator.pth`
  - `discriminator.pth`
- Checkpoint models saved per epoch
- Loss plot generated after training
- Sample generated images displayed

## Interruption Handling
- Training can be safely interrupted (Ctrl+C)
- Checkpoints saved as `generator_interrupt.pth` and `discriminator_interrupt.pth`

## Device Support
- Automatically detects and uses CUDA if available
- Falls back to CPU if no GPU detected

## Customization
- Modify hyperparameters in the script
- Adjust network architectures as needed
- Change dataset directory path

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Ivan Murzin	
```