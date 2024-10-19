# MNIST Denoising Autoencoder

A deep learning project that implements a Convolutional Autoencoder to remove noise from MNIST handwritten digit images using TensorFlow and Keras.

## Project Overview

This project demonstrates the implementation of a denoising autoencoder that:
1. Adds controlled Gaussian noise to MNIST images
2. Trains a convolutional autoencoder to reconstruct the original clean images
3. Visualizes the results showing original, noisy, and denoised images

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required packages using:
```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

```
mnist-denoising-autoencoder/
├── utils.py          # Utility functions for noise addition and model architecture
├── main.py          # Main script for training and visualization
└── README.md        # Project documentation
```

## Features

- Gaussian noise generation with configurable noise factor
- Convolutional autoencoder architecture
- Real-time visualization of:
  - Original MNIST images
  - Noisy images
  - Denoised (reconstructed) images
- Training progress monitoring with accuracy and loss metrics

## Implementation Details

### Model Architecture

The autoencoder consists of:

#### Encoder
- Input Layer (28x28x1)
- Conv2D (32 filters) + ReLU
- MaxPooling2D
- Conv2D (64 filters) + ReLU
- MaxPooling2D

#### Decoder
- Conv2D (64 filters) + ReLU
- UpSampling2D
- Conv2D (32 filters) + ReLU
- UpSampling2D
- Conv2D (1 filter) + Sigmoid

### Training Configuration
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 128
- Epochs: 10
- Data Split: Training and Test sets from MNIST

## Usage

1. Clone the repository:
```bash
git clone Ayoub-Elkhaiari/Denoising_Autoencoder.git
cd Denoising_Autoencoder
```

2. Run the main script:
```bash
python main.py
```

The script will:
- Load and preprocess the MNIST dataset
- Add noise to the images
- Train the autoencoder
- Display visualization of results

## Customization

You can modify the following parameters in the code:

In `utils.py`:
- `noise_factor`: Control the amount of noise (default: 0.5)
- Model architecture parameters in `build_autoencoder()`

In `main.py`:
- Training parameters (epochs, batch_size)
- Number of images to visualize
- Plot configurations

## Results

The project generates two sets of visualizations:
1. Initial comparison of original vs. noisy images
2. Final comparison of noisy vs. denoised vs. original images

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Results in 10 epochs:

![Screenshot 2024-10-19 202118](https://github.com/user-attachments/assets/ef1e1168-a308-4dee-b107-82870796e432)


## Acknowledgments

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- TensorFlow Documentation: https://www.tensorflow.org/
