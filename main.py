from utils import add_noise, build_autoencoder
import numpy as np  
from matplotlib import pylab as plt 
import tensorflow as tf


# 1. Load and preprocess MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to range [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Add a channel dimension (needed for Conv2D)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Add noise to the data
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# Visualize original and noisy images
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Noisy images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_train_noisy[i].reshape(28, 28), cmap='gray')
    plt.title('Noisy')
    plt.axis('off')

plt.show()




# Create the autoencoder model
autoencoder = build_autoencoder()

# 4. Train the model
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# 5. Visualize denoised results
decoded_imgs = autoencoder.predict(x_test_noisy)

# Plot noisy, original, and denoised images
n = 5
plt.figure(figsize=(10, 6))
for i in range(n):
    # Noisy images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title('Noisy')
    plt.axis('off')

    # Denoised images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title('Denoised')
    plt.axis('off')

    # Original images
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

plt.show()