"""
augmentation.py

This file does two things:
1. Visualizes CIFAR-10 image augmentation by showing original images
   alongside augmented versions.
2. Trains the baseline CNN model using augmented training data and
   saves TensorBoard logs for monitoring performance.

Why this file exists:
- It keeps augmentation preview and augmentation training together
  in one place.
- The augmentation settings are defined once and reused for both
  visualization and training.
"""

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn


def create_datagen():
    """
    Create and return the image augmentation generator.

    The same augmentation settings are used for both:
    - previewing augmented images
    - training the CNN model
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )


def preview_augmentation(x_train, y_train, class_names, datagen, num_samples=5):
    """
    Display a few original CIFAR-10 images next to their augmented versions.

    Args:
        x_train: Training images
        y_train: One-hot encoded training labels
        class_names: List of CIFAR-10 class names
        datagen: ImageDataGenerator object
        num_samples: Number of samples to display
    """
    # Select a small number of training images for visualization
    sample_images = x_train[:num_samples]

    # Create the plotting window
    plt.figure(figsize=(12, 2 * num_samples))

    for i in range(num_samples):
        # Show the original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Original: {class_names[y_train[i].argmax()]}")
        plt.axis("off")

        # Generate one augmented version of the same image
        aug_iter = datagen.flow(sample_images[i:i+1], batch_size=1, shuffle=False)
        aug_image = next(aug_iter)[0]

        # Show the augmented image
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(aug_image)
        plt.title("Augmented")
        plt.axis("off")

    # Adjust spacing and display the plot
    plt.tight_layout()
    plt.show()


def train_with_augmentation(x_train, y_train, x_test, y_test, datagen):
    """
    Train the baseline CNN model using augmented CIFAR-10 training data.

    Also creates TensorBoard logs for tracking training progress.

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        datagen: ImageDataGenerator object

    Returns:
        model: Trained CNN model
        history: Training history returned by model.fit()
    """
    print("Starting augmentation training...")

    # Fit the generator to the training data
    datagen.fit(x_train)

    # Build the baseline CNN model
    model = build_baseline_cnn()

    # Create a folder for TensorBoard logs
    log_dir = os.path.join(project_root, "logs", "augmentation")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard callback for logging training details
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    # Train the model using augmented batches
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("Augmented Test Accuracy:", test_acc)
    print("Augmented Test Loss:", test_loss)
    print("TensorBoard logs saved to:", log_dir)

    return model, history


def main():
    """
    Main function:
    - loads CIFAR-10 data
    - creates the augmentation generator
    - previews augmented images
    - trains the CNN with augmentation
    """
    # Load dataset and class labels
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    # Create augmentation pipeline
    datagen = create_datagen()

    # Show sample augmented images
    preview_augmentation(x_train, y_train, class_names, datagen)

    # Train the model using augmentation
    train_with_augmentation(x_train, y_train, x_test, y_test, datagen)


# Run the script only when executed directly
if __name__ == "__main__":
    main()