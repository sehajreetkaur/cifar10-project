import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10

x_train, y_train, x_test, y_test, class_names = load_cifar10()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

sample_images = x_train[:5]

plt.figure(figsize=(12, 10))

for i in range(5):
    # original
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"Original: {class_names[y_train[i].argmax()]}")
    plt.axis("off")

    # augmented
    aug_iter = datagen.flow(sample_images[i:i+1], batch_size=1, shuffle=False)
    aug_image = next(aug_iter)[0]

    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(aug_image)
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.show()