# CIFAR-10 Image Classification

A CNN-based image classification project on the CIFAR-10 dataset. Covers baseline training, data augmentation, and transfer learning with ResNet50.

---

## Setup

**Requirements:** Python 3.9

### 1. Clone the repository
```bash
git clone https://github.com/sehajreetkaur/cifar10-project.git
cd cifar10-project
```

### 2. Create and activate a virtual environment
```bash
python3.9 -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Baseline CNN
```bash
python src/train_baseline.py
```
Trains for 10 epochs, prints test accuracy, saves model to `outputs/CNN.keras`.

### Data Augmentation
```bash
python src/augmentation.py
```
Shows augmented image previews, trains both runs, plots and saves comparison curves.

To view TensorBoard logs after training:
```bash
tensorboard --logdir=logs/augmentation
```

### Transfer Learning
```bash
python src/transfer_learning.py
```
Downloads ResNet50 weights (first run only), trains in two phases, prints accuracy comparison, saves model and plots to `outputs/`.

---

## What Each File Does

### `src/data_loader.py`
Loads CIFAR-10 from Keras, normalises pixel values to [0,1], and one-hot encodes the labels. Returns train/test splits and class names.

### `src/model_CNN.py`
Defines the baseline CNN architecture:
- 3 × Conv2D layers (32 → 64 → 64 filters)
- MaxPooling after each of the first two
- Dense(64) → Dense(10, softmax)
- Compiled with Adam + categorical crossentropy

### `src/train_baseline.py`
Trains the baseline CNN on CIFAR-10 for 10 epochs. Evaluates on the test set and saves the model to `outputs/CNN.keras`.

### `src/augmentation.py`
- Applies augmentation (rotation, shifts, zoom, horizontal flip)
- Previews original vs augmented images
- Trains the CNN without and with augmentation for comparison
- Plots training curves for both runs side by side
- Saves comparison plot to `outputs/augmentation_comparison.png`
- Logs to TensorBoard at `logs/augmentation/`

### `src/transfer_learning.py`
- Loads ResNet50 pretrained on ImageNet (no top layer)
- Resizes CIFAR-10 images from 32×32 to 224×224
- Adds a custom head: GlobalAveragePooling → Dense(256) → Dropout → Dense(10)
- Phase 1: trains the head only (base frozen, 10 epochs)
- Phase 2: fine-tunes the full model at a low learning rate (5 epochs)
- Compares accuracy vs baseline CNN
- Saves model to `outputs/resnet50_cifar10.keras`
- Saves architecture diagram and training curves to `outputs/`
