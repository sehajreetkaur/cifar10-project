"""
model_improved_cnn.py

What this file does:
--------------------
Defines an improved CNN architecture for CIFAR-10 classification.

How it differs from model_CNN.py (the baseline):
-------------------------------------------------
Baseline (model_CNN.py):
  - Filters: 32 → 64 → 64
  - Activations: ReLU (inline, per Conv2D)
  - No Batch Normalization
  - No Dropout in conv blocks
  - No padding (spatial dims shrink each layer)

This file (model_improved_cnn.py):
  - Filters: 64 → 128 → 256  (higher capacity)
  - Activations: LeakyReLU(alpha=0.1)  (no dead neurons)
  - BatchNormalization after every Conv2D  (stable, faster training)
  - Dropout(0.4) before Dense layers  (reduces overfitting)
  - padding='same' on all Conv2D  (preserves spatial dims through deeper stack)

Lecture grounding:
  - Deeper filter progression → more capacity to learn complex features
    (Lecture: "depth vs width — deeper networks learn hierarchical features")
  - LeakyReLU fixes "dead neuron" problem described for ReLU
    (Lecture: "whenever a neuron receives negative input, gradient = 0")
  - BatchNorm stabilizes gradient flow through the network
  - Dropout prevents co-adaptation of neurons → better generalisation

How to use:
    from src.model_improved_cnn import build_improved_cnn
    model = build_improved_cnn()
    model.summary()
"""

from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU


def build_improved_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.4):
    """
    Build an improved CNN for CIFAR-10 with BatchNorm, LeakyReLU, and Dropout.

    Architecture overview:
        Block 1 — Conv2D(64) → BN → LeakyReLU → MaxPool
        Block 2 — Conv2D(128) → BN → LeakyReLU → MaxPool
        Block 3 — Conv2D(256) → BN → LeakyReLU  (no pool — preserve spatial info)
        Head    — Flatten → Dense(256) → BN → LeakyReLU → Dropout → Dense(10, softmax)

    Args:
        input_shape:  Shape of each input image. Default (32, 32, 3) for CIFAR-10.
        num_classes:  Number of output classes. Default 10.
        dropout_rate: Dropout probability before the final Dense layer. Default 0.4.
                      Increase toward 0.5 if you observe overfitting in training curves.

    Returns:
        model: Compiled Keras Sequential model.
    """
    model = models.Sequential([

        # ── Input ──────────────────────────────────────────────────────────
        layers.Input(shape=input_shape),

        # ── Block 1: 64 filters ────────────────────────────────────────────
        # padding='same' keeps the output the same spatial size as input (32×32),
        # preventing information loss at the edges in early layers.
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        # use_bias=False because BatchNormalization has its own bias term (beta).
        # Adding a Conv2D bias on top is redundant and wastes parameters.
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),   # 32×32 → 16×16

        # ── Block 2: 128 filters ───────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),   # 16×16 → 8×8

        # ── Block 3: 256 filters ───────────────────────────────────────────
        # No MaxPooling here — we keep the 8×8 spatial resolution before flattening
        # so the Dense head receives more spatial information.
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),

        # ── Classification head ────────────────────────────────────────────
        layers.Flatten(),

        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),

        # Dropout is placed here, after the activations, immediately before
        # the final output layer — this is the standard placement.
        layers.Dropout(dropout_rate),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",           # default lr=1e-3; train_improved.py will tune this
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick sanity check — run this file directly to confirm the architecture
    model = build_improved_cnn()
    model.summary()
    print("\nModel built successfully.")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Output shape:     {model.output_shape}")