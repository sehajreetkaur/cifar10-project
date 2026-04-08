"""
train_baseline.py

Train the baseline CNN on CIFAR-10, evaluate it, and save training artifacts.

Saved artifacts:
- outputs/CNN.keras
- outputs/baseline_results.json
- outputs/baseline_training_curves.png
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn


CONFIG = {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "validation_split": 0.2,
}


def plot_training_curves(history, output_dir):
    """Plot and save training/validation accuracy and loss curves."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.history["accuracy"], label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Baseline CNN - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Baseline CNN - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "baseline_training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


def train_baseline_model():
    """Run full baseline training and save artifacts."""
    print("=" * 60)
    print("  Training baseline CNN on CIFAR-10")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key:<18} {value}")
    print()

    x_train, y_train, x_test, y_test, _ = load_cifar10()

    model = build_baseline_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["validation_split"],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nBaseline CNN - Test Accuracy : {test_acc:.4f}")
    print(f"Baseline CNN - Test Loss     : {test_loss:.4f}")

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "CNN.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    results = {
        **CONFIG,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    results_path = os.path.join(output_dir, "baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    plot_training_curves(history, output_dir)
    return model, history, test_acc


def main():
    train_baseline_model()


if __name__ == "__main__":
    main()
