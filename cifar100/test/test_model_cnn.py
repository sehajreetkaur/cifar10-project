# Run with: python cifar100/test/test_model_cnn.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.model_cnn import build_cnn

model = build_cnn()

# Check output shape is 100 (one per CIFAR-100 class)
assert model.output_shape == (None, 100), f"Expected (None, 100), got {model.output_shape}"

# Check model has layers
assert len(model.layers) > 0, "Model should have layers"

model.summary()

print("Output shape:", model.output_shape)
print("All model tests passed.")
