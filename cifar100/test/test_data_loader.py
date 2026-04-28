# Run with: python cifar100/test/test_data_loader.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar100

x_train, y_train, x_test, y_test, class_names = load_cifar100()

# Check shapes
assert x_train.shape == (50000, 32, 32, 3), f"Expected (50000, 32, 32, 3), got {x_train.shape}"
assert x_test.shape  == (10000, 32, 32, 3), f"Expected (10000, 32, 32, 3), got {x_test.shape}"
assert y_train.shape == (50000, 100),        f"Expected (50000, 100), got {y_train.shape}"
assert y_test.shape  == (10000, 100),        f"Expected (10000, 100), got {y_test.shape}"

# Check pixel values are normalised to [0, 1]
assert x_train.max() <= 1.0, "Pixel values should be normalised to [0, 1]"
assert x_train.min() >= 0.0, "Pixel values should be normalised to [0, 1]"

# Check 100 class names
assert len(class_names) == 100, f"Expected 100 class names, got {len(class_names)}"

print("Train shape:", x_train.shape)
print("Test shape: ", x_test.shape)
print("Classes:    ", len(class_names), "—", class_names[:5], "...")
print("All data loader tests passed.")
