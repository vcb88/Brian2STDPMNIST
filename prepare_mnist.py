from sklearn.datasets import fetch_openml
import numpy as np
import os

print("Downloading MNIST data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.reshape(-1, 28, 28)

# Create mnist directory
os.makedirs('./mnist', exist_ok=True)

# Save training data
print("Saving training data...")
with open('./mnist/train-images-idx3-ubyte', 'wb') as f:
    magic = 2051
    n_images = 60000
    n_rows = 28
    n_cols = 28
    f.write(np.array([magic, n_images, n_rows, n_cols], dtype='>i4').tobytes())
    f.write(X[:60000].tobytes())

with open('./mnist/train-labels-idx1-ubyte', 'wb') as f:
    magic = 2049
    n_labels = 60000
    f.write(np.array([magic, n_labels], dtype='>i4').tobytes())
    f.write(y[:60000].astype(int).tobytes())

# Save test data
print("Saving test data...")
with open('./mnist/t10k-images-idx3-ubyte', 'wb') as f:
    magic = 2051
    n_images = 10000
    n_rows = 28
    n_cols = 28
    f.write(np.array([magic, n_images, n_rows, n_cols], dtype='>i4').tobytes())
    f.write(X[60000:].tobytes())

with open('./mnist/t10k-labels-idx1-ubyte', 'wb') as f:
    magic = 2049
    n_labels = 10000
    f.write(np.array([magic, n_labels], dtype='>i4').tobytes())
    f.write(y[60000:].astype(int).tobytes())

print("MNIST data preparation completed!")