"""MLP, reproducible split and shared normalization. No pretrained weights/CNN."""
import hashlib
import random
from pathlib import Path
import urllib.request
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        previous = 784
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Raw logits; CrossEntropyLoss includes log-softmax.

def normalize(images):
    return np.asarray(images, dtype=np.float32) / 255.0

def ensure_data(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        try:
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path

def load_train(path):
    # Test arrays intentionally not accessed here.
    with np.load(ensure_data(path), allow_pickle=False) as data:
        x, y = data['x_train'], data['y_train']
    if x.shape != (60000, 28, 28) or y.shape != (60000,):
        raise ValueError('Expected official MNIST train shapes.')
    return x, y

def split_indices(labels, seed=42):
    return train_test_split(np.arange(len(labels)), test_size=10000,
                            stratify=labels, random_state=seed)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
