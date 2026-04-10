"""
MultiBench Example Usage - Part 2: AVMNIST (MFAS / Architecture Search)

Demonstrates MultiBench's MultiModal Fusion Architecture Search (MFAS) system
on the AVMNIST dataset (audio + visual MNIST digits).
"""

import sys
import os

# Add the MultiBench root directory to the path so imports resolve correctly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# Verify data directory
data_path = os.path.join(repo_root, 'data', 'avmnist')
print(f"Data path: {data_path}")
print(f"Data directory exists: {os.path.isdir(data_path)}")

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import utils.surrogate as surr  # This imports a learned cost model from configurations to accuracies.
from datasets.avmnist.get_data import get_dataloader  # This imports the AVMNIST dataloader

traindata, validdata, testdata = get_dataloader(data_path, batch_size=32)

# To train the MFAS model, you don't need to feed in a fusion layer nor a classification
# head, as both of those are looked after through MFAS. Instead, just provide the pretrained
# encoder files for each modality encoder and the associated hyperparameters.
from training_structures.architecture_search import train  # This imports the MFAS training method.

s_data = train(
    [os.path.join(repo_root, 'pretrained', 'avmnist', 'image_encoder.pt'),
     os.path.join(repo_root, 'pretrained', 'avmnist', 'audio_encoder.pt')],
    16,              # Size of encoder output
    10,              # Number of classes
    [(6, 12, 24), (6, 12, 24, 48, 96)],  # Output of each layer within the unimodal encoders
    traindata,       # Training data loader
    validdata,       # Validation data loader
    surr.SimpleRecurrentSurrogate().to(device),  # Surrogate instance
    (3, 5, 2),       # Search space of the fusion layer
    epochs=6         # Number of epochs
)
