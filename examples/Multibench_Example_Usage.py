"""
MultiBench Example: MOSI with Simple Early Fusion

This example demonstrates basic usage of MultiBench with the affective computing 
dataset MOSI using a simple early fusion model.

This is the Python script version of Multibench_Example_Usage_Colab.ipynb
"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
import os

# Add repository to path (REQUIRED)
sys.path.append(os.getcwd())

from datasets.affect.get_data import get_dataloader  # noqa
from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
from training_structures.Supervised_Learning import train, test  # noqa


def main():
    """
    Train and test a simple early fusion model on MOSI dataset.
    
    Note: Download the MOSI data file first:
    - Create a 'data' directory in the repo root
    - Download from: https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU
    - Save as 'data/mosi_raw.pkl'
    
    Or use gdown: pip install gdown && gdown https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU
    mv mosi_raw.pkl data/mosi_raw.pkl
    """
    
    # Path to MOSI data - adjust this to your local path
    data_path = 'data/mosi_raw.pkl'
    
    print("Loading MOSI dataset...")
    print("Note: First run will download GloVe embeddings (~2GB) to ~/.cache/glove/")
    
    # Create the training, validation, and test-set dataloaders
    traindata, validdata, testdata = get_dataloader(
        data_path, 
        robust_test=False, 
        max_pad=True, 
        data_type='mosi', 
        max_seq_len=50
    )
    
    print("Building model components...")
    
    # Define encoders (using Identity for simplicity - passes through raw features)
    # MOSI has 3 modalities: text, audio, vision
    encoders = [Identity().to(device), Identity().to(device), Identity().to(device)]
    
    # Define fusion paradigm (early concatenation along dimension 2)
    fusion = ConcatEarly().to(device)
    
    # Define prediction head
    # Input dimension 409 comes from concatenated modality features
    # Output dimension 1 for regression task (sentiment prediction)
    head = Sequential(
        GRU(409, 512, dropout=True, has_padding=False, batch_first=True, last_only=True), 
        MLP(512, 512, 1)
    ).to(device)
    
    print("Training model...")
    
    # Train the model
    train(
        encoders, fusion, head, 
        traindata, validdata, 
        total_epochs=100,
        task="regression",
        optimtype=torch.optim.AdamW,
        is_packed=False,
        lr=1e-3,
        save='mosi_ef_r0.pt',
        weight_decay=0.01,
        objective=torch.nn.L1Loss()
    )
    
    print("\nTesting:")
    
    # Load best model and test
    model = torch.load('mosi_ef_r0.pt', weights_only=False).to(device)
    test(
        model, testdata, 
        dataset='affect',
        is_packed=False,
        criterion=torch.nn.L1Loss(),
        task="posneg-classification",
        no_robust=True
    )


if __name__ == "__main__":
    main()
