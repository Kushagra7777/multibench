"""
MultiBench Example: MOSI with Simple Early Fusion

This example demonstrates basic usage of MultiBench with the affective computing 
dataset MOSI using a simple early fusion model.

This is the Python script version of Multibench_Example_Usage_On_Colab_Part_1_EARL.ipynb
"""
import sys
import os

# Add the MultiBench root directory to the path so imports resolve correctly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# Verify data file
data_path = os.path.join(repo_root, 'data', 'affect', 'mosi_raw.pkl')
print(f"Data path: {data_path}")
print(f"Data file exists: {os.path.exists(data_path)}")

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from datasets.affect.get_data import get_dataloader  # noqa
from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
from training_structures.Supervised_Learning import train, test  # noqa


def main():
    """
    Train and test a simple early fusion model on MOSI dataset.
    
    Note: Download the MOSI data file first:
    wget https://filedn.eu/lDTxyzlMbdMJJq0AvECx20X/mosi_raw.pkl
    mv mosi_raw.pkl data/affect/mosi_raw.pkl
    """

    print("Loading MOSI dataset...")
    
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
            GRU(409, 
                512, 
                dropout=True, 
                has_padding=False, 
                batch_first=True, 
                last_only=True), 
            MLP(512, 512, 1)
    ).to(device)
    
    print("Training model...")
    
    # Train the model
    train(
        encoders, fusion, head, 
        traindata, validdata, 
        total_epochs=2,
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
    model = torch.load('mosi_ef_r0.pt', 
                       map_location=device,
                       weights_only=False).to(device)
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
