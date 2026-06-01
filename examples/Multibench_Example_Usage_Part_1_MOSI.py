"""
MultiBench Example - Part 1: Sentiment Analysis on MOSI with EARLY FUSION
=========================================================================

WHAT YOU WILL LEARN
-------------------
This is the "hello world" of multimodal learning. By the end you should be able
to explain:
  1. What the three modalities in MOSI are and how they are represented.
  2. What "early fusion" means and how it differs from late fusion.
  3. How MultiBench wires together encoders -> fusion -> prediction head and
     trains the whole thing end to end.

THE TASK
--------
CMU-MOSI is a multimodal sentiment dataset: short clips of people speaking to a
camera, each labelled with a sentiment score in the range [-3, +3] (very
negative ... very positive). Each clip provides THREE aligned modalities:
    - language  (text features, from word embeddings)
    - acoustic  (tone of voice features)
    - visual    (facial expression features)
We will predict the sentiment score (a regression task).

EARLY FUSION (the concept this example teaches)
-----------------------------------------------
"Fusion" is how a model combines information from several modalities. The two
classic strategies:
    * EARLY fusion  -> concatenate the raw per-modality features FIRST, then let
                       a single model reason over the combined vector. (This file.)
    * LATE  fusion  -> run a separate model per modality and combine their
                       PREDICTIONS at the end.
Early fusion lets the model learn cross-modal interactions from the very first
layer, at the cost of needing the modalities to be time-aligned (MOSI is).

DATA SETUP (run once before this script)
----------------------------------------
    wget https://filedn.eu/lDTxyzlMbdMJJq0AvECx20X/mosi_raw.pkl
    mv mosi_raw.pkl data/affect/mosi_raw.pkl

This is the Python-script version of Multibench_Example_Usage_Jupyter_Part_1_MOSI.ipynb.
"""

import sys
import os

# --- Make MultiBench importable -------------------------------------------
# This script lives in examples/, but the library lives one level up. We add the
# repo root to sys.path so that `import datasets`, `import fusions`, etc. resolve.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# --- Sanity-check the data file -------------------------------------------
# Catching a missing data file here gives a clearer error than a stack trace
# deep inside the dataloader.
data_path = os.path.join(repo_root, "data", "affect", "mosi_raw.pkl")
print(f"Data path: {data_path}")
print(f"Data file exists: {os.path.exists(data_path)}")

import torch

# Use a GPU if one is available; otherwise everything runs (more slowly) on CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MultiBench is organised into reusable building blocks. We import one of each:
from datasets.affect.get_data import get_dataloader  # noqa  -> data
from unimodals.common_models import (
    GRU,
    MLP,
    Sequential,
    Identity,
)  # noqa  -> encoders / head
from fusions.common_fusions import ConcatEarly  # noqa  -> fusion strategy
from training_structures.Supervised_Learning import (
    train,
    test,
)  # noqa  -> train/eval loop


def main():
    """Train and test a simple early-fusion model on MOSI."""

    print("Loading MOSI dataset...")

    # Each dataloader yields batches of [text, audio, vision, label].
    # max_pad / max_seq_len pad every clip to a fixed length of 50 timesteps so
    # the three modalities stay aligned and can be concatenated.
    traindata, validdata, testdata = get_dataloader(
        data_path, robust_test=False, max_pad=True, data_type="mosi", max_seq_len=50
    )

    print("Building model components...")

    # 1) ENCODERS — one per modality, applied BEFORE fusion.
    #    We deliberately use Identity (a pass-through) so that no per-modality
    #    learning happens here. This keeps the example focused on the fusion
    #    step: the GRU after fusion does all the representation learning.
    #    MOSI has 3 modalities, so we need 3 encoders: [text, audio, vision].
    encoders = [Identity().to(device), Identity().to(device), Identity().to(device)]

    # 2) FUSION — this is the heart of the example.
    #    ConcatEarly concatenates the three modality feature vectors along the
    #    feature dimension at EVERY timestep, producing one fused sequence.
    fusion = ConcatEarly().to(device)

    # 3) PREDICTION HEAD — consumes the fused sequence and outputs a score.
    #    The input width 409 is the sum of the three modalities' feature
    #    dimensions (text + audio + vision), i.e. the width of the concatenated
    #    vector produced by ConcatEarly. The GRU summarises the 50-step sequence
    #    into one 512-d vector (last_only=True), and the MLP maps that to a
    #    single number (output dim 1) because sentiment regression has 1 target.
    head = Sequential(
        GRU(
            409, 512, dropout=True, has_padding=False, batch_first=True, last_only=True
        ),
        MLP(512, 512, 1),
    ).to(device)

    print("Training model...")

    # The Supervised_Learning.train loop ties encoders + fusion + head together
    # and optimises them jointly.
    train(
        encoders,  # list of per-modality encoders (defined above)
        fusion,  # how modalities are combined (early concat)
        head,  # maps fused features -> prediction
        traindata,  # training batches
        validdata,  # validation batches (for model selection)
        total_epochs=5,  # short demo run; raise for real training
        task="regression",  # MOSI sentiment is a continuous score
        optimtype=torch.optim.AdamW,  # optimiser
        is_packed=False,  # inputs are padded (not PackedSequence)
        lr=1e-3,  # learning rate
        save="results/models/mosi_ef_r0.pt",  # best model is written here
        weight_decay=0.01,  # L2 regularisation
        objective=torch.nn.L1Loss(),  # mean-absolute-error loss for regression
    )

    print("\nTesting:")

    # Reload the BEST checkpoint saved during training (not necessarily the last
    # epoch) and evaluate it on the held-out test set.
    model = torch.load(
        "results/models/mosi_ef_r0.pt", map_location=device, weights_only=False
    ).to(device)
    # We trained on the regression score but report a binary metric here:
    # "posneg-classification" thresholds the predicted score at 0 to ask the
    # simpler question "did we get the sentiment polarity right?".
    test(
        model,
        testdata,
        dataset="affect",
        is_packed=False,
        criterion=torch.nn.L1Loss(),
        task="posneg-classification",
        no_robust=True,
    )


if __name__ == "__main__":
    main()
