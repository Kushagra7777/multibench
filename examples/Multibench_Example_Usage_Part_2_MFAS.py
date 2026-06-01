"""
MultiBench Example - Part 2: AVMNIST with MFAS (Fusion Architecture SEARCH)
==========================================================================

WHAT YOU WILL LEARN
-------------------
In Part 1 we hand-designed the fusion (a simple early concatenation). Here we let
an algorithm DESIGN THE FUSION FOR US. By the end you should be able to explain:
  1. Why choosing *how* and *where* to fuse modalities is itself a hard problem.
  2. The core idea of MFAS: search over fusion architectures instead of guessing.
  3. The role of the "surrogate" model in making that search affordable.

THE TASK
--------
AVMNIST is a simple two-modality dataset built for teaching:
    - image  modality: a normal MNIST handwritten digit
    - audio  modality: a spectrogram of the spoken digit
Goal: classify the digit (10 classes) using both modalities.

MFAS: Multimodal Fusion Architecture Search (the concept this example teaches)
------------------------------------------------------------------------------
Given two pretrained unimodal encoders, there are MANY ways to fuse them: which
layer of the image encoder do you combine with which layer of the audio encoder,
and with what operation? Trying every combination by training each from scratch
is far too expensive.

MFAS (Pérez-Rúa et al., CVPR 2019) treats fusion design as a SEARCH problem:
    1. A "configuration" describes one candidate fusion architecture.
    2. A small, cheap-to-evaluate SURROGATE model learns to predict a
       configuration's accuracy without fully training it.
    3. The search uses the surrogate to propose promising configurations,
       trains those for real, and feeds the results back to improve the
       surrogate — progressively zeroing in on good fusion architectures.
The unimodal encoders are PRETRAINED and reused; only the fusion is searched.

DATA / WEIGHTS SETUP
--------------------
Expects the AVMNIST data under  data/avmnist/  and the two pretrained encoders
under  pretrained/avmnist/  (image_encoder.pt, audio_encoder.pt).

This is the Python-script version of Multibench_Example_Usage_Jupyter_Part_2_MFAS.ipynb.
"""

import sys
import os

# --- Make MultiBench importable (see Part 1 for the full explanation) ------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# --- Sanity-check the data directory --------------------------------------
data_path = os.path.join(repo_root, 'data', 'avmnist')
print(f"Data path: {data_path}")
print(f"Data directory exists: {os.path.isdir(data_path)}")

import torch
from torch import nn

# Use a GPU if available; the search trains many candidate models, so a GPU
# helps a lot here. It still runs on CPU, just slowly.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# `surrogate` is the learned cost model that predicts a fusion configuration's
# accuracy — the component that makes the architecture search affordable.
import utils.surrogate as surr  # noqa
from datasets.avmnist.get_data import get_dataloader  # noqa  -> AVMNIST data

# AVMNIST batches are [image, audio, label].
traindata, validdata, testdata = get_dataloader(data_path, batch_size=32)

# For MFAS you do NOT supply a fusion layer or a classification head: the search
# discovers both. You only provide the PRETRAINED unimodal encoders and the
# hyperparameters that define the search space.
from training_structures.architecture_search import train  # noqa  -> MFAS search

s_data = train(
    # Pretrained unimodal encoders to fuse (image first, then audio):
    [os.path.join(repo_root, 'pretrained', 'avmnist', 'image_encoder.pt'),
     os.path.join(repo_root, 'pretrained', 'avmnist', 'audio_encoder.pt')],
    16,              # rep_size: width of each encoder's output representation
    10,              # num classes (digits 0-9)
    [(6, 12, 24), (6, 12, 24, 48, 96)],  # per-layer output channels of each encoder
                                         # (image encoder, then audio encoder) — these
                                         # define which intermediate features can be fused
    traindata,       # training data loader
    validdata,       # validation data loader (used to score candidate fusions)
    surr.SimpleRecurrentSurrogate().to(device),  # surrogate accuracy predictor
    (3, 5, 2),       # search space of the fusion layer (depth / #ops / etc.)
    epochs=6         # epochs used to (partially) train each candidate during search
)
