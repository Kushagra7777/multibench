"""
MultiBench Example - Part 3: MOSI with L2-MCTN (Cyclic TRANSLATION fusion)
=========================================================================

WHAT YOU WILL LEARN
-------------------
Parts 1 and 2 combined modalities that are all present at test time. But what if
a modality is missing or unreliable when you deploy? MCTN tackles exactly that.
By the end you should be able to explain:
  1. The idea of learning a JOINT representation by TRANSLATING between modalities.
  2. Why "cyclic" translation (there and back) makes that representation robust.
  3. What the "Level 2" (hierarchical) variant adds, and what each loss term does.

THE TASK
--------
Same dataset as Part 1 — CMU-MOSI sentiment regression — with three modalities
(language, acoustic, visual). The twist is HOW we fuse them.

L2-MCTN: Multimodal Cyclic Translation Network, Level 2 (the concept taught here)
---------------------------------------------------------------------------------
MCTN (Pham et al., AAAI 2019, "Found in Translation") learns a joint multimodal
representation by training a model to TRANSLATE from a source modality to a
target modality and then CYCLE BACK to the source:

        source --(encoder)--> joint repr --(decoder)--> target
        target --(encoder)--> joint repr --(decoder)--> source   (the cycle)

The encoder is forced to capture information about BOTH modalities, so the joint
representation it produces is useful for prediction. Crucially, at TEST time you
only need the SOURCE modality — the translation targets are not required — which
makes the model robust to missing modalities.

"Level 2" is the HIERARCHICAL version: two translation stages chained together
(encoder0/decoder0 then encoder1/decoder1) so the joint representation absorbs a
third modality. A small regression head then predicts sentiment from it.

This is the Python-script version of Multibench_Example_Usage_Jupyter_Part_3_MCTN.ipynb.
"""

import sys
import os

# --- Make MultiBench importable (see Part 1 for the full explanation) ------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# --- Sanity-check the data file -------------------------------------------
data_path = os.path.join(repo_root, "data", "affect", "mosi_raw.pkl")
print(f"Data path: {data_path}")
print(f"Data file exists: {os.path.exists(data_path)}")

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from datasets.affect.get_data import get_dataloader  # noqa

traindata, validdata, testdata = get_dataloader(data_path, robust_test=False)

from unimodals.common_models import GRU, MLP  # noqa
from fusions.MCTN import Encoder, Decoder  # noqa

# Feature dimensions for the translation network.
max_seq = 20         # max sequence length (timesteps per clip)
feature_dim = 300    # width of a raw modality feature vector (e.g. text embedding)
hidden_dim = 32      # width of the shared/joint representation

# TRANSLATION STAGE 0: source modality <-> first target modality.
#   encoder0 maps the source (feature_dim) into the joint space (hidden_dim);
#   decoder0 maps the joint space back out to a modality (feature_dim) — this is
#   what makes the cycle "there and back".
encoder0 = Encoder(feature_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
decoder0 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)

# TRANSLATION STAGE 1 (this is the "Level 2" part): take the joint representation
# from stage 0 and translate again to fold in the next modality.
encoder1 = Encoder(hidden_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
decoder1 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)

# REGRESSION ENCODER: summarises the final joint representation into a vector the
# prediction head can use.
reg_encoder = nn.GRU(hidden_dim, 32).to(device)

# PREDICTION HEAD: joint representation -> single sentiment score (regression).
head = MLP(32, 64, 1).to(device)

from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.MCTN_Level2 import train, test  # noqa

# all_in_one_train wraps the training in a memory/usage profiler. It needs every
# module so it can move them to the device and track their parameters.
allmodules = [encoder0, decoder0, encoder1, decoder1, reg_encoder, head]


def trainprocess():
    train(
        traindata,
        validdata,
        encoder0,                       # stage-0 encoder (source -> joint)
        decoder0,                       # stage-0 decoder (joint -> target, the cycle)
        encoder1,                       # stage-1 encoder ("Level 2" translation)
        decoder1,                       # stage-1 decoder
        reg_encoder,                    # summarises joint repr for the head
        head,                           # joint repr -> sentiment score
        # --- LOSS TERMS ---------------------------------------------------
        # Each translation/cycle step and the final prediction has its own loss:
        criterion_t0=nn.MSELoss(),      # stage-0 translation reconstruction loss
        criterion_c=nn.MSELoss(),       # cyclic-consistency loss (back to source)
        criterion_t1=nn.MSELoss(),      # stage-1 translation reconstruction loss
        criterion_r=nn.L1Loss(),        # regression loss on the sentiment score
        max_seq_len=20,
        # --- LOSS WEIGHTS (mu_*) ------------------------------------------
        # How much each translation loss counts relative to the regression loss.
        # Small values keep translation as an auxiliary objective, not the goal.
        mu_t0=0.01,                     # weight for criterion_t0
        mu_c=0.01,                      # weight for criterion_c
        mu_t1=0.01,                     # weight for criterion_t1
        dropout_p=0.15,
        early_stop=False,
        patience_num=15,                # epochs without improvement before stopping
        lr=1e-4,
        weight_decay=0.01,
        op_type=torch.optim.AdamW,
        epoch=50,                       # number of training epochs
        model_save="results/models/best_mctn.pt",  # best model is written here
    )


# Run training (profiled) and then evaluate the best checkpoint on the test set.
all_in_one_train(trainprocess, allmodules)

model = torch.load(
    "results/models/best_mctn.pt", map_location=device, weights_only=False
).to(device)

test(model, testdata, "mosi", no_robust=True)
