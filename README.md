# MultiBench

**Standardized toolkit for multimodal deep learning research**

[![codecov](https://codecov.io/gh/pliang279/MultiBench/branch/main/graph/badge.svg?token=IN899HIWCF)](https://codecov.io/gh/pliang279/MultiBench)
[![Documentation Status](https://readthedocs.org/projects/multibench/badge/?version=latest)](https://multibench.readthedocs.io/en/latest/?badge=latest)

![Overview](images/overview.png)

MultiBench is a systematic, unified large-scale benchmark spanning **15 datasets**, **10 modalities**, **20 prediction tasks**, and **6 research areas**. It provides an automated end-to-end ML pipeline that simplifies data loading, experimental setup, and model evaluation.

Paired with **MultiZoo**, a modular collection of 20 core multimodal learning approaches, MultiBench holistically evaluates:

1. **Performance** across domains and modalities
2. **Complexity** during training and inference
3. **Robustness** to noisy and missing modalities

[Website](https://cmu-multicomp-lab.github.io/multibench/) · [Documentation](https://multibench.readthedocs.io/en/latest/) · [Tutorials](https://github.com/pliang279/MultiBench/tree/main/examples)

## Supported datasets

| Research Area        | Datasets                                           |
| -------------------- | -------------------------------------------------- |
| Affective Computing  | CMU-MOSI, CMU-MOSEI, MUStARD, UR-FUNNY            |
| Healthcare           | MIMIC                                              |
| Robotics             | Vision & Touch, MuJoCo Push (Gentle Push)          |
| Finance              | Stocks-Food, Stocks-Health, Stocks-Tech            |
| HCI                  | ENRICO                                             |
| Multimedia           | AV-MNIST, MM-IMDb, Kinetics-S, Kinetics-L         |

![Datasets](images/datasets.png)

## Supported algorithms

![MultiZoo](images/multizoo.png)

- **Unimodal models** — MLP, GRU, LSTM, CNN, LeNet, ResNet, Transformer, FCN, Random Forest, etc. (`unimodals/`)
- **Fusion paradigms** — Early/Late Fusion, Tensor Fusion, Multiplicative Interactions, Low-Rank Tensor Fusion, NL-Gate, MulT, etc. (`fusions/`)
- **Objective functions** — CrossEntropy, MSE, ELBO, CCA, Contrastive Loss, Reconstruction Loss, etc. (`objective_functions/`)
- **Training structures** — Supervised Learning (Early/Late Fusion, MVAE, MFM), Gradient Blend, Architecture Search, MCTN, etc. (`training_structures/`)

## Getting started

### Prerequisites

- Python >= 3.10
- PyTorch (CPU or GPU)

### Installation

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install memory-profiler scikit-learn scipy matplotlib h5py tqdm
```

### Quick example

This example trains a simple early fusion model on the [CMU-MOSI](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing) sentiment dataset.

```bash
# Download the MOSI dataset
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU
mkdir -p data/affect && mv mosi_raw.pkl data/affect/
```

```python
import torch
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Sequential, Identity
from fusions.common_fusions import ConcatEarly
from training_structures.Supervised_Learning import train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data (3 modalities: text, audio, vision)
traindata, validdata, testdata = get_dataloader(
    'data/affect/mosi_raw.pkl',
    data_type='mosi', max_pad=True, max_seq_len=50
)

# Define model components
encoders = [Identity().to(device) for _ in range(3)]
fusion = ConcatEarly().to(device)
head = Sequential(
    GRU(409, 512, dropout=True, has_padding=False, batch_first=True, last_only=True),
    MLP(512, 512, 1)
).to(device)

# Train
train(encoders, fusion, head, traindata, validdata,
      total_epochs=10, task="regression",
      optimtype=torch.optim.AdamW, lr=1e-3,
      save='mosi_ef_r0.pt', objective=torch.nn.L1Loss())

# Test
model = torch.load('mosi_ef_r0.pt', weights_only=False).to(device)
test(model, testdata, dataset='affect', is_packed=False,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
```

> [!TIP]
> Check out the [Colab tutorials](https://github.com/pliang279/MultiBench/tree/main/examples) for step-by-step walkthroughs covering basic fusion, architecture search (MFAS), and cross-modal translation (MCTN).

## Running experiments

Each dataset has dedicated example scripts under `examples/`. Here are some common ones:

```bash
# Affective computing
python examples/affect/affect_late_fusion.py

# Healthcare (requires MIMIC access — see dataset section below)
python examples/healthcare/mimic_low_rank_tensor.py

# Robotics
python examples/robotics/LRTF.py
python examples/gentle_push/LF.py

# Finance (specify input and target stocks)
python examples/finance/stocks_late_fusion.py --input-stocks 'AAPL MSFT AMZN INTC AMD MSI' --target-stock 'MSFT'

# HCI
python examples/hci/enrico_simple_late_fusion.py

# Multimedia
python examples/multimedia/avmnist_simple_late_fusion.py
python examples/multimedia/mmimdb_simple_late_fusion.py
```

## Dataset access

Most datasets require a one-time download. Links to processed data:

| Dataset | Download |
| ------- | -------- |
| CMU-MOSI | [Google Drive](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing) |
| CMU-MOSEI | [Google Drive](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv?usp=sharing) |
| MUStARD | [Google Drive](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7?usp=sharing) |
| UR-FUNNY | [Google Drive](https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=sharing) |
| AV-MNIST | [Google Drive](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing) |
| MM-IMDb | [Archive.org (hdf5)](https://archive.org/download/mmimdb/multimodal_imdb.hdf5) · [Archive.org (raw)](https://archive.org/download/mmimdb/mmimdb.tar.gz) |
| Vision & Touch | Run `datasets/robotics/download_data.sh` |
| MuJoCo Push | Auto-downloads on first run |
| Finance | Auto-downloads on first run |
| ENRICO | [GitHub](https://github.com/luileito/enrico) |
| Clotho | [GitHub](https://github.com/audio-captioning/clotho-dataset) |

> [!NOTE]
> **MIMIC** has restricted access. Follow [these instructions](https://mimic.mit.edu/iv/access/) to obtain credentials, then email yiweilyu@umich.edu with proof to request the preprocessed `im.pk` file.

## Evaluation

### Complexity

Track peak memory, parameter count, and runtime using `eval_scripts/complexity.py`:

```python
from eval_scripts.complexity import all_in_one_train, all_in_one_test
```

See `examples/healthcare/mimic_baseline_track_complexity.py` for a full example.

### Robustness

Modality-specific noise implementations are in `robustness/`, with evaluation scripts in `eval_scripts/robustness.py`. Robustness testing is integrated into the standard training/testing pipeline.

![Robustness plots](images/robustness_plots.png)

## Project structure

```
multibench/
├── datasets/              # Data loaders for all 15 datasets
├── unimodals/             # Single-modality encoders
├── fusions/               # Multimodal fusion strategies
├── objective_functions/   # Loss functions and objectives
├── training_structures/   # Training paradigms (supervised, NAS, etc.)
├── eval_scripts/          # Performance, complexity, robustness evaluation
├── robustness/            # Modality-specific noise implementations
├── examples/              # Runnable scripts and Colab notebooks
├── pretrained/            # Pre-trained model weights
└── utils/                 # Helper modules
```

## Adding new datasets or algorithms

**New dataset:**

1. Add a folder under `datasets/` with a `get_data.py` containing a `get_dataloader` function that returns `(train, valid, test)` dataloaders
2. Add example scripts under `examples/`

**New algorithm:**

1. Add your module to the appropriate subfolder: `unimodals/`, `fusions/`, `objective_functions/`, or `training_structures/`
2. Add example scripts and document input/output formats

## Citation

If you use MultiBench in your research, please cite:

```bibtex
@article{liang2023multizoo,
  title={MULTIZOO \& MULTIBENCH: A Standardized Toolkit for Multimodal Deep Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Agarwal, Arav and Cheng, Yun and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  journal={Journal of Machine Learning Research},
  volume={24},
  pages={1--7},
  year={2023}
}
```

```bibtex
@inproceedings{liang2021multibench,
  title={MultiBench: Multiscale Benchmarks for Multimodal Representation Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Wu, Zetian and Cheng, Yun and Wu, Jason and Chen, Leslie Yufan and Wu, Peter and Lee, Michelle A and Zhu, Yuke and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021}
}
```

## Contributors

- [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) · [Yiwei Lyu](https://github.com/lvyiwei1) · [Xiang Fan](https://github.com/sfanxiang) · [Zetian Wu](http://neal-ztwu.github.io) · [Yun Cheng](https://kapikantzari.github.io) · [Arav Agarwal](https://www.linkedin.com/in/arav-agarwal-941b44109/) · [Jason Wu](https://jasonwunix.com/) · Leslie Chen · [Peter Wu](https://peter.onrender.com/) · [Michelle A. Lee](http://stanford.edu/~mishlee/) · [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) · [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) · [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)
