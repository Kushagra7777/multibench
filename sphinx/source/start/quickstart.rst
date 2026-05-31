Quick Start
***********

This guide gets you from a fresh install to a trained multimodal model in a
few minutes. Make sure you have followed the :doc:`installation` steps first.

A first experiment: early fusion on CMU-MOSI
============================================

This example trains a simple early-fusion model on the
`CMU-MOSI <https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing>`_
sentiment dataset (three modalities: text, audio, vision).

First, download the MOSI data:

.. code-block:: bash

   pip install gdown
   gdown https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU
   mkdir -p data/affect && mv mosi_raw.pkl data/affect/

Then train and test the model:

.. code-block:: python

   import torch
   from datasets.affect.get_data import get_dataloader
   from unimodals.common_models import GRU, MLP, Sequential, Identity
   from fusions.common_fusions import ConcatEarly
   from training_structures.Supervised_Learning import train, test
   from utils.device import get_device

   device = get_device()  # automatically selects CUDA, MPS (Apple Silicon), or CPU

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
         save='results/models/mosi_ef_r0.pt', objective=torch.nn.L1Loss())

   # Test
   model = torch.load('results/models/mosi_ef_r0.pt', weights_only=False).to(device)
   test(model, testdata, dataset='affect', is_packed=False,
        criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)

.. note::

   Trained checkpoints are saved to ``results/models/`` and robustness plots to
   ``results/images/`` by default, so your experiment artifacts stay in one
   place instead of scattering across the repository root.

Quickest experiments to get started
====================================

If you just want to confirm your install works and see the full
data → train → evaluate pipeline run end to end, these are the fastest
entry points. All run on **CPU** in well under a minute with the default
2-epoch example settings.

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 14 10

   * - Experiment
     - Script
     - Data
     - CPU runtime
     - Model params
   * - Stock prediction
     - ``examples/finance/stocks_late_fusion.py``
     - Auto-downloads (e.g. ``AAPL MSFT``)
     - ~15 s
     - ~2.5 K
   * - AV-MNIST (late fusion)
     - ``examples/multimedia/avmnist_simple_late_fusion.py``
     - 2,000-sample subset (of 60 K)
     - ~1 min
     - ~261 K
   * - Gentle Push (unimodal)
     - ``examples/gentle_push/unimodal_image.py``
     - 10-trajectory eval set
     - ~1 min
     - varies

**Smallest / fastest overall:** Stock prediction needs no manual download
(data is fetched on first run via ``yfinance``) and finishes in seconds,
making it the best choice for a first smoke test or for quickly iterating on
model architecture. AV-MNIST is the simplest *multimodal* starting point — its
example already subsets to 2,000 training samples and 2 epochs.

Smoke-test results
==================

The numbers below were captured from quick CPU runs of the pipeline. They
illustrate relative speed and model size, not benchmark accuracy.

.. list-table::
   :header-rows: 1
   :widths: 26 16 16 18 14

   * - Metric
     - Stock
     - AV-MNIST
     - Gentle Push
     - Device
   * - Total runtime
     - ~15 s
     - ~57 s
     - ~3.4 s
     - CPU
   * - Training time
     - ~4.8 s
     - ~32.2 s
     - ~0.35 s
     - CPU
   * - Inference time
     - ~0.23 s
     - ~14.3 s
     - ~0.008 s
     - CPU
   * - Model parameters
     - 2,465
     - 260,922
     - 2,130,499
     - --
   * - Modalities
     - 6 stocks
     - 2 (vision+audio)
     - 3-4 (vision, position, sensors, control)
     - --

.. warning::

   Only the **stock-prediction** run above used real data; the AV-MNIST and
   Gentle Push timings were measured against small synthetic placeholder arrays
   (the real datasets had not yet been downloaded). Treat those rows as
   *pipeline speed* measurements, not model accuracy. For real benchmark
   numbers, download the datasets via the :doc:`datadownload` guide and train
   for the full epoch counts.

Running other experiments
=========================

Each dataset has dedicated example scripts under ``examples/``:

.. code-block:: bash

   # Affective computing
   python examples/affect/affect_late_fusion.py

   # Healthcare (requires MIMIC access)
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
