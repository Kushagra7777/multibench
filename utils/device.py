# utils/device.py
import os
import torch


def get_device():
    if os.getenv("MULTIBENCH_FORCE_CPU", "0") == "1":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

# ### Apple Silicon / MPS note

# Some models may raise tensor device mismatch errors on Apple Silicon when MPS is selected automatically. To force CPU execution, run:

# ```bash

# MULTIBENCH_FORCE_CPU=1 python examples/finance/stocks_mult.py \

#   --input-stocks "AAPL MSFT GOOG" \

#   --target-stock AAPL
  
# ###