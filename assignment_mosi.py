"""
TASK: Multimodal Fusion Comparison on CMU-MOSI
"""

import sys
import os
import time
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Sequential, Identity
from fusions.common_fusions import ConcatEarly, Concat, TensorFusion
from training_structures.Supervised_Learning import train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

data_path = os.path.join(repo_root, "data", "affect", "mosi_raw.pkl")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"MOSI data not found at {data_path}\n"
        f"Please download from: https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU\n"
        f"And place at: {data_path}"
    )

print(f"Data file found: {data_path}\n")

configs = {
    "Early Fusion": {
        "fusion_class": ConcatEarly,
        "encoder_config": {
            "type": "Identity",
            "count": 3,
            "params": {}
        },
        "head_config": {
            "gru_input": 409,
            "gru_hidden": 512,
            "gru_dropout": True,
            "mlp_output": 1
        },
        "data_loading": {
            "max_pad": True,
            "data_type": "mosi",
            "max_seq_len": 50
        },
        "training_config": {
            "is_packed": False,
            "epochs": 5
        },
        "save_path": "results/models/mosi_ef_r0.pt",
        "description": "Concatenate raw features, single GRU encoder over fused sequence"
    },
    "Late Fusion": {
        "fusion_class": Concat,
        "encoder_config": {
            "type": "GRU_per_modality",
            "dims": [(35, 70), (74, 200), (300, 600)],
            "has_padding": True,
            "dropout": True
        },
        "head_config": {
            "mlp_input": 870,
            "mlp_hidden": 870,
            "mlp_output": 1
        },
        "data_loading": {
            "max_pad": False,
            "data_type": "mosi",
            "max_seq_len": None
        },
        "training_config": {
            "is_packed": True,
            "epochs": 5
        },
        "save_path": "results/models/mosi_lf_best.pt",
        "description": "Per-modality GRU encoders, concatenate predictions"
    },
    "Tensor Fusion": {
        "fusion_class": TensorFusion,
        "encoder_config": {
            "type": "GRU_last_only",
            "dims": [(35, 4), (74, 19), (300, 79)],
            "last_only": True,
            "dropout": True
        },
        "head_config": {
            "mlp_input": 8000,
            "mlp_hidden": 512,
            "mlp_output": 1
        },
        "data_loading": {
            "max_pad": True,
            "data_type": None,
            "max_seq_len": None
        },
        "training_config": {
            "is_packed": False,
            "epochs": 5
        },
        "save_path": "results/models/mosi_tf_best.pt",
        "description": "Bilinear fusion via outer product of encoder vectors"
    }
}

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data_for_method(method_name: str, data_path: str, config: Dict[str, Any]):
    data_cfg = config["data_loading"]
    print(f"\nLoading data for {method_name}...")
    
    loader_kwargs = {
        "robust_test": False,
        "num_workers": 0
    }
    
    if data_cfg.get("max_pad") is not None:
        loader_kwargs["max_pad"] = data_cfg["max_pad"]
    if data_cfg.get("data_type") is not None:
        loader_kwargs["data_type"] = data_cfg["data_type"]
    if data_cfg.get("max_seq_len") is not None:
        loader_kwargs["max_seq_len"] = data_cfg["max_seq_len"]
    
    traindata, validdata, testdata = get_dataloader(data_path, **loader_kwargs)
    print(f"Data loaded: train={len(traindata.dataset)}, valid={len(validdata.dataset)}, test={len(testdata.dataset)}")
    return traindata, validdata, testdata

def build_model(method_name: str, config: Dict[str, Any], device: torch.device):
    print(f"\nBuilding {method_name} model...")
    encoder_cfg = config["encoder_config"]
    
    if encoder_cfg["type"] == "Identity":
        encoders = [Identity().to(device) for _ in range(encoder_cfg["count"])]
    elif encoder_cfg["type"] == "GRU_per_modality":
        encoders = [
            GRU(indim=dim[0], hiddim=dim[1], dropout=encoder_cfg["dropout"],
                has_padding=encoder_cfg["has_padding"], batch_first=True).to(device)
            for dim in encoder_cfg["dims"]
        ]
    elif encoder_cfg["type"] == "GRU_last_only":
        encoders = [
            GRU(indim=dim[0], hiddim=dim[1], dropout=encoder_cfg["dropout"],
                last_only=True).to(device)
            for dim in encoder_cfg["dims"]
        ]
    
    fusion = config["fusion_class"]().to(device)
    head_cfg = config["head_config"]
    
    if "gru_input" in head_cfg:
        head = Sequential(
            GRU(head_cfg["gru_input"], head_cfg["gru_hidden"], dropout=head_cfg["gru_dropout"],
                has_padding=False, batch_first=True, last_only=True),
            MLP(head_cfg["gru_hidden"], head_cfg["gru_hidden"], head_cfg["mlp_output"])
        ).to(device)
    else:
        head = MLP(head_cfg["mlp_input"], head_cfg["mlp_hidden"], head_cfg["mlp_output"]).to(device)
    
    return encoders, fusion, head

results = {}

print("\n" + "="*40)
print("Training all fusion methods")
print("="*40)

for method_name, config in configs.items():
    print(f"\n{'='*40}")
    print(f"Training: {method_name}")
    print(f"{'='*40}")
    
    traindata, validdata, testdata = load_data_for_method(method_name, data_path, config)
    encoders, fusion, head = build_model(method_name, config, device)
    
    param_count = sum(count_parameters(e) for e in encoders) + \
                  count_parameters(fusion) + count_parameters(head)
    print(f"Total trainable parameters: {param_count:,}")
    
    train_config = config["training_config"]
    train_start = time.time()
    
    train(
        encoders, fusion, head, traindata, validdata,
        5, task="regression", optimtype=torch.optim.AdamW,
        is_packed=train_config["is_packed"], lr=1e-3,
        save=config["save_path"], weight_decay=0.01,
        objective=torch.nn.L1Loss(), track_complexity=False
    )
    
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f}s")
    
    model = torch.load(config["save_path"], weights_only=False).to(device)
    
    print("Testing on held-out test set:")
    test(
        model=model, test_dataloaders_all=testdata, dataset="affect",
        is_packed=train_config["is_packed"], criterion=torch.nn.L1Loss(),
        task="posneg-classification", no_robust=True
    )
    
    results[method_name] = {
        "fusion_class": config["fusion_class"].__name__,
        "parameters": param_count,
        "training_time": train_time,
        "model_path": config["save_path"],
        "description": config["description"]
    }
    
    print(f"\n{method_name} complete!")

print("\n" + "="*40)
print("Re-evaluating all models...")
print("="*40 + "\n")

for method_name, config in configs.items():
    print(f"Evaluating {method_name}...")
    model = torch.load(config["save_path"], weights_only=False).to(device)
    _, _, testdata = load_data_for_method(method_name, data_path, config)
    
    train_config = config["training_config"]
    test(
        model=model, test_dataloaders_all=testdata, dataset="affect",
        is_packed=train_config["is_packed"], criterion=torch.nn.L1Loss(),
        task="posneg-classification", no_robust=True
    )
    print()

print("="*40)
print("FINAL COMPARISON TABLE")
print("="*40 + "\n")

comparison_data = []
for method_name, result in results.items():
    comparison_data.append({
        "Method": method_name,
        "Fusion Class": result["fusion_class"],
        "Parameters": f"{result['parameters']:,}",
        "Training Time (s)": f"{result['training_time']:.2f}",
        "Model Path": result["model_path"]
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))
print()

print("="*40)
print("SAVING RESULTS")
print("="*40 + "\n")

os.makedirs("results", exist_ok=True)
csv_path = "results/metrics_comparison.csv"
df_comparison.to_csv(csv_path, index=False)
print(f"Comparison table saved to: {csv_path}")

detailed_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "CMU-MOSI",
    "task": "Sentiment Analysis (Regression -> Polarity Classification)",
    "methods": results
}

json_path = "results/task_results.json"
with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")

print("\n" + "="*40)
print("INTERPRETATION GUIDE")
print("="*40 + "\n")

print("Early Fusion:")
print("  - Simplest method: concatenates raw features before processing")
print("  - Lowest parameters (no per-modality encoders)")
print("  - May miss modality-specific representations")
print()

print("Late Fusion:")
print("  - Most parameters: dedicated GRU encoder per modality")
print("  - Learns modality-specific representations separately")
print("  - May miss cross-modal interactions")
print()

print("Tensor Fusion:")
print("  - Explicit modeling of cross-modal interactions via outer product")
print("  - Highest parameter count due to outer product dimension")
print("  - Trade-off: richer feature space vs. computational cost")
print()

print("="*40)
print("TASK COMPLETE")
print("="*40)
print(f"\nAll results and models have been saved to results/")
print(f"CSV Summary: {csv_path}")
print(f"Detailed Results: {json_path}")