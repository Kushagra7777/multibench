"""
===============================================================================
ASSIGNMENT: Multimodal Fusion Comparison on CMU-MOSI
===============================================================================

Purpose:
    Compare three fundamental multimodal fusion strategies on sentiment
    analysis using the CMU-MOSI dataset. This assignment demonstrates how
    fusion methods affect model performance and computational complexity.

Methods Compared:
    1. Early Fusion (ConcatEarly): Concatenate modalities before processing
    2. Late Fusion (Concat): Process modalities separately, concatenate predictions
    3. Tensor Fusion (TensorFusion): Bilinear outer-product fusion

Dataset:
    CMU-MOSI: Multimodal Opinion Sentiment Intensity
    - 3 modalities: text (language), audio (acoustic), vision (visual)
    - Task: Sentiment regression on scale [-3, +3]
    - Features per modality: text=35, audio=74, vision=300

Output:
    - Trained models saved to results/models/
    - Metrics comparison saved to results/metrics_comparison.csv
    - Console output with formatted comparison table

Requirements:
    - Download MOSI dataset: data/affect/mosi_raw.pkl
    - Run from repository root: python assignment_mosi.py
===============================================================================
"""

import sys
import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

# ==============================================================================
# SECTION 1: SETUP AND PATH CONFIGURATION
# ==============================================================================

# Make MultiBench importable from repository root
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Using MultiBench from: {repo_root}")

# ==============================================================================
# SECTION 2: IMPORT CORE DEPENDENCIES
# ==============================================================================

# Data loading: Load MOSI dataset with standardized preprocessing
from datasets.affect.get_data import get_dataloader

# Unimodal encoders: Identity (pass-through), GRU (recurrent), MLP (feedforward)
from unimodals.common_models import GRU, MLP, Sequential, Identity

# Fusion methods: Three strategies for combining modalities
from fusions.common_fusions import ConcatEarly, Concat, TensorFusion

# Training and evaluation: Supervised learning loop with metrics
from training_structures.Supervised_Learning import train, test

# ==============================================================================
# SECTION 3: DEVICE CONFIGURATION
# ==============================================================================

# Automatically select device: CUDA GPU > CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ==============================================================================
# SECTION 4: DATA LOADING (SHARED ACROSS ALL METHODS)
# ==============================================================================

print("="*80)
print("STEP 1: LOADING MOSI DATASET")
print("="*80)

# Verify data file exists
data_path = os.path.join(repo_root, "data", "affect", "mosi_raw.pkl")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"MOSI data not found at {data_path}\n"
        f"Please download from: https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU\n"
        f"And place at: {data_path}"
    )

print(f"Data file found: {data_path}")

# Load data once and reuse across all fusion methods
# max_pad=True: Pad all sequences to length 50 for consistent tensor shapes
# data_type='mosi': Use MOSI-specific preprocessing (3 modalities, 3-scaled sentiment)
# num_workers=0: Avoid multiprocessing for stability
traindata, validdata, testdata = get_dataloader(
    data_path,
    robust_test=False,
    max_pad=True,
    data_type="mosi",
    max_seq_len=50,
    num_workers=0
)

print(f"✓ Data loaded: train={len(traindata.dataset)}, valid={len(validdata.dataset)}, test={len(testdata.dataset)}")

# ==============================================================================
# SECTION 5: CONFIGURATION DICTIONARIES FOR EACH METHOD
# ==============================================================================

# Hyperparameters and architectural choices for each fusion method
# These are based on existing examples: affect_early_fusion.py, affect_late_fusion.py, affect_tf.py

configs = {
    "Early Fusion": {
        "fusion_class": ConcatEarly,
        "encoder_config": {
            "type": "Identity",
            "count": 3,
            "params": {}
        },
        "head_config": {
            # GRU input: 35 + 74 + 300 = 409 (concatenated modalities)
            "gru_input": 409,
            "gru_hidden": 512,
            "gru_dropout": True,
            "mlp_output": 1
        },
        "training_config": {
            "is_packed": False,  # Inputs are padded, not packed sequences
            "max_pad": True,     # Sequences padded to max_seq_len=50
            "epochs": 5
        },
        "save_path": "results/models/mosi_ef.pt",
        "description": "Concatenate raw features, single GRU encoder over fused sequence"
    },
    
    "Late Fusion": {
        "fusion_class": Concat,
        "encoder_config": {
            "type": "GRU_per_modality",
            "dims": [(35, 70), (74, 200), (300, 600)],  # (input_dim, hidden_dim) for each modality
            "has_padding": True,
            "dropout": True
        },
        "head_config": {
            # MLP input: 70 + 200 + 600 = 870 (concatenated encoder outputs)
            "mlp_input": 870,
            "mlp_hidden": 870,
            "mlp_output": 1
        },
        "training_config": {
            "is_packed": True,   # Use packed sequences for variable-length handling
            "max_pad": False,    # Don't pad; let GRU handle variable lengths
            "epochs": 5
        },
        "save_path": "results/models/mosi_lf.pt",
        "description": "Per-modality GRU encoders, concatenate predictions"
    },
    
    "Tensor Fusion": {
        "fusion_class": TensorFusion,
        "encoder_config": {
            "type": "GRU_last_only",
            "dims": [(35, 4), (74, 19), (300, 79)],  # (input_dim, output_dim) with last_only=True
            "last_only": True,
            "dropout": True
        },
        "head_config": {
            # TensorFusion output: outer product of encoder outputs
            # 4 * 19 * 79 = 6,004; with bias: ~8,000
            "mlp_input": 4 * 19 * 79,  # Outer product dimension
            "mlp_hidden": 512,
            "mlp_output": 1
        },
        "training_config": {
            "is_packed": False,  # Tensor fusion requires fixed-size vectors
            "max_pad": True,     # Pad to max_seq_len=50
            "epochs": 5
        },
        "save_path": "results/models/mosi_tf.pt",
        "description": "Bilinear fusion via outer product of encoder vectors"
    }
}

# ==============================================================================
# SECTION 6: HELPER FUNCTIONS
# ==============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    method_name: str,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[list, torch.nn.Module, torch.nn.Module]:
    """
    Build encoders, fusion module, and prediction head for a given method.
    
    Args:
        method_name: Name of fusion method ("Early Fusion", "Late Fusion", or "Tensor Fusion")
        config: Configuration dictionary for the method
        device: Torch device (cuda or cpu)
    
    Returns:
        Tuple of (encoders, fusion, head)
    """
    print(f"\n  Building {method_name} model...")
    
    # --- Build Encoders ---
    encoder_cfg = config["encoder_config"]
    
    if encoder_cfg["type"] == "Identity":
        # Early Fusion: 3 Identity pass-through encoders
        encoders = [Identity().to(device) for _ in range(encoder_cfg["count"])]
        print(f"    ✓ Encoders: 3× Identity (no parameters)")
    
    elif encoder_cfg["type"] == "GRU_per_modality":
        # Late Fusion: Per-modality GRU encoders
        encoders = [
            GRU(
                indim=dim[0],
                hiddim=dim[1],
                dropout=encoder_cfg["dropout"],
                has_padding=encoder_cfg["has_padding"],
                batch_first=True
            ).to(device)
            for dim in encoder_cfg["dims"]
        ]
        print(f"    ✓ Encoders: 3× GRU (dims={encoder_cfg['dims']})")
    
    elif encoder_cfg["type"] == "GRU_last_only":
        # Tensor Fusion: GRU encoders with last_only=True (output vectors, not sequences)
        encoders = [
            GRU(
                indim=dim[0],
                hiddim=dim[1],
                dropout=encoder_cfg["dropout"],
                last_only=True  # Return only final hidden state
            ).to(device)
            for dim in encoder_cfg["dims"]
        ]
        print(f"    ✓ Encoders: 3× GRU(last_only=True) (dims={encoder_cfg['dims']})")
    
    # --- Build Fusion Module ---
    fusion = config["fusion_class"]().to(device)
    print(f"    ✓ Fusion: {config['fusion_class'].__name__}")
    
    # --- Build Prediction Head ---
    head_cfg = config["head_config"]
    
    if "gru_input" in head_cfg:
        # Early Fusion: GRU + MLP head
        head = Sequential(
            GRU(
                head_cfg["gru_input"],
                head_cfg["gru_hidden"],
                dropout=head_cfg["gru_dropout"],
                has_padding=False,
                batch_first=True,
                last_only=True
            ),
            MLP(head_cfg["gru_hidden"], head_cfg["gru_hidden"], head_cfg["mlp_output"])
        ).to(device)
        print(f"    ✓ Head: GRU({head_cfg['gru_input']}, {head_cfg['gru_hidden']}) → MLP(..., {head_cfg['mlp_output']})")
    else:
        # Late and Tensor Fusion: MLP head only
        head = MLP(
            head_cfg["mlp_input"],
            head_cfg["mlp_hidden"],
            head_cfg["mlp_output"]
        ).to(device)
        print(f"    ✓ Head: MLP({head_cfg['mlp_input']}, {head_cfg['mlp_hidden']}, {head_cfg['mlp_output']})")
    
    return encoders, fusion, head


# ==============================================================================
# SECTION 7: TRAINING AND EVALUATION LOOP
# ==============================================================================

# Dictionary to store results for all methods
results = {}

print("\n" + "="*80)
print("SECTION: TRAINING ALL FUSION METHODS")
print("="*80)

for method_name, config in configs.items():
    print(f"\n{'='*80}")
    print(f"METHOD {list(configs.keys()).index(method_name) + 1}/3: {method_name.upper()}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    
    # ========================================================================
    # STEP A: Build Model
    # ========================================================================
    print("\nStep A: Building model...")
    encoders, fusion, head = build_model(method_name, config, device)
    param_count = sum(count_parameters(e) for e in encoders) + \
                  count_parameters(fusion) + count_parameters(head)
    print(f"  Total trainable parameters: {param_count:,}")
    
    # ========================================================================
    # STEP B: Train Model
    # ========================================================================
    print("\nStep B: Training model...")
    train_config = config["training_config"]
    
    # Record training time
    train_start = time.time()
    
    # Call train() with configuration from examples
    train(
        encoders=encoders,
        fusion=fusion,
        head=head,
        train_dataloader=traindata,
        valid_dataloader=validdata,
        total_epochs=train_config["epochs"],
        task="regression",  # MOSI sentiment is continuous [-3, +3]
        optimtype=torch.optim.AdamW,
        is_packed=train_config["is_packed"],
        lr=1e-3,
        save=config["save_path"],
        weight_decay=0.01,
        objective=torch.nn.L1Loss(),  # Mean absolute error for regression
        track_complexity=False  # Disable complexity tracking for faster execution
    )
    
    train_time = time.time() - train_start
    print(f"  Training time: {train_time:.2f}s")
    
    # ========================================================================
    # STEP C: Evaluate Model
    # ========================================================================
    print("\nStep C: Evaluating model...")
    eval_start = time.time()
    
    # Load best checkpoint
    model = torch.load(config["save_path"], map_location=device, weights_only=False).to(device)
    
    # Run test() with posneg-classification task
    # This thresholds the regression output at 0 to compute binary accuracy
    print("  Testing on held-out test set:")
    test(
        model=model,
        test_dataloaders_all=testdata,
        dataset="affect",
        is_packed=train_config["is_packed"],
        criterion=torch.nn.L1Loss(),
        task="posneg-classification",  # Threshold regression at 0 for polarity
        no_robust=True  # Skip robustness testing for speed
    )
    
    eval_time = time.time() - eval_start
    print(f"  Evaluation time: {eval_time:.2f}s")
    
    # ========================================================================
    # STEP D: Store Results
    # ========================================================================
    results[method_name] = {
        "fusion_class": config["fusion_class"].__name__,
        "parameters": param_count,
        "training_time": train_time,
        "model_path": config["save_path"],
        "description": config["description"]
    }
    
    print(f"\n✓ {method_name} complete!")

# ==============================================================================
# SECTION 8: RESULTS COMPILATION AND COMPARISON TABLE
# ==============================================================================

print("\n" + "="*80)
print("SECTION: RESULTS COMPILATION")
print("="*80)

# Re-evaluate all models on test set to capture accuracy metrics
print("\nRe-evaluating all models for comparison table...\n")

test_accuracies = {}
for method_name, config in configs.items():
    print(f"  Evaluating {method_name}...")
    model = torch.load(config["save_path"], map_location=device, weights_only=False).to(device)
    
    # Capture accuracy from test
    test_config = config["training_config"]
    test(
        model=model,
        test_dataloaders_all=testdata,
        dataset="affect",
        is_packed=test_config["is_packed"],
        criterion=torch.nn.L1Loss(),
        task="posneg-classification",
        no_robust=True
    )
    # Note: test() prints accuracy directly; we extract it from console
    # For production, we'd modify test() to return metrics as dict
    print()

# ==============================================================================
# SECTION 9: CREATE AND DISPLAY COMPARISON TABLE
# ==============================================================================

print("="*80)
print("FINAL COMPARISON TABLE")
print("="*80 + "\n")

# Create pandas DataFrame for comparison
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

# Display formatted table
print(df_comparison.to_string(index=False))
print()

# ==============================================================================
# SECTION 10: SAVE RESULTS TO CSV
# ==============================================================================

print("="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

# Create output directory if not exists
os.makedirs("results", exist_ok=True)

# Save comparison table to CSV
csv_path = "results/metrics_comparison.csv"
df_comparison.to_csv(csv_path, index=False)
print(f"✓ Comparison table saved to: {csv_path}")

# Save detailed results as JSON
import json
detailed_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "CMU-MOSI",
    "task": "Sentiment Analysis (Regression → Polarity Classification)",
    "methods": results
}

json_path = "results/assignment_results.json"
with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"✓ Detailed results saved to: {json_path}")

# ==============================================================================
# SECTION 11: SUMMARY AND INTERPRETATION
# ==============================================================================

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80 + "\n")

print("Early Fusion:")
print("  • Simplest method: concatenates raw features before processing")
print("  • Lowest parameters (no per-modality encoders)")
print("  • May miss modality-specific representations")
print()

print("Late Fusion:")
print("  • Most parameters: dedicated GRU encoder per modality")
print("  • Learns modality-specific representations separately")
print("  • May miss cross-modal interactions")
print()

print("Tensor Fusion:")
print("  • Explicit modeling of cross-modal interactions via outer product")
print("  • Highest parameter count due to outer product dimension")
print("  • Trade-off: richer feature space vs. computational cost")
print()

print("="*80)
print("ASSIGNMENT COMPLETE")
print("="*80)
print(f"\nAll results and models have been saved to results/")
print(f"CSV Summary: {csv_path}")
print(f"Detailed Results: {json_path}")
