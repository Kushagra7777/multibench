# MultiBench AI Coding Agent Instructions

## Project Overview

MultiBench is a standardized toolkit for multimodal deep learning research. It provides modular implementations of 20+ fusion methods across 15 datasets and 10 modalities (vision, audio, text, time-series, tabular). The architecture separates concerns into **encoders** (unimodal processing), **fusion** (combining modalities), **heads** (task prediction), and **training structures** (optimization strategies).

## Critical Working Directory Requirement

**All scripts must run from the repository root**. The codebase uses `sys.path.append(os.getcwd())` everywhere. Always execute commands from the workspace root (the directory containing this repository).

## Architecture Patterns

### Standard Module Pipeline

Every multimodal experiment follows this pattern:

```python
# 1. Add repository to path (REQUIRED at top of every script)
import sys, os
sys.path.append(os.getcwd())

# 2. Get dataloaders (train, valid, test)
from datasets.{dataset}/get_data import get_dataloader
traindata, validdata, testdata = get_dataloader(data_path)

# 3. Build components
encoders = [EncoderModal1().cuda(), EncoderModal2().cuda()]  # List order matches modality order
fusion = Concat().cuda()  # Or TensorFusion, MVAE, etc.
head = MLP(fusion_dim, hidden, output_classes).cuda()

# 4. Train using training structure
from training_structures.Supervised_Learning import train, test
train(encoders, fusion, head, traindata, validdata, epochs=20)

# 5. Test saved model
model = torch.load('best.pt').cuda()
test(model, testdata)
```

### Module Locations

- **Encoders**: `unimodals/common_models.py` - LeNet, MLP, GRU, LSTM, Transformer, ResNet, etc.
- **Fusions**: `fusions/common_fusions.py` - Concat, TensorFusion, NLGate, LowRankTensorFusion
- **Training**: `training_structures/Supervised_Learning.py` - Main training loop with automatic complexity tracking
- **Objectives**: `objective_functions/` - Custom losses beyond CrossEntropy/MSE (MFM, MVAE, CCA, contrastive)
- **Dataloaders**: `datasets/{dataset}/get_data.py` - Always named `get_dataloader()`, returns `(train, valid, test)` tuple

### Dimension Matching Rules

1. **Encoder outputs** must match **fusion input expectations**
   - `Concat()` expects list of tensors → outputs `sum(all_dims)`
   - `TensorFusion()` with 2 modals of dims [d1, d2] → outputs `(d1+1)*(d2+1)`
2. **Fusion output** must match **head input**
   - Example: `encoders=[LeNet(1,6,3), LeNet(1,6,5)]` → outputs `[48, 192]`
   - `Concat()` → 240-dim → `head=MLP(240, 100, 10)`
3. **Always pass `.cuda()` modules to training** - no CPU support in examples

## Key Conventions

### Task Types
Specify via `task` parameter (default: `"classification"`):
- `"classification"` - CrossEntropyLoss, accuracy metrics
- `"regression"` - MSELoss  
- `"multilabel"` - BCEWithLogitsLoss
- `"posneg-classification"` - Special affect dataset handling

### Custom Objectives

When using complex architectures (MFM, MVAE), provide additional modules and args:

```python
from objective_functions.objectives_for_supervised_learning import MFM_objective
from objective_functions.recon import sigmloss1dcentercrop

objective = MFM_objective(2.0, [sigmloss1dcentercrop(28,34), sigmloss1dcentercrop(112,130)], [1.0, 1.0])

train(encoders, fusion, head, traindata, validdata, 25,
      additional_optimizing_modules=decoders+intermediates,  # Modules to optimize beyond encoders/fusion/head
      objective=objective,
      objective_args_dict={'decoders': decoders, 'intermediates': intermediates})  # Extra args for objective
```

The training structure automatically passes `pred`, `truth`, `args` to objectives. Custom objectives receive:
- `args['model']` - The full MMDL wrapper
- `args['reps']` - Encoder outputs before fusion
- Plus any keys from `objective_args_dict`

### Robustness Evaluation

Robustness tests are integrated into standard `test()`. Pass `test_dataloaders_all` as dict:

```python
test_robust = {
    'clean': [test_clean_dataloader],
    'noisy_audio': [test_noise1, test_noise2, ...],  # Multiple noise levels
    'missing_vision': [test_missing_dataloader]
}
test(model, test_robust, dataset='avmnist', no_robust=False)
```

Auto-generates relative/effective robustness plots in working directory.

## Dataset Specifics

### Data Organization
- Download instructions per dataset in `datasets/{dataset}/README.md`
- Most datasets need manual download (restricted/large files)
- **MIMIC**: Restricted access - email yiweilyu@umich.edu with credentials
- **AV-MNIST**: Download tar from Google Drive, untar, pass directory path
- **Gentle Push**: Auto-downloads to `datasets/gentle_push/cache/` on first run

### Affect Datasets (MOSI, MOSEI, MUStARD, UR-FUNNY)

Special handling for variable-length sequences:

```python
# Packed sequences (default - recommended)
traindata, validdata, testdata = get_dataloader(pkl_path, data_type='mosi')
train(encoders, fusion, head, traindata, validdata, epochs, is_packed=True)

# Fixed-length padding (alternative)
traindata, validdata, testdata = get_dataloader(pkl_path, data_type='mosi', max_pad=True, max_seq_len=50)
train(encoders, fusion, head, traindata, validdata, epochs, is_packed=False)
```

## Testing & Development

### Running Tests
```bash
# From repository root only
pytest tests/  # Unit tests for modules
```

Tests use fixtures from `tests/common.py`. Mock data paths use `os.path.expanduser('~')` for portability.

### Tracking Complexity

Automatic by default (`track_complexity=True` in train). Prints:
- Training: peak memory, total params, runtime
- Testing: total params, runtime

Disable with `track_complexity=False`. Requires `memory-profiler` package.

### Model Checkpointing

- `train()` auto-saves best validation epoch to `best.pt` (or `save='custom.pt'`)
- Contains full `MMDL` wrapper with encoders, fusion, head
- Load with `model = torch.load('best.pt').cuda()`

## Common Gotchas

1. **Import errors**: Forgot `sys.path.append(os.getcwd())` at script top
2. **Dimension mismatch**: Encoder outputs don't sum to head input (check fusion output size)
3. **Wrong directory**: Running from subdirectory instead of repo root
4. **Missing `.cuda()`**: Models not moved to GPU before training
5. **Dataloader order**: Modalities must match encoder list order exactly
6. **Custom objectives**: Forgot `additional_optimizing_modules` for decoders/intermediates

## Adding New Components

### New Dataset
1. Create `datasets/{name}/get_data.py` with `get_dataloader(path, ...)` returning `(train, valid, test)`
2. Follow existing patterns (see `datasets/avmnist/get_data.py`)
3. Add example in `examples/{category}/{dataset}_{method}.py`

### New Fusion/Encoder
1. Add class to `fusions/common_fusions.py` or `unimodals/common_models.py`
2. Inherit from `nn.Module`, document input/output shapes in docstring
3. Test dimension flow: encoder outputs → fusion → head input

### New Objective
Return a closure that accepts `(pred, truth, args)`:

```python
def custom_objective(weight):
    def actualfunc(pred, truth, args):
        base_loss = torch.nn.CrossEntropyLoss()(pred, truth)
        custom_term = compute_custom(args['model'])  # Access via args dict
        return base_loss + weight * custom_term
    return actualfunc
```

## File Naming
- Example scripts: `{dataset}_{method}.py` (e.g., `mimic_baseline.py`, `avmnist_simple_late_fusion.py`)
- Saved models: `best.pt` (default), or custom via `save` parameter
- Always use `# noqa` after imports that depend on `sys.path.append(os.getcwd())`
