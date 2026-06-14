````markdown
# Human-AI Interaction Assignment 3
## Multimodal Sentiment Analysis using MultiBench

### Author
Kushagra Srivastav  
Student ID: 2611373

---

## Overview

This assignment compares three multimodal fusion methods on the CMU-MOSI dataset using the MultiBench framework:

- Early Fusion (ConcatEarly)
- Late Fusion (Concat)
- Tensor Fusion (TensorFusion)

The implementation is provided in:

```text
assignment_mosi.py
````

The script trains all three models, evaluates them on the test set, and generates comparison results.

---

## Requirements

Python 3.10+

Required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas tqdm scikit-learn
```

Install any additional dependencies required by MultiBench.

---

## Dataset

The script expects the CMU-MOSI dataset file at:

```text
data/affect/mosi_raw.pkl
```

---

## Running the Assignment

From the root directory of the MultiBench repository, run:

```bash
python assignment_mosi.py
```

---

## Output Files

After execution, the following files will be generated:

```text
results/
├── metrics_comparison.csv
├── assignment_results.json
└── models/
    ├── mosi_ef_r0.pt
    ├── mosi_lf_best.pt
    └── mosi_tf_best.pt
```

---

## Fusion Methods Evaluated

### Early Fusion

Concatenates raw modality features before encoding.

### Late Fusion

Processes each modality separately and combines learned representations.

### Tensor Fusion

Models higher-order interactions between modalities using tensor products.

---

## Expected Results

The reported results obtained in this assignment were:

| Method        | Accuracy |
| ------------- | -------: |
| Early Fusion  |   43.75% |
| Late Fusion   |   72.71% |
| Tensor Fusion |   43.14% |

Late Fusion achieved the highest performance on the CMU-MOSI dataset.

---

## Notes

* All experiments were performed using the MultiBench framework.
* Models were trained for 5 epochs.
* Results may vary slightly depending on hardware and software versions.

```
```
