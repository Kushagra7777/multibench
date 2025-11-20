# GloVe Loader - TorchText Replacement

## Overview

This module provides a replacement for the deprecated `torchtext.vocab.GloVe` functionality. It was created to address compatibility issues between torchtext and newer PyTorch versions.

## Why This Replacement?

The original MultiBench codebase used `torchtext` for loading GloVe word embeddings. However:

1. **torchtext is deprecated** - The library has undergone significant API changes and older versions are incompatible with modern PyTorch
2. **Installation issues** - torchtext can be difficult to install and has binary compatibility issues
3. **Minimal usage** - MultiBench only needs GloVe embeddings, not the full torchtext library

## Features

- **Automatic fallback**: The code automatically detects if torchtext is unavailable and uses this replacement
- **Compatible API**: Provides the same interface as the old `torchtext.vocab.GloVe` for backward compatibility
- **Caching**: Downloads and caches embeddings to `~/.cache/glove/` for reuse
- **Multiple corpora**: Supports different GloVe corpora (840B, 6B, 42B, twitter.27B)

## Usage

The replacement is automatically used when torchtext is not available. No code changes are needed in most cases.

### Direct Usage

```python
from datasets.affect.glove_loader import GloVe

# Load GloVe embeddings (840B corpus, 300 dimensions)
vec = GloVe(name='840B', dim=300)

# Get embeddings for tokens
tokens = ['hello', 'world']
embeddings = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
# Returns: torch.Tensor of shape (2, 300)
```

### Via Compatibility Layer

```python
from datasets.affect import glove_loader

# Use the vocab interface (compatible with old torchtext API)
vec = glove_loader.vocab.GloVe(name='840B', dim=300)
```

## Supported GloVe Corpora

- **840B**: 840 billion tokens, 300d vectors (default, ~2GB download)
- **6B**: 6 billion tokens, 300d vectors (~800MB download)
- **42B**: 42 billion tokens, 300d vectors (~1.5GB download)
- **twitter.27B**: 27 billion tweets, 200d vectors (~1.4GB download)

## Implementation Details

The loader:
1. Downloads GloVe embeddings from Stanford NLP servers on first use
2. Caches them locally to avoid re-downloading
3. Loads embeddings into memory as a PyTorch tensor
4. Provides fast lookup for word vectors
5. Returns zero vectors for unknown words

## Files Modified

The following files have been updated to use this replacement:
- `datasets/affect/get_data.py`
- `datasets/affect/get_raw_data.py`
- `deprecated/dataloaders/affect/get_data_robust.py`

## Network Requirements

The first time you use GloVe embeddings, you'll need internet access to download them. After that, they're cached locally and no network is needed.

## Troubleshooting

**Problem**: Download fails with network error
- **Solution**: Check internet connectivity. The Stanford NLP servers may be temporarily unavailable.

**Problem**: Out of memory when loading embeddings
- **Solution**: The 840B corpus requires ~6GB RAM. Use a smaller corpus like 6B if memory is limited.

**Problem**: Wrong embedding dimensions
- **Solution**: Ensure the `dim` parameter matches the corpus (e.g., 6B supports 50, 100, 200, 300; 840B only supports 300)

## License

This code is part of MultiBench and follows the same license. GloVe embeddings are provided by Stanford NLP under their own license terms.
