"""
GloVe embeddings loader - replacement for deprecated torchtext.vocab.GloVe

This module provides functionality to download and load GloVe word embeddings
without depending on torchtext, which has compatibility issues with newer
PyTorch versions.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import torch
from typing import List, Dict


class GloVe:
    """
    Load pre-trained GloVe word embeddings.
    
    This class provides an interface similar to the old torchtext.vocab.GloVe
    for backward compatibility with existing code.
    """
    
    # GloVe download URLs
    GLOVE_URLS = {
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    }
    
    def __init__(self, name='840B', dim=300, cache_dir=None):
        """
        Initialize GloVe embeddings loader.
        
        Args:
            name (str): Name of the GloVe corpus (e.g., '840B', '6B', '42B', 'twitter.27B')
            dim (int): Dimension of embeddings (300 for most, 6B has 50,100,200,300)
            cache_dir (str): Directory to cache downloaded embeddings. If None, uses ~/.cache/glove
        """
        self.name = name
        self.dim = dim
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), '.cache', 'glove')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load embeddings
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vectors: torch.Tensor = None
        
        self._load_embeddings()
    
    def _get_embedding_file_name(self) -> str:
        """Get the expected embedding file name."""
        return f"glove.{self.name}.{self.dim}d.txt"
    
    def _download_and_extract(self):
        """Download and extract GloVe embeddings if not already cached."""
        zip_file = os.path.join(self.cache_dir, f"glove.{self.name}.zip")
        embedding_file = os.path.join(self.cache_dir, self._get_embedding_file_name())
        
        # Check if already downloaded
        if os.path.exists(embedding_file):
            print(f"Found cached embeddings at {embedding_file}")
            return embedding_file
        
        # Download
        if self.name not in self.GLOVE_URLS:
            raise ValueError(f"Unknown GloVe corpus: {self.name}. Available: {list(self.GLOVE_URLS.keys())}")
        
        url = self.GLOVE_URLS[self.name]
        print(f"Downloading GloVe embeddings from {url}...")
        
        try:
            urllib.request.urlretrieve(url, zip_file)
            print(f"Download complete. Extracting...")
            
            # Extract
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            
            # Clean up zip file
            os.remove(zip_file)
            print(f"Extraction complete.")
            
            return embedding_file
        except Exception as e:
            print(f"Error downloading GloVe embeddings: {e}")
            raise
    
    def _load_embeddings(self):
        """Load embeddings from file into memory."""
        embedding_file = self._download_and_extract()
        
        print(f"Loading embeddings from {embedding_file}...")
        
        # First pass: count lines to pre-allocate array
        with open(embedding_file, 'r', encoding='utf-8') as f:
            num_words = sum(1 for _ in f)
        
        # Pre-allocate arrays
        vectors_list = []
        
        # Second pass: load embeddings
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.rstrip().split(' ')
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                
                # Verify dimension
                if len(vector) != self.dim:
                    continue
                
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                vectors_list.append(vector)
        
        # Convert to torch tensor
        self.vectors = torch.tensor(vectors_list, dtype=torch.float32)
        print(f"Loaded {len(self.word_to_idx)} word vectors of dimension {self.dim}")
    
    def get_vecs_by_tokens(self, tokens: List[str], lower_case_backup: bool = False) -> torch.Tensor:
        """
        Get embedding vectors for a list of tokens.
        
        Args:
            tokens (List[str]): List of words/tokens
            lower_case_backup (bool): If True, try lowercase version if original not found
            
        Returns:
            torch.Tensor: Tensor of shape (len(tokens), dim) containing embeddings
        """
        result_vectors = []
        
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                result_vectors.append(self.vectors[idx])
            elif lower_case_backup and token.lower() in self.word_to_idx:
                idx = self.word_to_idx[token.lower()]
                result_vectors.append(self.vectors[idx])
            else:
                # Return zero vector for unknown words
                result_vectors.append(torch.zeros(self.dim))
        
        return torch.stack(result_vectors)
    
    def __getitem__(self, word: str) -> torch.Tensor:
        """Get embedding for a single word."""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.vectors[idx]
        else:
            return torch.zeros(self.dim)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word_to_idx
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word_to_idx)


# For backward compatibility, create a vocab module structure
class VocabModule:
    """Namespace to mimic torchtext.vocab structure."""
    GloVe = GloVe


# Create module-level vocab object for compatibility
vocab = VocabModule()
