"""
Text feature extraction for multilingual complaints.
Uses sentence-transformers for multilingual embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import warnings

# Suppress transformer warnings
warnings.filterwarnings('ignore')


class TextFeatureExtractor:
    """Extract text embeddings from complaint texts using multilingual models."""
    
    # Model options for different embedding dimensions
    MODEL_OPTIONS = {
        "small": "distiluse-base-multilingual-cased-v2",      # 512 dims, faster
        "medium": "paraphrase-multilingual-mpnet-base-v2",     # 768 dims, balanced
        "large": "paraphrase-multilingual-MiniLM-L12-v2",      # 384 dims, quality
    }
    
    def __init__(self, model_name: str = "medium", cache_dir: Optional[str] = None):
        """
        Initialize text feature extractor.
        
        Args:
            model_name: Model size ('small', 'medium', 'large')
            cache_dir: Directory to cache model and embeddings
        """
        self.model_name = model_name
        self.model_path = self.MODEL_OPTIONS.get(model_name, model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model (will download if not cached)
        print(f"[INFO] Loading text model: {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[OK] Text model loaded. Embedding dimension: {self.embedding_dim}")
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from list of texts.
        
        Args:
            texts: List of complaint texts
            batch_size: Batch size for processing
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        print(f"[INFO] Extracting embeddings from {len(texts)} texts...")
        
        # Normalize texts (remove extra whitespace)
        normalized_texts = [text.strip() for text in texts]
        
        # Encode texts
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"[OK] Extracted {embeddings.shape[0]} embeddings of shape {embeddings.shape}")
        return embeddings
    
    def extract_single_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding from single text.
        
        Args:
            text: Single complaint text
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        
        Returns:
            Cosine similarity score (0-1)
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity)
    
    def compute_similarities(self, emb1: np.ndarray, emb2s: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between one embedding and multiple embeddings.
        
        Args:
            emb1: Single embedding of shape (embedding_dim,)
            emb2s: Multiple embeddings of shape (n, embedding_dim)
        
        Returns:
            Similarity scores of shape (n,)
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2s_norm = emb2s / (np.linalg.norm(emb2s, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(emb2s_norm, emb1_norm)
        return similarities
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[OK] Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"[OK] Loaded embeddings from {filepath}")
        return embeddings
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "embedding_dimension": self.embedding_dim,
            "framework": "sentence-transformers",
            "multilingual": True,
        }


class TextFeatures:
    """Container for text features and metadata."""
    
    def __init__(self, complaint_ids: List[str], embeddings: np.ndarray):
        """
        Initialize text features container.
        
        Args:
            complaint_ids: List of complaint IDs
            embeddings: Embedding matrix (n_samples, embedding_dim)
        """
        self.complaint_ids = complaint_ids
        self.embeddings = embeddings
        self.n_samples = len(complaint_ids)
        self.embedding_dim = embeddings.shape[1]
        
        # Create index mapping
        self.id_to_idx = {cid: idx for idx, cid in enumerate(complaint_ids)}
    
    def get_embedding(self, complaint_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific complaint."""
        if complaint_id in self.id_to_idx:
            idx = self.id_to_idx[complaint_id]
            return self.embeddings[idx]
        return None
    
    def get_embeddings_batch(self, complaint_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple complaints."""
        indices = [self.id_to_idx[cid] for cid in complaint_ids if cid in self.id_to_idx]
        return self.embeddings[indices]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "complaint_ids": self.complaint_ids,
            "embeddings_shape": self.embeddings.shape,
            "embedding_dim": self.embedding_dim,
            "n_samples": self.n_samples,
        }


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 60)
    print("Text Feature Extraction Demo")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "I was charged twice for my subscription.",
        "El producto que recibí es defectuoso.",  # Spanish
        "Je veux un remboursement immédiatement.",  # French
    ]
    
    # Extract features
    extractor = TextFeatureExtractor(model_name="medium")
    
    print("\nExtracting text embeddings...")
    embeddings = extractor.extract_embeddings(texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {extractor.embedding_dim}")
    
    # Test similarity
    print("\nComputing similarities...")
    sim_01 = extractor.compute_similarity(embeddings[0], embeddings[1])
    sim_02 = extractor.compute_similarity(embeddings[0], embeddings[2])
    print(f"Similarity(en, es): {sim_01:.4f}")
    print(f"Similarity(en, fr): {sim_02:.4f}")
    
    print("\n" + "=" * 60)
