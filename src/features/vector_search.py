"""
Vector search module for complaint similarity matching.
Uses FAISS for fast approximate nearest neighbor search.
"""

import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle


class VectorSearchIndex:
    """Build and query vector search index for complaint similarity."""
    
    def __init__(self, embedding_dim: int, use_gpu: bool = False):
        """
        Initialize vector search index.
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration (requires FAISS GPU build)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.complaint_ids = []
        self.id_to_idx = {}
        
        print(f"[INFO] Vector Search Index initialized")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  GPU acceleration: {use_gpu}")
    
    def build_index(self, embeddings: np.ndarray, complaint_ids: List[str]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Feature matrix (n_samples, embedding_dim)
            complaint_ids: List of complaint IDs corresponding to rows
        """
        n_samples = embeddings.shape[0]
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")
        
        if len(complaint_ids) != n_samples:
            raise ValueError(f"Number of IDs doesn't match embeddings: {len(complaint_ids)} != {n_samples}")
        
        print(f"[INFO] Building index from {n_samples} embeddings...")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Create index: using IVFFlat for fast approximate search
        # For smaller datasets, could use simpler index like IndexFlatL2
        if n_samples < 10000:
            # For smaller datasets, use exact search
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # For larger datasets, use IVF with quantization
            nlist = min(100, n_samples // 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store complaint IDs for lookup
        self.complaint_ids = complaint_ids.copy()
        self.id_to_idx = {cid: idx for idx, cid in enumerate(complaint_ids)}
        
        print(f"[OK] Index built with {n_samples} vectors")
    
    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors to a query embedding.
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            k: Number of nearest neighbors to return
        
        Returns:
            List of (complaint_id, distance) tuples, sorted by distance
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index first.")
        
        # Ensure correct shape and type
        query = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, k=k)
        
        # Convert distances to similarities (lower distance = higher similarity)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.complaint_ids):
                cid = self.complaint_ids[int(idx)]
                distance = float(distances[0][i])
                # Convert L2 distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + distance)
                results.append((cid, similarity))
        
        return results
    
    def search_by_complaint_id(self, complaint_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar complaints to a given complaint.
        
        Args:
            complaint_id: Query complaint ID
            k: Number of nearest neighbors (including itself)
        
        Returns:
            List of (complaint_id, similarity) tuples, sorted by similarity (descending)
        """
        if complaint_id not in self.id_to_idx:
            raise ValueError(f"Complaint ID not found: {complaint_id}")
        
        idx = self.id_to_idx[complaint_id]
        query_embedding = self.get_embedding_by_id(complaint_id)
        
        # Search and filter out the query itself
        results = self.search_by_embedding(query_embedding, k=k+1)
        results = [(cid, sim) for cid, sim in results if cid != complaint_id][:k]
        
        return results
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        Search for multiple query embeddings.
        
        Args:
            query_embeddings: Query embeddings (batch_size, embedding_dim)
            k: Number of nearest neighbors per query
        
        Returns:
            Dictionary mapping query index to list of (complaint_id, similarity) tuples
        """
        batch_size = query_embeddings.shape[0]
        query_embeddings = query_embeddings.astype('float32')
        
        distances, indices = self.index.search(query_embeddings, k=k)
        
        results = {}
        for i in range(batch_size):
            batch_results = []
            for j, idx in enumerate(indices[i]):
                if idx < len(self.complaint_ids):
                    cid = self.complaint_ids[int(idx)]
                    distance = float(distances[i][j])
                    similarity = 1.0 / (1.0 + distance)
                    batch_results.append((cid, similarity))
            results[i] = batch_results
        
        return results
    
    def get_embedding_by_id(self, complaint_id: str) -> np.ndarray:
        """Get embedding for a specific complaint (not efficient for single queries)."""
        if complaint_id not in self.id_to_idx:
            raise ValueError(f"Complaint ID not found: {complaint_id}")
        
        idx = self.id_to_idx[complaint_id]
        
        # Reconstruct embedding from index (may not be exact for IVF)
        embedding = self.index.reconstruct(int(idx))
        return embedding
    
    def save_index(self, dirpath: str):
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Build index first.")
        
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = dirpath / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save complaint IDs
        ids_path = dirpath / "complaint_ids.pkl"
        with open(ids_path, 'wb') as f:
            pickle.dump({
                'complaint_ids': self.complaint_ids,
                'id_to_idx': self.id_to_idx
            }, f)
        
        print(f"[OK] Saved index to {dirpath}")
    
    def load_index(self, dirpath: str):
        """Load index from disk."""
        dirpath = Path(dirpath)
        
        # Load FAISS index
        index_path = dirpath / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load complaint IDs
        ids_path = dirpath / "complaint_ids.pkl"
        with open(ids_path, 'rb') as f:
            data = pickle.load(f)
            self.complaint_ids = data['complaint_ids']
            self.id_to_idx = data['id_to_idx']
        
        print(f"[OK] Loaded index from {dirpath}")
        print(f"  Index contains {len(self.complaint_ids)} embeddings")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "n_complaints": len(self.complaint_ids),
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__ if self.index else None,
            "index_trained": self.index.is_trained if self.index else False,
        }


class SimilarityMatcher:
    """High-level interface for finding similar complaints."""
    
    def __init__(self, text_embeddings: np.ndarray, complaint_ids: List[str]):
        """
        Initialize similarity matcher.
        
        Args:
            text_embeddings: Text embedding matrix (n_samples, embedding_dim)
            complaint_ids: List of complaint IDs
        """
        self.embedding_dim = text_embeddings.shape[1]
        self.index = VectorSearchIndex(embedding_dim=self.embedding_dim)
        self.index.build_index(text_embeddings, complaint_ids)
        
        print(f"[OK] Similarity Matcher initialized")
    
    def find_similar(self, complaint_id: str, k: int = 5) -> List[Dict]:
        """
        Find similar complaints.
        
        Args:
            complaint_id: Query complaint ID
            k: Number of similar complaints to return
        
        Returns:
            List of dicts with 'complaint_id' and 'similarity' keys
        """
        results = self.index.search_by_complaint_id(complaint_id, k=k)
        return [{'complaint_id': cid, 'similarity': float(sim)} for cid, sim in results]
    
    def find_similar_batch(self, complaint_ids: List[str], k: int = 5) -> Dict[str, List[Dict]]:
        """Find similar complaints for multiple query complaints."""
        results = {}
        for cid in complaint_ids:
            results[cid] = self.find_similar(cid, k=k)
        return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vector Search Demo")
    print("=" * 60)
    
    # Example with dummy embeddings
    n_samples = 100
    embedding_dim = 384
    
    # Create dummy embeddings
    embeddings = np.random.rand(n_samples, embedding_dim).astype('float32')
    complaint_ids = [f"CMP_{i:06d}" for i in range(n_samples)]
    
    # Build index
    index = VectorSearchIndex(embedding_dim=embedding_dim)
    index.build_index(embeddings, complaint_ids)
    
    # Search
    query_id = complaint_ids[0]
    print(f"\nSearching for complaints similar to {query_id}...")
    results = index.search_by_complaint_id(query_id, k=5)
    
    for i, (cid, sim) in enumerate(results, 1):
        print(f"  {i}. {cid}: {sim:.4f}")
    
    # Get stats
    print("\nIndex statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
