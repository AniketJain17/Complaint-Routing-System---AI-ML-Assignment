"""
Officer routing model - matches complaints to optimal officers.
Uses semantic similarity between complaint text and officer expertise.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class OfficerRoutingModel:
    """Route complaints to optimal officers based on semantic similarity."""
    
    def __init__(self, officers: List, complaint_text_dim: int = 768):
        """
        Initialize routing model.
        
        Args:
            officers: List of officer objects
            complaint_text_dim: Dimension of text embeddings
        """
        self.officers = {o.officer_id: o for o in officers}
        self.officer_ids = [o.officer_id for o in officers]
        self.text_dim = complaint_text_dim
        
        # Officer expertise embeddings (to be computed)
        self.officer_embeddings = {}
        self.officer_names = {o.officer_id: o.name for o in officers}
        
        print(f"[INFO] Officer Routing Model initialized with {len(officers)} officers")
    
    def train(self, text_embeddings: np.ndarray, assigned_officers: List[str]):
        """
        Train routing model by computing officer expertise embeddings.
        Averages embeddings of complaints assigned to each officer.
        
        Args:
            text_embeddings: Text embeddings (n_samples, text_dim)
            assigned_officers: Officer ID for each complaint
        """
        print(f"[INFO] Training Officer Routing Model...")
        
        # Compute average embedding for each officer
        officer_embeddings = {}
        officer_complaint_counts = {}
        
        for i, officer_id in enumerate(assigned_officers):
            if officer_id not in officer_embeddings:
                officer_embeddings[officer_id] = np.zeros(self.text_dim)
                officer_complaint_counts[officer_id] = 0
            
            officer_embeddings[officer_id] += text_embeddings[i]
            officer_complaint_counts[officer_id] += 1
        
        # Average embeddings
        for officer_id in officer_embeddings:
            count = officer_complaint_counts[officer_id]
            officer_embeddings[officer_id] /= count
        
        # Store officer embeddings
        self.officer_embeddings = officer_embeddings
        
        # Log trained officers
        print(f"[OK] Trained embeddings for {len(officer_embeddings)} officers")
        for officer_id, count in officer_complaint_counts.items():
            print(f"   {officer_id} ({self.officer_names.get(officer_id, '?')}): {count} complaints")
    
    def predict(self, complaint_embedding: np.ndarray, k: int = 5, 
                include_workload: bool = True, include_rating: bool = True) -> List[Dict]:
        """
        Route complaint to top-k officers.
        
        Args:
            complaint_embedding: Text embedding of complaint
            k: Number of top officers to return
            include_workload: Whether to factor in officer workload
            include_rating: Whether to factor in officer rating
        
        Returns:
            List of dicts with officer_id, similarity, workload_score, final_score
        """
        if not self.officer_embeddings:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Compute similarities
        similarities = {}
        for officer_id, officer_emb in self.officer_embeddings.items():
            # Cosine similarity
            dot_product = np.dot(complaint_embedding, officer_emb)
            norm1 = np.linalg.norm(complaint_embedding)
            norm2 = np.linalg.norm(officer_emb)
            
            similarity = dot_product / (norm1 * norm2 + 1e-8)
            similarities[officer_id] = similarity
        
        # Build results
        results = []
        for officer_id, similarity in similarities.items():
            officer = self.officers.get(officer_id)
            if not officer:
                continue
            
            # Normalize similarity to 0-1
            sim_score = (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Workload score (lower workload = higher score)
            workload_score = 1.0 - (officer.current_workload / (officer.max_capacity + 1e-8))
            
            # Rating score (normalize to 0-1)
            rating_score = officer.rating / 5.0
            
            # Combine scores
            final_score = sim_score
            if include_workload:
                final_score = final_score * 0.7 + workload_score * 0.2
            if include_rating:
                final_score = final_score * 0.9 + rating_score * 0.1
            
            results.append({
                'officer_id': officer_id,
                'officer_name': officer.name,
                'similarity': float(sim_score),
                'workload_score': float(workload_score),
                'rating': officer.rating,
                'available': officer.is_available(),
                'final_score': float(final_score),
            })
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:k]
    
    def predict_batch(self, complaint_embeddings: np.ndarray, k: int = 5) -> List[List[Dict]]:
        """
        Route multiple complaints.
        
        Args:
            complaint_embeddings: Text embeddings (n_samples, text_dim)
            k: Number of top officers per complaint
        
        Returns:
            List of lists of recommendations
        """
        results = []
        for embedding in complaint_embeddings:
            pred = self.predict(embedding, k=k)
            results.append(pred)
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'officer_embeddings': self.officer_embeddings,
            'officer_ids': self.officer_ids,
            'officer_names': self.officer_names,
            'text_dim': self.text_dim,
        }
        
        joblib.dump(model_data, filepath)
        print(f"[OK] Saved routing model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.officer_embeddings = model_data['officer_embeddings']
        self.officer_ids = model_data['officer_ids']
        self.officer_names = model_data['officer_names']
        self.text_dim = model_data['text_dim']
        
        print(f"[OK] Loaded routing model from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': 'Semantic Similarity Matching',
            'n_officers': len(self.officer_ids),
            'text_embedding_dim': self.text_dim,
            'trained_officers': len(self.officer_embeddings),
            'training_status': 'trained' if self.officer_embeddings else 'untrained',
        }


class OfficerExpertiseMatcher:
    """Match complaints to officers based on expertise areas."""
    
    def __init__(self, officers: List):
        """
        Initialize expertise matcher.
        
        Args:
            officers: List of officer objects
        """
        self.officers = {o.officer_id: o for o in officers}
        self.expertise_to_officers = self._build_expertise_index(officers)
        
        print(f"[INFO] Expertise Matcher initialized")
    
    def _build_expertise_index(self, officers: List) -> Dict[str, List[str]]:
        """Build index mapping expertise areas to officers."""
        index = {}
        for officer in officers:
            for expertise in officer.expertise_areas:
                if expertise not in index:
                    index[expertise] = []
                index[expertise].append(officer.officer_id)
        return index
    
    def find_experts(self, category: str, k: int = 5) -> List[Dict]:
        """
        Find officers with expertise in a category.
        
        Args:
            category: Complaint category
            k: Number of experts to return
        
        Returns:
            List of expert officer dicts
        """
        if category not in self.expertise_to_officers:
            # Return top-rated officers if no expertise match
            officers_list = list(self.officers.values())
            officers_list.sort(key=lambda o: o.rating, reverse=True)
            officers_list = officers_list[:k]
        else:
            officer_ids = self.expertise_to_officers[category]
            officers_list = [self.officers[oid] for oid in officer_ids]
            officers_list.sort(key=lambda o: o.rating, reverse=True)
            officers_list = officers_list[:k]
        
        return [{
            'officer_id': o.officer_id,
            'officer_name': o.name,
            'expertise': o.expertise_areas,
            'rating': o.rating,
            'available': o.is_available(),
        } for o in officers_list]
    
    def get_expertise_stats(self) -> Dict:
        """Get statistics about expertise coverage."""
        return {
            'total_expertise_areas': len(self.expertise_to_officers),
            'expertise_areas': {
                area: len(officers)
                for area, officers in self.expertise_to_officers.items()
            }
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Officer Routing Model Demo")
    print("=" * 60)
    
    # Dummy data
    from collections import namedtuple
    Officer = namedtuple('Officer', ['officer_id', 'name', 'expertise_areas', 
                                     'languages', 'current_workload', 'max_capacity',
                                     'avg_resolution_time_days', 'rating'])
    
    def is_available(self):
        return self.current_workload < self.max_capacity
    
    Officer.is_available = is_available
    
    officers_data = [
        Officer('OFC_001', 'Alice', ['billing', 'payment'], ['en', 'es'], 3, 10, 3.5, 4.8),
        Officer('OFC_002', 'Bob', ['technical', 'account'], ['en', 'fr'], 2, 10, 4.0, 4.5),
        Officer('OFC_003', 'Carol', ['service', 'refunds'], ['en', 'de'], 5, 10, 2.8, 4.9),
    ]
    
    # Create model
    model = OfficerRoutingModel(officers_data)
    
    # Dummy embeddings
    n_samples = 20
    text_embeddings = np.random.randn(n_samples, 768)
    assigned_officers = ['OFC_001'] * 8 + ['OFC_002'] * 7 + ['OFC_003'] * 5
    
    # Train
    model.train(text_embeddings, assigned_officers)
    
    # Predict
    query_emb = np.random.randn(768)
    predictions = model.predict(query_emb, k=3)
    
    print("\nPredictions for sample complaint:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['officer_name']} ({pred['officer_id']})")
        print(f"     Similarity: {pred['similarity']:.4f}, Score: {pred['final_score']:.4f}")
    
    print("\n" + "=" * 60)
