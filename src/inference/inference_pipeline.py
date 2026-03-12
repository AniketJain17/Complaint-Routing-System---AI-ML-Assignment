"""
Unified inference pipeline for complaint routing system.
Loads trained models and makes predictions for new complaints.
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
import sys
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data.schemas import Complaint, PredictionResult
from features.text_features import TextFeatureExtractor
from features.audio_features import AudioFeatureExtractor
from features.video_features import VideoFeatureExtractor


class ComplaintRoutingInference:
    """
    Unified inference system that loads all trained models and makes predictions.
    
    Usage:
        inference = ComplaintRoutingInference(models_dir='data/models')
        result = inference.predict(complaint)
        print(result.to_dict())
    """
    
    def __init__(self, models_dir: str = 'data/models'):
        """
        Initialize inference pipeline by loading all trained models.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        
        # Initialize feature extractors
        self.text_extractor = TextFeatureExtractor(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.audio_extractor = AudioFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor()
        
        # Load trained models
        self._load_models()
        print(f"[OK] Inference pipeline initialized from {self.models_dir}")
    
    def _load_models(self):
        """Load all trained models and scalers from disk."""
        # Load officer routing model
        routing_path = self.models_dir / 'routing_model.pkl'
        self.routing_model = joblib.load(routing_path)
        print(f"[OK] Officer routing model loaded from {routing_path.name}")
        
        # Load priority classifier
        priority_path = self.models_dir / 'priority_model.pkl'
        self.priority_model = joblib.load(priority_path)
        print(f"[OK] Priority classifier loaded from {priority_path.name}")
        
        # Load ETA regressor
        eta_path = self.models_dir / 'eta_model.pkl'
        self.eta_model = joblib.load(eta_path)
        print(f"[OK] ETA regressor loaded from {eta_path.name}")
        
        # Load feature scalers
        scalers_dir = self.models_dir / 'scalers'
        self.text_scaler = joblib.load(scalers_dir / 'text_scaler.pkl')
        print(f"[OK] Text scaler loaded")
        
        # Load similarity index (FAISS)
        self.similarity_index = None
        try:
            import faiss
            index_path = self.models_dir / 'similarity_index' / 'faiss.index'
            self.similarity_index = faiss.read_index(str(index_path))
            
            # Load complaint ID mapping
            ids_path = self.models_dir / 'similarity_index' / 'complaint_ids.pkl'
            self.similarity_ids = joblib.load(ids_path)
            print(f"[OK] Similarity index loaded ({len(self.similarity_ids)} embeddings)")
        except Exception as e:
            print(f"[WARNING] Similarity index not loaded: {e}")
            self.similarity_index = None
    
    def predict(self, complaint: Dict) -> PredictionResult:
        """
        Make predictions for a single complaint.
        
        Args:
            complaint: Dictionary with keys 'text', 'language', optional 'audio_path', 'video_path'
        
        Returns:
            PredictionResult with assigned officers, priority, ETA, and similar complaints
        """
        # Extract text features
        text = complaint.get('text', '')
        language = complaint.get('language', 'en')
        
        text_embedding = self.text_extractor.extract_embeddings([text])[0]  # Shape: (768,)
        text_features = self.text_scaler.transform([text_embedding])[0]  # Normalized
        
        # Predict officer routing
        assigned_officers_tuples = self._predict_officer_routing(text_features, top_k=3)
        assigned_officers = [{'officer_id': oid, 'score': float(score)} for oid, score in assigned_officers_tuples]
        
        # Predict priority
        predicted_priority = self._predict_priority(text_features)
        
        # Predict ETA
        predicted_eta_days = self._predict_eta(text_features)
        
        # Find similar complaints
        similar_ids = self._find_similar_complaints(text_embedding, top_k=5)
        similar_complaints = [{'complaint_id': cid, 'similarity_score': 0.0} for cid in similar_ids]
        
        # Create result
        result = PredictionResult(
            complaint_id=complaint.get('id', 'unknown'),
            assigned_officers=assigned_officers,
            predicted_priority=predicted_priority,
            predicted_eta_days=predicted_eta_days,
            similar_complaints=similar_complaints,
            priority_confidence=float(self._get_priority_confidence()),
            eta_confidence=float(self._get_eta_confidence())
        )
        
        return result
    
    def _predict_officer_routing(self, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict which officers are best to handle the complaint.
        
        Args:
            features: Normalized text features
            top_k: Number of top officers to return
        
        Returns:
            List of (officer_id, confidence_score) tuples
        """
        routing_model = self.routing_model
        officer_ids = routing_model['officer_ids']
        officer_embeddings = routing_model['officer_embeddings']  # Dict: {officer_id: embedding}
        
        # Compute similarity between complaint and each officer's expertise
        similarities = []
        for oid in officer_ids:
            officer_emb = officer_embeddings[oid]  # Get embedding for this officer
            # Cosine similarity
            sim = np.dot(features, officer_emb) / (np.linalg.norm(features) * np.linalg.norm(officer_emb) + 1e-8)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top-k officers
        top_indices = np.argsort(-similarities)[:top_k]
        result = [(officer_ids[idx], float(similarities[idx])) for idx in top_indices]
        
        return result
    
    def _predict_priority(self, features: np.ndarray) -> str:
        """
        Predict complaint priority (LOW, MEDIUM, HIGH).
        
        Args:
            features: Normalized text features
        
        Returns:
            Predicted priority level
        """
        # Extract actual model if wrapped in dict
        if isinstance(self.priority_model, dict):
            model = self.priority_model['model']
            label_encoder = self.priority_model.get('label_encoder')
        else:
            model = self.priority_model
            label_encoder = None
        
        # Get prediction
        prediction = model.predict([features])[0]
        
        # Map to label using encoder
        if label_encoder:
            priority_label = label_encoder.inverse_transform([prediction])[0]
        else:
            priority_label = prediction
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            self._last_priority_proba = np.max(model.predict_proba([features]))
        else:
            self._last_priority_proba = 0.5
        
        return priority_label
    
    def _predict_eta(self, features: np.ndarray) -> int:
        """
        Predict estimated time to resolution (in days).
        
        Args:
            features: Normalized text features
        
        Returns:
            Predicted ETA in days (minimum 1)
        """
        # Extract actual model if wrapped in dict
        if isinstance(self.eta_model, dict):
            model = self.eta_model['model']
        else:
            model = self.eta_model
        
        eta_days = model.predict([features])[0]
        self._last_eta_pred = eta_days
        
        # Ensure minimum 1 day
        return max(1, int(round(eta_days)))
    
    def _find_similar_complaints(self, embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """
        Find similar complaints using FAISS similarity search.
        
        Args:
            embedding: Text embedding of complaint
            top_k: Number of similar complaints to return
        
        Returns:
            List of similar complaint IDs
        """
        if self.similarity_index is None:
            return []
        
        try:
            # Reshape for FAISS (1, 768)
            query_embedding = embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            distances, indices = self.similarity_index.search(query_embedding, k=top_k + 1)
            
            # Filter out exact match (first result) and convert indices to IDs
            similar_ids = []
            for idx in indices[0][1:top_k+1]:  # Skip first (self match)
                if idx < len(self.similarity_ids):
                    similar_ids.append(self.similarity_ids[int(idx)])
            
            return similar_ids
        except Exception as e:
            print(f"[WARNING] Similarity search failed: {e}")
            return []
    
    def _get_priority_confidence(self) -> float:
        """Get confidence score for last priority prediction."""
        return getattr(self, '_last_priority_proba', 0.5)
    
    def _get_eta_confidence(self) -> float:
        """Get confidence score for last ETA prediction."""
        return 0.73  # Based on validation: 73.3% within ±3 days
    
    def batch_predict(self, complaints: List[Dict]) -> List[PredictionResult]:
        """
        Make predictions for multiple complaints.
        
        Args:
            complaints: List of complaint dictionaries
        
        Returns:
            List of PredictionResult objects
        """
        results = []
        for complaint in complaints:
            try:
                result = self.predict(complaint)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Prediction failed for complaint: {e}")
                results.append(None)
        
        return results


def load_inference_pipeline(models_dir: str = 'data/models') -> ComplaintRoutingInference:
    """
    Convenience function to load the inference pipeline.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        Initialized ComplaintRoutingInference instance
    """
    return ComplaintRoutingInference(models_dir=models_dir)


if __name__ == '__main__':
    # Example usage
    inference = ComplaintRoutingInference('data/models')
    
    # Test complaint
    test_complaint = {
        'id': 'test_001',
        'text': 'My Internet connection keeps dropping. I have tried rebooting the modem multiple times.',
        'language': 'en'
    }
    
    # Make prediction
    print("\n[INFO] Making prediction on test complaint...")
    result = inference.predict(test_complaint)
    
    # Print results
    print(f"\nComplaint ID: {result.complaint_id}")
    print(f"Suggested Officers: {result.assigned_officers}")
    print(f"Predicted Priority: {result.predicted_priority}")
    print(f"Predicted ETA: {result.predicted_eta_days} days")
    print(f"Similar Complaints: {result.similar_complaint_ids}")
    print(f"Confidence Scores: {result.confidence_scores}")
