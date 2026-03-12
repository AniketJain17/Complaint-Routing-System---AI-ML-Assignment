"""
Complete model training pipeline for complaint routing system.
Trains all 4 models: routing, priority, ETA, and similarity search.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'features'))

from data_loader import DataLoader
from feature_pipeline import FeaturePipeline
from vector_search import VectorSearchIndex, SimilarityMatcher
from officer_router import OfficerRoutingModel
from priority_classifier import PriorityClassifier
from eta_regressor import ETARegressor

# Import metrics
from metrics import (
    RoutingMetrics, PriorityMetrics, ETAMetrics,
    SimilarityMetrics, EvaluationReport
)


class ComplaintRoutingTrainer:
    """Train all models for complaint routing system."""
    
    def __init__(self, data_dir: str, models_dir: str):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing raw data
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("[INFO] Loading data...")
        self.loader = DataLoader(data_dir=data_dir)
        self.officers = self.loader.officers
        self.complaints = self.loader.complaints
        
        # Print statistics
        self.loader.print_statistics()
        
        # Initialize pipeline
        print("\n[INFO] Initializing feature pipeline...")
        self.feature_pipeline = FeaturePipeline(text_model="medium")
        
        # Data splits
        self.train_complaints = None
        self.val_complaints = None
        self.test_complaints = None
        
        # Trained models
        self.routing_model = None
        self.priority_model = None
        self.eta_model = None
        self.similarity_matcher = None
    
    def prepare_data(self):
        """Prepare and split data."""
        print("\n" + "=" * 70)
        print("1. DATA PREPARATION")
        print("=" * 70)
        
        # Get labeled complaints
        labeled_complaints = self.loader.get_labeled_complaints()
        
        # Split data
        self.train_complaints, self.val_complaints, self.test_complaints = \
            self.loader.split_complaints(test_size=0.15, val_size=0.15)
        
        print(f"\n[OK] Data split:")
        print(f"  Train: {len(self.train_complaints)} complaints")
        print(f"  Val:   {len(self.val_complaints)} complaints")
        print(f"  Test:  {len(self.test_complaints)} complaints")
    
    def extract_features(self):
        """Extract features for all complaints."""
        print("\n" + "=" * 70)
        print("2. FEATURE EXTRACTION")
        print("=" * 70)
        
        print("\n[INFO] Extracting features...")
        
        # Extract text features for all splits
        print("\n[INFO] Processing text features...")
        
        train_texts = [c.text for c in self.train_complaints]
        val_texts = [c.text for c in self.val_complaints]
        test_texts = [c.text for c in self.test_complaints]
        
        self.train_text_feats = self.feature_pipeline.text_extractor.extract_embeddings(train_texts)
        self.val_text_feats = self.feature_pipeline.text_extractor.extract_embeddings(val_texts)
        self.test_text_feats = self.feature_pipeline.text_extractor.extract_embeddings(test_texts)
        
        # Normalize text features
        self.train_text_feats_norm = self.feature_pipeline.normalize_features(
            self.train_text_feats, "text"
        )
        self.val_text_feats_norm = self.feature_pipeline.normalize_features(
            self.val_text_feats, "text"
        )
        self.test_text_feats_norm = self.feature_pipeline.normalize_features(
            self.test_text_feats, "text"
        )
        
        print(f"[OK] Text features extracted")
        print(f"  Train: {self.train_text_feats.shape}")
        print(f"  Val:   {self.val_text_feats.shape}")
        print(f"  Test:  {self.test_text_feats.shape}")
    
    def train_routing_model(self):
        """Train officer routing model."""
        print("\n" + "=" * 70)
        print("3. OFFICER ROUTING MODEL")
        print("=" * 70)
        
        # Create and train model
        self.routing_model = OfficerRoutingModel(self.officers)
        
        train_officers = [c.assigned_officer_id for c in self.train_complaints]
        self.routing_model.train(self.train_text_feats_norm, train_officers)
        
        # Validate on validation set
        print("\n[INFO] Validating routing model...")
        val_officers_true = [c.assigned_officer_id for c in self.val_complaints]
        val_predictions = self.routing_model.predict_batch(self.val_text_feats_norm, k=5)
        
        # Extract top-1 predictions for evaluation
        val_top1_preds = [preds[0]['officer_id'] for preds in val_predictions]
        
        # Compute metrics
        accuracy = sum(1 for t, p in zip(val_officers_true, val_top1_preds) if t == p) / len(val_officers_true)
        
        # MRR
        ranked_preds = [[p['officer_id'] for p in preds] for preds in val_predictions]
        mrr = RoutingMetrics.mean_reciprocal_rank(val_officers_true, ranked_preds)
        
        print(f"[OK] Routing Model Validation:")
        print(f"  Top-1 Accuracy: {accuracy:.4f}")
        print(f"  MRR@5: {mrr:.4f}")
        
        # Save model
        model_path = self.models_dir / 'routing_model.pkl'
        self.routing_model.save_model(str(model_path))
    
    def train_priority_model(self):
        """Train priority classification model."""
        print("\n" + "=" * 70)
        print("4. PRIORITY CLASSIFICATION MODEL")
        print("=" * 70)
        
        # Create and train model
        self.priority_model = PriorityClassifier(model_type='random_forest')
        
        train_priorities = [c.priority for c in self.train_complaints]
        self.priority_model.train(self.train_text_feats_norm, train_priorities)
        
        # Validate on validation set
        print("\n[INFO] Validating priority model...")
        val_priorities_true = [c.priority for c in self.val_complaints]
        val_priorities_pred = self.priority_model.predict(self.val_text_feats_norm)
        
        # Compute metrics
        accuracy = sum(1 for t, p in zip(val_priorities_true, val_priorities_pred) if t == p) / len(val_priorities_true)
        macro_f1 = PriorityMetrics.macro_f1(val_priorities_true, val_priorities_pred)
        
        print(f"[OK] Priority Model Validation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        
        # Save model
        model_path = self.models_dir / 'priority_model.pkl'
        self.priority_model.save_model(str(model_path))
    
    def train_eta_model(self):
        """Train ETA regression model."""
        print("\n" + "=" * 70)
        print("5. ETA REGRESSION MODEL")
        print("=" * 70)
        
        # Create and train model
        self.eta_model = ETARegressor(model_type='gradient_boosting')
        
        train_etas = np.array([c.eta_days for c in self.train_complaints], dtype=float)
        self.eta_model.train(self.train_text_feats_norm, train_etas)
        
        # Validate on validation set
        print("\n[INFO] Validating ETA model...")
        val_etas_true = np.array([c.eta_days for c in self.val_complaints], dtype=float)
        val_etas_pred = self.eta_model.predict(self.val_text_feats_norm)
        
        # Compute metrics
        mae = ETAMetrics.mae(val_etas_true, val_etas_pred)
        rmse = ETAMetrics.rmse(val_etas_true, val_etas_pred)
        within_tol = ETAMetrics.within_tolerance(val_etas_true, val_etas_pred, tolerance=3)
        
        print(f"[OK] ETA Model Validation:")
        print(f"  MAE: {mae:.4f} days")
        print(f"  RMSE: {rmse:.4f} days")
        print(f"  Within 3 days: {within_tol:.1f}%")
        
        # Save model
        model_path = self.models_dir / 'eta_model.pkl'
        self.eta_model.save_model(str(model_path))
    
    def build_similarity_index(self):
        """Build FAISS vector search index for similarity."""
        print("\n" + "=" * 70)
        print("6. SIMILARITY SEARCH INDEX")
        print("=" * 70)
        
        # Train embeddings
        train_complaint_ids = [c.complaint_id for c in self.train_complaints]
        
        # Build index
        print(f"\n[INFO] Building FAISS index from {len(train_complaint_ids)} complaints...")
        self.similarity_matcher = SimilarityMatcher(
            self.train_text_feats_norm,
            train_complaint_ids
        )
        
        # Validate on test set
        print(f"\n[INFO] Validating similarity search...")
        test_complaint_ids = [c.complaint_id for c in self.test_complaints]
        
        # Find similar complaints for test set samples
        similar_results = self.similarity_matcher.index.search_batch(
            self.test_text_feats_norm, k=5
        )
        
        coverage = SimilarityMetrics.coverage_at_k(
            test_complaint_ids, similar_results, k=5
        )
        
        print(f"[OK] Similarity Index Validation:")
        print(f"  Coverage@5: {coverage:.1f}%")
        
        # Save index
        index_dir = self.models_dir / 'similarity_index'
        self.similarity_matcher.index.save_index(str(index_dir))
    
    def save_feature_scalers(self):
        """Save feature scalers for inference."""
        print("\n[INFO] Saving feature scalers...")
        scalers_dir = self.models_dir / 'scalers'
        self.feature_pipeline.save_scalers(str(scalers_dir))
    
    def train_all(self):
        """Run complete training pipeline."""
        print("\n" + "=" * 70)
        print("COMPLAINT ROUTING SYSTEM - MODEL TRAINING")
        print("=" * 70)
        
        # Prepare data
        self.prepare_data()
        
        # Extract features
        self.extract_features()
        
        # Train all models
        self.train_routing_model()
        self.train_priority_model()
        self.train_eta_model()
        self.build_similarity_index()
        
        # Save scalers
        self.save_feature_scalers()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - All models saved!")
        print("=" * 70)
        print(f"\nModels saved to: {self.models_dir}")
        print(f"  - routing_model.pkl")
        print(f"  - priority_model.pkl")
        print(f"  - eta_model.pkl")
        print(f"  - similarity_index/")
        print(f"  - scalers/")


if __name__ == "__main__":
    # Configuration
    data_dir = r"e:\Project\Ivx_assignment\data"
    models_dir = r"e:\Project\Ivx_assignment\data\models"
    
    # Create trainer
    trainer = ComplaintRoutingTrainer(data_dir, models_dir)
    
    # Train all models
    trainer.train_all()
