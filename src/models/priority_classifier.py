"""
Priority classification model - predicts complaint priority (low/medium/high).
Uses multiple algorithms: Logistic Regression, Random Forest, Gradient Boosting.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path


class PriorityClassifier:
    """Classify complaint priority as low, medium, or high."""
    
    PRIORITY_LEVELS = ['low', 'medium', 'high']
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize priority classifier.
        
        Args:
            model_type: 'logistic_regression', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        self._init_model()
        print(f"[INFO] Priority Classifier initialized with {model_type}")
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: List[str], validate: bool = True) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Priority labels (low/medium/high)
            validate: Whether to compute cross-validation scores
        
        Returns:
            Training results dictionary
        """
        print(f"[INFO] Training Priority Classifier...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Compute cross-validation scores
        results = {
            'model_type': self.model_type,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'classes': list(self.label_encoder.classes_),
        }
        
        if validate:
            cv_scores = cross_val_score(
                self.model, X_scaled, y_encoded,
                cv=5, scoring='f1_weighted'
            )
            results['cv_mean_f1'] = cv_scores.mean()
            results['cv_std_f1'] = cv_scores.std()
            
            print(f"[OK] Training complete")
            print(f"   CV F1 Score: {results['cv_mean_f1']:.4f} (+/- {results['cv_std_f1']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict priority for new complaints.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            List of predicted priority labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        y_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_encoded)
        
        return list(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError(f"Model {self.model_type} doesn't support predict_proba")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predict priority with confidence scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            List of dicts with predicted_priority and confidence
        """
        predictions = self.predict(X)
        
        try:
            probabilities = self.predict_proba(X)
        except:
            # Fallback if probabilities not available
            probabilities = None
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'predicted_priority': pred,
                'confidence': float(np.max(probabilities[i])) if probabilities is not None else None,
                'probabilities': {
                    cls: float(prob)
                    for cls, prob in zip(self.label_encoder.classes_, probabilities[i])
                } if probabilities is not None else {}
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Cannot save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
        }
        
        joblib.dump(model_data, filepath)
        print(f"[OK] Saved priority classifier to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"[OK] Loaded priority classifier from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'priority_levels': self.PRIORITY_LEVELS,
            'training_status': 'trained' if self.is_trained else 'untrained',
            'n_features': self.scaler.n_features_in_ if self.is_trained else None,
        }
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores (if available)."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return {i: float(imp) for i, imp in enumerate(importances)}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Priority Classifier Demo")
    print("=" * 60)
    
    # Dummy data
    n_samples = 100
    n_features = 768
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(['low', 'medium', 'high'], size=n_samples, p=[0.2, 0.5, 0.3])
    
    # Train classifier
    clf = PriorityClassifier(model_type='random_forest')
    results = clf.train(X, y)
    
    print(f"\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Make predictions
    X_test = np.random.randn(5, n_features)
    predictions = clf.predict(X_test)
    predictions_conf = clf.predict_with_confidence(X_test)
    
    print(f"\nSample Predictions:")
    for i, pred in enumerate(predictions_conf, 1):
        print(f"  {i}. Priority: {pred['predicted_priority']}, "
              f"Confidence: {pred['confidence']:.4f}")
    
    print("\n" + "=" * 60)
