"""
ETA regression model - predicts resolution time in days.
Uses multiple algorithms: Linear Regression, Random Forest, Gradient Boosting.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path


class ETARegressor:
    """Predict complaint resolution time (ETA in days)."""
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize ETA regressor.
        
        Args:
            model_type: 'linear', 'ridge', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        self._init_model()
        print(f"[INFO] ETA Regressor initialized with {model_type}")
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                loss='huber'  # Robust to outliers
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, validate: bool = True) -> Dict:
        """
        Train the regressor.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: ETA labels in days (continuous values)
            validate: Whether to compute cross-validation scores
        
        Returns:
            Training results dictionary
        """
        print(f"[INFO] Training ETA Regressor...")
        
        # Ensure y is numeric
        y = np.asarray(y, dtype=float)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Compute training metrics
        train_predictions = self.model.predict(X_scaled)
        train_mae = np.mean(np.abs(y - train_predictions))
        train_rmse = np.sqrt(np.mean((y - train_predictions) ** 2))
        
        results = {
            'model_type': self.model_type,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
        }
        
        if validate:
            # Cross-validation with MAE scoring
            cv_scores = -cross_val_score(
                self.model, X_scaled, y,
                cv=5, scoring='neg_mean_absolute_error'
            )
            results['cv_mean_mae'] = cv_scores.mean()
            results['cv_std_mae'] = cv_scores.std()
            
            print(f"[OK] Training complete")
            print(f"   Train MAE: {train_mae:.4f} days")
            print(f"   CV MAE: {results['cv_mean_mae']:.4f} (+/- {results['cv_std_mae']:.4f}) days")
        else:
            print(f"[OK] Training complete")
            print(f"   Train MAE: {train_mae:.4f} days")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ETA for new complaints.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted ETAs in days (array)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 1)
        
        return y_pred
    
    def predict_with_interval(self, X: np.ndarray, confidence: float = 0.95) -> List[Dict]:
        """
        Predict ETA with confidence intervals.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            confidence: Confidence level (0-1)
        
        Returns:
            List of dicts with predicted_eta and confidence_interval
        """
        predictions = self.predict(X)
        
        # Estimate prediction uncertainty (using residual std from training)
        # For simplicity, use a fixed uncertainty margin
        uncertainty = np.std(predictions) * 0.5
        
        results = []
        for pred in predictions:
            margin = uncertainty * 1.96  # 95% confidence interval
            results.append({
                'predicted_eta_days': float(pred),
                'lower_bound': float(max(1, pred - margin)),
                'upper_bound': float(pred + margin),
                'confidence': confidence,
            })
        
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
            'model_type': self.model_type,
        }
        
        joblib.dump(model_data, filepath)
        print(f"[OK] Saved ETA regressor to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"[OK] Loaded ETA regressor from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'task': 'ETA Prediction (days)',
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
    print("ETA Regressor Demo")
    print("=" * 60)
    
    # Dummy data
    n_samples = 100
    n_features = 768
    X = np.random.randn(n_samples, n_features)
    y = np.random.uniform(1, 14, size=n_samples)  # ETA between 1-14 days
    
    # Train regressor
    regressor = ETARegressor(model_type='gradient_boosting')
    results = regressor.train(X, y)
    
    print(f"\nTraining Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Make predictions
    X_test = np.random.randn(5, n_features)
    predictions = regressor.predict(X_test)
    predictions_conf = regressor.predict_with_interval(X_test)
    
    print(f"\nSample Predictions:")
    for i, pred in enumerate(predictions_conf, 1):
        print(f"  {i}. ETA: {pred['predicted_eta_days']:.1f} days "
              f"({pred['lower_bound']:.1f} - {pred['upper_bound']:.1f})")
    
    print("\n" + "=" * 60)
