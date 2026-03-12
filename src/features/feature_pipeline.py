"""
Unified feature engineering pipeline for complaint routing.
Combines text, audio, and video features with proper preprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

from .text_features import TextFeatureExtractor, TextFeatures
from .audio_features import AudioFeatureExtractor, AudioFeatures
from .video_features import VideoFeatureExtractor, VideoFeatures


class FeaturePipeline:
    """End-to-end feature engineering pipeline."""
    
    def __init__(self, text_model: str = "medium", audio_sr: int = 22050):
        """
        Initialize feature pipeline.
        
        Args:
            text_model: Text embedding model ('small', 'medium', 'large')
            audio_sr: Sample rate for audio
        """
        print("[INFO] Initializing Feature Pipeline...")
        
        # Initialize feature extractors
        self.text_extractor = TextFeatureExtractor(model_name=text_model)
        self.audio_extractor = AudioFeatureExtractor(sr=audio_sr, n_mfcc=13)
        self.video_extractor = VideoFeatureExtractor(frame_sample_rate=30)
        
        # Track feature dimensions
        self.text_dim = self.text_extractor.embedding_dim
        self.audio_dim = self.audio_extractor.get_feature_dimension()
        self.video_dim = self.video_extractor.get_feature_dimension()
        
        # Total dimension (when all modalities present)
        self.total_dim = self.text_dim + self.audio_dim + self.video_dim
        
        self.scalers = {}  # Scalers for each modality
        
        print(f"[OK] Feature Pipeline ready")
        print(f"  Text features: {self.text_dim}")
        print(f"  Audio features: {self.audio_dim}")
        print(f"  Video features: {self.video_dim}")
        print(f"  Total features (when all present): {self.total_dim}")
    
    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """Extract text embeddings."""
        print("[INFO] Extracting text features...")
        embeddings = self.text_extractor.extract_embeddings(texts)
        return embeddings
    
    def extract_audio_features(self, audio_paths: List[str]) -> Dict[str, np.ndarray]:
        """Extract audio features from files."""
        print("[INFO] Extracting audio features...")
        features_dict = {}
        
        for i, path in enumerate(audio_paths, 1):
            print(f"  [{i}/{len(audio_paths)}] Processing: {Path(path).name}")
            features = self.audio_extractor.extract_audio_features(path)
            if features is not None:
                features_dict[path] = features
        
        return features_dict
    
    def extract_video_features(self, video_paths: List[str]) -> Dict[str, np.ndarray]:
        """Extract video features from files."""
        print("[INFO] Extracting video features...")
        features_dict = {}
        
        for i, path in enumerate(video_paths, 1):
            print(f"  [{i}/{len(video_paths)}] Processing: {Path(path).name}")
            features = self.video_extractor.extract_video_features(path)
            if features is not None:
                features_dict[path] = features
        
        return features_dict
    
    def normalize_features(self, features: np.ndarray, modality: str = "text") -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            features: Feature matrix
            modality: Type of features ('text', 'audio', 'video')
        
        Returns:
            Normalized features
        """
        if modality not in self.scalers:
            self.scalers[modality] = StandardScaler()
            features_norm = self.scalers[modality].fit_transform(features)
        else:
            features_norm = self.scalers[modality].transform(features)
        
        return features_norm
    
    def combine_features(self, text_feats: np.ndarray, 
                         audio_feats: Optional[np.ndarray] = None,
                         video_feats: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine text, audio, and video features into single vector.
        Handles missing modalities gracefully.
        
        Args:
            text_feats: Text embeddings (n_samples, text_dim)
            audio_feats: Audio features (n_samples, audio_dim) or None
            video_feats: Video features (n_samples, video_dim) or None
        
        Returns:
            Combined features (n_samples, combined_dim)
        """
        combined_list = [text_feats]
        
        if audio_feats is not None:
            # Pad audio features to match number of samples
            if len(audio_feats) < len(text_feats):
                n_missing = len(text_feats) - len(audio_feats)
                padding = np.zeros((n_missing, audio_feats.shape[1]))
                audio_feats = np.vstack([audio_feats, padding])
            combined_list.append(audio_feats[:len(text_feats)])
        
        if video_feats is not None:
            # Pad video features to match number of samples
            if len(video_feats) < len(text_feats):
                n_missing = len(text_feats) - len(video_feats)
                padding = np.zeros((n_missing, video_feats.shape[1]))
                video_feats = np.vstack([video_feats, padding])
            combined_list.append(video_feats[:len(text_feats)])
        
        combined = np.hstack(combined_list)
        print(f"[OK] Combined features shape: {combined.shape}")
        return combined
    
    def process_complaints(self, complaints: List,
                          extract_audio: bool = False,
                          extract_video: bool = False) -> np.ndarray:
        """
        Process complaints and extract all features.
        
        Args:
            complaints: List of complaint objects
            extract_audio: Whether to extract audio features
            extract_video: Whether to extract video features
        
        Returns:
            Combined feature matrix
        """
        print("[INFO] Processing complaints...")
        
        # Extract text features
        texts = [c.text for c in complaints]
        text_feats = self.extract_text_features(texts)
        text_feats_norm = self.normalize_features(text_feats, "text")
        
        audio_feats = None
        video_feats = None
        
        # Extract audio features if requested and available
        if extract_audio:
            audio_paths = [c.audio_file for c in complaints if c.audio_file]
            if audio_paths:
                audio_dict = self.extract_audio_features(audio_paths)
                # Convert dict to array (order-preserving)
                audio_feats = np.array([audio_dict.get(p, np.zeros(self.audio_dim)) 
                                        for p in audio_paths])
                audio_feats_norm = self.normalize_features(audio_feats, "audio")
        
        # Extract video features if requested and available
        if extract_video:
            video_paths = [c.video_file for c in complaints if c.video_file]
            if video_paths:
                video_dict = self.extract_video_features(video_paths)
                # Convert dict to array (order-preserving)
                video_feats = np.array([video_dict.get(p, np.zeros(self.video_dim)) 
                                        for p in video_paths])
                video_feats_norm = self.normalize_features(video_feats, "video")
        
        # Combine all features
        combined = self.combine_features(text_feats_norm, audio_feats, video_feats)
        
        return combined
    
    def save_features(self, features: np.ndarray, filepath: str):
        """Save feature matrix to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
        print(f"[OK] Saved features to {filepath}")
    
    def load_features(self, filepath: str) -> np.ndarray:
        """Load feature matrix from file."""
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        print(f"[OK] Loaded features from {filepath}")
        return features
    
    def save_scalers(self, dirpath: str):
        """Save feature scalers for inference."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        for modality, scaler in self.scalers.items():
            filepath = dirpath / f'{modality}_scaler.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"[OK] Saved {modality} scaler to {filepath}")
    
    def load_scalers(self, dirpath: str):
        """Load feature scalers for inference."""
        dirpath = Path(dirpath)
        
        for modality_file in dirpath.glob('*_scaler.pkl'):
            modality = modality_file.stem.replace('_scaler', '')
            with open(modality_file, 'rb') as f:
                self.scalers[modality] = pickle.load(f)
            print(f"[OK] Loaded {modality} scaler from {modality_file}")
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline."""
        return {
            "text_model": self.text_extractor.model_name,
            "text_dimension": self.text_dim,
            "audio_dimension": self.audio_dim,
            "video_dimension": self.video_dim,
            "total_dimension": self.total_dim,
            "feature_components": ["text", "audio", "video"],
            "normalization": "StandardScaler per modality",
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Feature Pipeline Demo")
    print("=" * 60)
    
    pipeline = FeaturePipeline(text_model="medium")
    
    print("\nPipeline info:")
    info = pipeline.get_pipeline_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
