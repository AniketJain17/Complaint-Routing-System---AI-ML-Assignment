"""
Audio feature extraction for complaint recordings.
Uses librosa for MFCC, mel-spectrogram, and other audio features.
"""

import numpy as np
import librosa
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract features from audio files."""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize audio feature extractor.
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sr = sr  # Sample rate (Hz)
        self.n_mfcc = n_mfcc
        self.hop_length = 512
        self.n_fft = 2048
        
        print(f"[INFO] Audio Feature Extractor initialized")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  MFCC coefficients: {self.n_mfcc}")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            print(f"[OK] Loaded audio: {audio_path} (duration: {len(y)/sr:.2f}s)")
            return y, sr
        except Exception as e:
            print(f"[ERROR] Failed to load audio {audio_path}: {e}")
            return None, None
    
    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
        
        Args:
            y: Audio time series
            sr: Sample rate
        
        Returns:
            MFCC feature matrix of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return mfcc
    
    def extract_mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract mel-spectrogram features.
        
        Args:
            y: Audio time series
            sr: Sample rate
        
        Returns:
            Mel-spectrogram of shape (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=128
        )
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract chroma features (pitch information).
        
        Args:
            y: Audio time series
            sr: Sample rate
        
        Returns:
            Chroma feature matrix of shape (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return chroma
    
    def aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Aggregate temporal features to fixed-size vector.
        Computes mean, std, min, max across time dimension.
        
        Args:
            features: Feature matrix of shape (n_features, time_steps)
        
        Returns:
            Aggregated features of shape (n_features * 4,)
        """
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        min_val = np.min(features, axis=1)
        max_val = np.max(features, axis=1)
        
        aggregated = np.concatenate([mean, std, min_val, max_val])
        return aggregated
    
    def extract_audio_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract complete audio feature vector from file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Fixed-size feature vector concatenating MFCC, mel-spec, chroma
        """
        # Load audio
        y, sr = self.load_audio(audio_path)
        if y is None:
            return None
        
        # Extract features
        mfcc = self.extract_mfcc(y, sr)
        mel_spec = self.extract_mel_spectrogram(y, sr)
        chroma = self.extract_chroma(y, sr)
        
        # Aggregate
        mfcc_agg = self.aggregate_features(mfcc)  # n_mfcc * 4
        mel_spec_agg = self.aggregate_features(mel_spec)  # 128 * 4
        chroma_agg = self.aggregate_features(chroma)  # 12 * 4
        
        # Concatenate
        features = np.concatenate([mfcc_agg, mel_spec_agg, chroma_agg])
        
        print(f"[OK] Extracted audio features: shape {features.shape}")
        return features
    
    def extract_batch_features(self, audio_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            Dictionary mapping file paths to feature vectors
        """
        features_dict = {}
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"[INFO] Processing {i}/{len(audio_paths)}...")
            features = self.extract_audio_features(audio_path)
            if features is not None:
                features_dict[audio_path] = features
        
        print(f"[OK] Extracted features for {len(features_dict)}/{len(audio_paths)} files")
        return features_dict
    
    def get_feature_dimension(self) -> int:
        """Get dimension of output feature vector."""
        # MFCC (n_mfcc * 4) + Mel-spec (128 * 4) + Chroma (12 * 4)
        return self.n_mfcc * 4 + 128 * 4 + 12 * 4
    
    def get_extractor_info(self) -> Dict:
        """Get information about the extractor."""
        return {
            "sample_rate": self.sr,
            "n_mfcc": self.n_mfcc,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "feature_components": ["MFCC", "Mel-Spectrogram", "Chroma"],
            "output_dimension": self.get_feature_dimension(),
            "aggregation": ["mean", "std", "min", "max"],
        }


class AudioFeatures:
    """Container for audio features."""
    
    def __init__(self, complaint_ids: List[str], features: Dict[str, np.ndarray]):
        """
        Initialize audio features container.
        
        Args:
            complaint_ids: List of complaint IDs
            features: Dictionary mapping complaint IDs to feature vectors
        """
        self.complaint_ids = complaint_ids
        self.features = features
        self.n_samples = len([cid for cid in complaint_ids if cid in features])
        
        # Get feature dimension
        if self.features:
            first_features = next(iter(self.features.values()))
            self.feature_dim = len(first_features)
        else:
            self.feature_dim = 0
    
    def get_features(self, complaint_id: str) -> Optional[np.ndarray]:
        """Get features for specific complaint."""
        return self.features.get(complaint_id)
    
    def has_features(self, complaint_id: str) -> bool:
        """Check if complaint has audio features."""
        return complaint_id in self.features
    
    def get_available_ids(self) -> List[str]:
        """Get IDs of complaints with extracted features."""
        return list(self.features.keys())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "feature_dim": self.feature_dim,
            "available_ids": self.get_available_ids(),
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Audio Feature Extraction Demo")
    print("=" * 60)
    
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
    
    print(f"\nFeature extractor info:")
    info = extractor.get_extractor_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nOutput feature dimension:", extractor.get_feature_dimension())
    print("\n" + "=" * 60)
