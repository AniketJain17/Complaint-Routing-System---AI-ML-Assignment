"""
Video feature extraction for complaint recordings.
Uses OpenCV for frame extraction and pre-trained models for semantic features.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')


class VideoFeatureExtractor:
    """Extract features from video files."""
    
    def __init__(self, frame_sample_rate: int = 1):
        """
        Initialize video feature extractor.
        
        Args:
            frame_sample_rate: Sample every nth frame (e.g., 1 = every frame, 30 = every 30th)
        """
        self.frame_sample_rate = frame_sample_rate
        print(f"[INFO] Video Feature Extractor initialized")
        print(f"  Frame sampling rate: every {frame_sample_rate} frame(s)")
    
    def load_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Load video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            OpenCV VideoCapture object or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"[OK] Loaded video: {video_path}")
            print(f"  FPS: {fps:.1f}, Frames: {frame_count}, Duration: {duration:.2f}s")
            
            return cap
        except Exception as e:
            print(f"[ERROR] Failed to load video {video_path}: {e}")
            return None
    
    def extract_frames(self, video_path: str, max_frames: int = 100) -> Optional[List[np.ndarray]]:
        """
        Extract sampled frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frame arrays (RGB) or None if failed
        """
        cap = self.load_video(video_path)
        if cap is None:
            return None
        
        frames = []
        frame_idx = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % self.frame_sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"[OK] Extracted {len(frames)} frames")
        return frames
    
    def extract_color_histogram(self, frame: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram from frame.
        
        Args:
            frame: Frame array (H x W x 3)
            bins: Number of histogram bins per channel
        
        Returns:
            Flattened histogram vector
        """
        hist_features = []
        
        # Extract histogram for each RGB channel
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [bins], [0, 256])
            # Normalize and flatten
            hist_norm = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist_norm)
        
        # Concatenate all histograms
        features = np.concatenate(hist_features)
        return features
    
    def extract_edge_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract edge detection features from frame.
        
        Args:
            frame: Frame array (H x W x 3)
        
        Returns:
            Edge features vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Count edges and compute statistics
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / edges.size
        
        features = np.array([
            edge_count,
            edge_density,
            np.mean(edges),
            np.std(edges),
        ])
        
        return features
    
    def aggregate_frame_features(self, frame_features: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate features across frames.
        
        Args:
            frame_features: List of feature vectors from frames
        
        Returns:
            Aggregated feature vector
        """
        if not frame_features:
            return np.array([])
        
        frame_array = np.array(frame_features)
        
        # Compute statistics across frames
        mean = np.mean(frame_array, axis=0)
        std = np.std(frame_array, axis=0)
        min_val = np.min(frame_array, axis=0)
        max_val = np.max(frame_array, axis=0)
        
        aggregated = np.concatenate([mean, std, min_val, max_val])
        return aggregated
    
    def extract_video_features(self, video_path: str, max_frames: int = 50) -> Optional[np.ndarray]:
        """
        Extract complete video feature vector.
        Combines color histograms and edge features from sampled frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
        
        Returns:
            Fixed-size feature vector or None if failed
        """
        # Extract frames
        frames = self.extract_frames(video_path, max_frames=max_frames)
        if not frames:
            return None
        
        frame_features_list = []
        
        # Extract features from each frame
        for i, frame in enumerate(frames):
            # Color histogram features (96 dims for 32 bins x 3 channels)
            hist_features = self.extract_color_histogram(frame, bins=32)
            
            # Edge features (4 dims)
            edge_features = self.extract_edge_features(frame)
            
            # Combine
            frame_feat = np.concatenate([hist_features, edge_features])
            frame_features_list.append(frame_feat)
        
        # Aggregate across frames
        aggregated_features = self.aggregate_frame_features(frame_features_list)
        
        print(f"[OK] Extracted video features: shape {aggregated_features.shape}")
        return aggregated_features
    
    def extract_batch_features(self, video_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple video files.
        
        Args:
            video_paths: List of video file paths
        
        Returns:
            Dictionary mapping file paths to feature vectors
        """
        features_dict = {}
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"[INFO] Processing {i}/{len(video_paths)}...")
            features = self.extract_video_features(video_path)
            if features is not None:
                features_dict[video_path] = features
        
        print(f"[OK] Extracted features for {len(features_dict)}/{len(video_paths)} files")
        return features_dict
    
    def get_feature_dimension(self) -> int:
        """Get dimension of output feature vector."""
        # (Color histogram: 32*3) + (Edge features: 4)
        # Aggregated (mean, std, min, max): ((96+4)*4) = 400
        per_frame = 32 * 3 + 4  # 100
        aggregated = per_frame * 4  # 400
        return aggregated
    
    def get_extractor_info(self) -> Dict:
        """Get information about the extractor."""
        return {
            "frame_sample_rate": self.frame_sample_rate,
            "feature_components": ["Color Histogram", "Edge Detection"],
            "histogram_bins": 32,
            "histogram_channels": 3,
            "edge_features": 4,
            "aggregation": ["mean", "std", "min", "max"],
            "output_dimension": self.get_feature_dimension(),
        }


class VideoFeatures:
    """Container for video features."""
    
    def __init__(self, complaint_ids: List[str], features: Dict[str, np.ndarray]):
        """
        Initialize video features container.
        
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
        """Check if complaint has video features."""
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
    print("Video Feature Extraction Demo")
    print("=" * 60)
    
    extractor = VideoFeatureExtractor(frame_sample_rate=30)
    
    print(f"\nFeature extractor info:")
    info = extractor.get_extractor_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nOutput feature dimension:", extractor.get_feature_dimension())
    print("\n" + "=" * 60)
