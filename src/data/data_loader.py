"""
Data loaders and utilities for the Complaint Routing System.
Handles loading, preprocessing, and splitting data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from .schemas import Officer, Complaint


class DataLoader:
    """Load and manage complaint routing data."""
    
    def __init__(self, data_dir: str = "../../data"):
        """Initialize data loader with data directory path."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self._officers: Optional[List[Officer]] = None
        self._complaints: Optional[List[Complaint]] = None
    
    @property
    def officers(self) -> List[Officer]:
        """Lazy load officers."""
        if self._officers is None:
            self._officers = self.load_officers()
        return self._officers
    
    @property
    def complaints(self) -> List[Complaint]:
        """Lazy load complaints."""
        if self._complaints is None:
            self._complaints = self.load_complaints()
        return self._complaints
    
    def load_officers(self) -> List[Officer]:
        """Load officers from JSON file."""
        filepath = self.raw_dir / "officers.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Officers file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        officers = [Officer.from_dict(o) for o in data]
        print(f"[OK] Loaded {len(officers)} officers from {filepath}")
        return officers
    
    def load_complaints(self) -> List[Complaint]:
        """Load complaints from JSON file."""
        filepath = self.raw_dir / "complaints.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Complaints file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        complaints = [Complaint.from_dict(c) for c in data]
        print(f"[OK] Loaded {len(complaints)} complaints from {filepath}")
        return complaints
    
    def complaints_to_dataframe(self, complaints: List[Complaint] = None) -> pd.DataFrame:
        """Convert complaints list to DataFrame."""
        if complaints is None:
            complaints = self.complaints
        
        data = [c.to_dict() for c in complaints]
        df = pd.DataFrame(data)
        return df
    
    def officers_to_dataframe(self, officers: List[Officer] = None) -> pd.DataFrame:
        """Convert officers list to DataFrame."""
        if officers is None:
            officers = self.officers
        
        data = [o.to_dict() for o in officers]
        df = pd.DataFrame(data)
        return df
    
    def get_complaints_by_officer(self, officer_id: str) -> List[Complaint]:
        """Get all complaints assigned to an officer."""
        return [c for c in self.complaints if c.assigned_officer_id == officer_id]
    
    def get_complaints_by_priority(self, priority: str) -> List[Complaint]:
        """Get all complaints with a specific priority."""
        return [c for c in self.complaints if c.priority == priority]
    
    def get_complaints_by_language(self, language: str) -> List[Complaint]:
        """Get all complaints in a specific language."""
        return [c for c in self.complaints if c.language == language]
    
    def get_complaints_by_category(self, category: str) -> List[Complaint]:
        """Get all complaints in a specific category."""
        return [c for c in self.complaints if c.category == category]
    
    def get_multilingual_complaints(self) -> List[Complaint]:
        """Get all non-English complaints."""
        return [c for c in self.complaints if c.language != "en"]
    
    def get_labeled_complaints(self) -> List[Complaint]:
        """Get all complaints with labels (priority, ETA, assigned officer)."""
        return [c for c in self.complaints 
                if c.priority is not None 
                and c.eta_days is not None 
                and c.assigned_officer_id is not None]
    
    def get_complaints_with_audio(self) -> List[Complaint]:
        """Get complaints that have audio recordings."""
        return [c for c in self.complaints if c.audio_file is not None]
    
    def get_complaints_with_video(self) -> List[Complaint]:
        """Get complaints that have video recordings."""
        return [c for c in self.complaints if c.video_file is not None]
    
    def split_complaints(self, test_size: float = 0.2, val_size: float = 0.1,
                        random_state: int = 42) -> Tuple[List[Complaint], List[Complaint], List[Complaint]]:
        """
        Split complaints into train, validation, and test sets.
        
        Args:
            test_size: Fraction for test set (default 0.2)
            val_size: Fraction for validation set from remaining data (default 0.1)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_complaints, val_complaints, test_complaints)
        """
        complaints = self.get_labeled_complaints()
        
        # First split: separate test set
        train_val, test = train_test_split(
            complaints,
            test_size=test_size,
            random_state=random_state,
            stratify=[c.priority for c in complaints]  # Stratify by priority
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=[c.priority for c in train_val]  # Stratify by priority
        )
        
        print(f"[OK] Split complaints:")
        print(f"  Train: {len(train)} ({len(train)/len(complaints)*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/len(complaints)*100:.1f}%)")
        print(f"  Test:  {len(test)} ({len(test)/len(complaints)*100:.1f}%)")
        
        return train, val, test
    
    def get_statistics(self) -> Dict:
        """Get data statistics."""
        complaints = self.complaints
        officers = self.officers
        
        df_complaints = self.complaints_to_dataframe(complaints)
        
        stats = {
            "total_officers": len(officers),
            "total_complaints": len(complaints),
            "languages": df_complaints['language'].nunique(),
            "priority_distribution": df_complaints['priority'].value_counts().to_dict() if 'priority' in df_complaints else {},
            "category_distribution": df_complaints['category'].value_counts().to_dict() if 'category' in df_complaints else {},
            "avg_workload": sum(o.current_workload for o in officers) / len(officers) if officers else 0,
            "labeled_complaints": len(self.get_labeled_complaints()),
            "multilingual_complaints": len(self.get_multilingual_complaints()),
            "with_audio": len(self.get_complaints_with_audio()),
            "with_video": len(self.get_complaints_with_video()),
        }
        
        return stats
    
    def print_statistics(self):
        """Print data statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("Data Statistics")
        print("=" * 60)
        print(f"Total Officers:          {stats['total_officers']}")
        print(f"Total Complaints:        {stats['total_complaints']}")
        print(f"Unique Languages:        {stats['languages']}")
        print(f"Labeled Complaints:      {stats['labeled_complaints']}")
        print(f"Multilingual:            {stats['multilingual_complaints']}")
        print(f"With Audio:              {stats['with_audio']}")
        print(f"With Video:              {stats['with_video']}")
        print(f"Average Officer Workload: {stats['avg_workload']:.1f}")
        print("\nPriority Distribution:")
        for priority, count in stats['priority_distribution'].items():
            print(f"  {priority.upper():8} {count:4d} ({count/stats['total_complaints']*100:5.1f}%)")
        print("\nCategory Distribution:")
        for category, count in stats['category_distribution'].items():
            print(f"  {category:20} {count:4d} ({count/stats['total_complaints']*100:5.1f}%)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load data
    officers = loader.officers
    complaints = loader.complaints
    
    # Print statistics
    loader.print_statistics()
    
    # Show sample data
    print("\nSample Officer:")
    print(officers[0].to_json())
    
    print("\nSample Complaint:")
    print(complaints[0].to_json())
    
    # Split data
    print("\nSplitting data...")
    train, val, test = loader.split_complaints()
    
    # Filter examples
    print("\nFiltering examples:")
    print(f"Complaints by priority 'high': {len(loader.get_complaints_by_priority('high'))}")
    print(f"Multilingual complaints: {len(loader.get_multilingual_complaints())}")
    print(f"With audio recordings: {len(loader.get_complaints_with_audio())}")
