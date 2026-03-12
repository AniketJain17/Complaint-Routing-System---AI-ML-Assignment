"""
Data schemas for the Complaint Routing System.
Defines the structure of Officers and Complaints.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime
import json
from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for complaints."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplaintStatus(str, Enum):
    """Status of a complaint."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Officer:
    """Officer schema for complaint routing."""
    
    officer_id: str
    name: str
    expertise_areas: List[str]  # e.g., ["billing", "technical", "service"]
    languages: List[str]  # e.g., ["en", "es", "fr"]
    current_workload: int  # Number of open complaints
    max_capacity: int  # Maximum complaints they can handle
    avg_resolution_time_days: float  # Historical average resolution time
    rating: float  # Customer satisfaction rating (0-5)
    available: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Officer":
        """Create from dictionary."""
        return cls(**data)
    
    def get_availability(self) -> float:
        """Get availability percentage (0-1)."""
        if self.max_capacity == 0:
            return 0.0
        return max(0.0, min(1.0, (self.max_capacity - self.current_workload) / self.max_capacity))
    
    def is_available(self) -> bool:
        """Check if officer has capacity."""
        return self.available and self.current_workload < self.max_capacity


@dataclass
class Complaint:
    """Complaint schema for the routing system."""
    
    complaint_id: str
    text: str  # Complaint description (multilingual)
    language: str  # ISO language code (e.g., "en", "es", "fr")
    submitted_at: str  # ISO timestamp
    customer_id: str
    
    # Optional multimodal data paths
    audio_file: Optional[str] = None  # Path to audio recording
    video_file: Optional[str] = None  # Path to video recording
    
    # Ground truth labels (for training)
    priority: Optional[str] = None  # "low", "medium", "high"
    assigned_officer_id: Optional[str] = None
    eta_days: Optional[int] = None  # Expected days to resolution
    
    # Tracking
    status: str = ComplaintStatus.OPEN.value
    resolved_at: Optional[str] = None
    actual_resolution_days: Optional[int] = None
    
    # Metadata
    category: Optional[str] = None  # e.g., "billing", "technical", "service"
    source: str = "web"  # "web", "email", "phone", "mobile_app"
    sentiment: Optional[str] = None  # "negative", "neutral", "positive"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Complaint":
        """Create from dictionary."""
        return cls(**data)
    
    def get_priority_level(self) -> Optional[str]:
        """Get priority level."""
        if self.priority:
            return self.priority
        return None
    
    def is_multilingual(self) -> bool:
        """Check if complaint has non-English content."""
        return self.language != "en"


@dataclass
class PredictionResult:
    """Result from the complaint routing inference."""
    
    complaint_id: str
    
    # Routing predictions
    assigned_officers: List[Dict[str, float]] = field(default_factory=list)  # [{officer_id, score}, ...]
    
    # Priority prediction
    predicted_priority: str = "medium"
    priority_confidence: float = 0.0
    priority_scores: Dict[str, float] = field(default_factory=dict)  # {"low": 0.2, "medium": 0.5, "high": 0.3}
    
    # ETA prediction
    predicted_eta_days: float = 0.0
    eta_confidence: float = 0.0
    
    # Similarity search
    similar_complaints: List[Dict[str, float]] = field(default_factory=list)  # [{complaint_id, similarity_score}, ...]
    
    # Metadata
    inference_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PredictionResult":
        """Create from dictionary."""
        return cls(**data)


if __name__ == "__main__":
    # Example usage
    officer = Officer(
        officer_id="O001",
        name="John Smith",
        expertise_areas=["billing", "technical"],
        languages=["en", "es"],
        current_workload=5,
        max_capacity=10,
        avg_resolution_time_days=3.5,
        rating=4.8
    )
    
    print("Officer Example:")
    print(officer.to_json())
    print()
    
    complaint = Complaint(
        complaint_id="C001",
        text="I was charged twice for my subscription",
        language="en",
        submitted_at="2026-03-12T10:30:00",
        customer_id="CUST001",
        priority="high",
        assigned_officer_id="O001",
        eta_days=2,
        category="billing"
    )
    
    print("Complaint Example:")
    print(complaint.to_json())
