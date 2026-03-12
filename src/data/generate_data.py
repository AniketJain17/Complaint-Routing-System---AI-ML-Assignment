"""
Synthetic data generation for the Complaint Routing System.
Generates realistic officers and complaints for training.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid

from .schemas import Officer, Complaint, PriorityLevel


class SyntheticDataGenerator:
    """Generate synthetic data for training."""
    
    # Predefined options for realistic data
    EXPERTISE_AREAS = [
        "billing",
        "technical_support",
        "refunds",
        "account_management",
        "service_quality",
        "product_issues",
        "subscription",
        "payment_methods",
        "data_privacy",
        "general_inquiry"
    ]
    
    LANGUAGES = ["en", "es", "fr", "de", "pt", "it", "zh", "ja"]
    
    CATEGORIES = [
        "billing",
        "technical",
        "service",
        "account",
        "product",
        "payment",
        "warranty",
        "delivery",
        "returns",
        "general"
    ]
    
    SOURCES = ["web", "email", "phone", "mobile_app", "chat", "social_media"]
    
    SENTIMENTS = ["negative", "neutral", "positive"]
    
    # Complaint templates for multilingual generation
    COMPLAINT_TEMPLATES = {
        "en": [
            "I was charged {amount} without authorization. This is not acceptable.",
            "The product I received is defective. I want a refund immediately.",
            "Your customer service is terrible. I've been waiting for {days} days with no response.",
            "My account was hacked. Please help me secure it.",
            "The billing information is incorrect. I was charged twice.",
            "I cannot access my account. Password reset is not working.",
            "The delivery took {days} days instead of the promised 2 days.",
            "The product does not match the description. I want to return it.",
            "Your service has been down for {hours} hours. This is unacceptable.",
            "I need to update my payment method but the system is broken.",
        ],
        "es": [
            "Fui cobrado sin autorización. Esto es inaceptable.",
            "El producto que recibí es defectuoso. Quiero un reembolso de inmediato.",
            "El servicio al cliente es terrible. He estado esperando {days} días sin respuesta.",
            "Mi cuenta fue hackeada. Por favor ayúdame a asegurarla.",
            "La información de facturación es incorrecta. Fui cobrado dos veces.",
        ],
        "fr": [
            "J'ai été débité sans autorisation. C'est inacceptable.",
            "Le produit que j'ai reçu est défectueux. Je veux un remboursement immédiatement.",
            "Le service client est terrible. J'attends {days} jours sans réponse.",
            "Mon compte a été piraté. Veuillez m'aider à le sécuriser.",
            "Les informations de facturation sont incorrectes. J'ai été débité deux fois.",
        ],
    }
    
    @staticmethod
    def generate_officer_id() -> str:
        """Generate a unique officer ID."""
        return f"OFC_{random.randint(1000, 9999)}"
    
    @staticmethod
    def generate_complaint_id() -> str:
        """Generate a unique complaint ID."""
        return f"CMP_{random.randint(100000, 999999)}"
    
    @staticmethod
    def generate_customer_id() -> str:
        """Generate a unique customer ID."""
        return f"CUST_{random.randint(10000, 99999)}"
    
    @staticmethod
    def generate_officer(officer_id: str = None) -> Officer:
        """Generate a single synthetic officer."""
        officer_id = officer_id or SyntheticDataGenerator.generate_officer_id()
        
        names = [
            "Alice Johnson", "Bob Williams", "Carol Davis", "David Miller",
            "Emma Wilson", "Frank Moore", "Grace Lee", "Henry Taylor",
            "Iris Anderson", "Jack Thomas", "Karen Martin", "Leo Jackson",
            "Maria White", "Nathan Harris", "Olivia Martin", "Peter Thompson"
        ]
        
        officer = Officer(
            officer_id=officer_id,
            name=random.choice(names),
            expertise_areas=random.sample(SyntheticDataGenerator.EXPERTISE_AREAS, 
                                         k=random.randint(2, 4)),
            languages=random.sample(SyntheticDataGenerator.LANGUAGES, 
                                   k=random.randint(1, 3)),
            current_workload=random.randint(0, 8),
            max_capacity=random.randint(8, 15),
            avg_resolution_time_days=round(random.uniform(1.5, 7.0), 1),
            rating=round(random.uniform(3.5, 5.0), 1),
            available=random.choice([True, True, True, False])  # 75% available
        )
        
        return officer
    
    @staticmethod
    def generate_complaint(complaint_id: str = None, language: str = None,
                          officer_id: str = None) -> Complaint:
        """Generate a single synthetic complaint."""
        complaint_id = complaint_id or SyntheticDataGenerator.generate_complaint_id()
        language = language or random.choice(SyntheticDataGenerator.LANGUAGES)
        
        # Generate complaint text
        if language in SyntheticDataGenerator.COMPLAINT_TEMPLATES:
            template = random.choice(SyntheticDataGenerator.COMPLAINT_TEMPLATES[language])
            complaint_text = template.format(
                amount=random.randint(50, 1000),
                days=random.randint(1, 30),
                hours=random.randint(1, 24)
            )
        else:
            complaint_text = f"I have a complaint about your service. Please resolve this issue {language}."
        
        # Determine category based on templates
        if "payment" in complaint_text.lower() or "charged" in complaint_text.lower():
            category = "billing"
        elif "product" in complaint_text.lower() or "defective" in complaint_text.lower():
            category = "product"
        elif "account" in complaint_text.lower():
            category = "account"
        else:
            category = random.choice(SyntheticDataGenerator.CATEGORIES)
        
        # Generate timestamps
        days_ago = random.randint(0, 60)
        submitted_at = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        # Determine priority and ETA
        priority = random.choices(
            [PriorityLevel.LOW, PriorityLevel.MEDIUM, PriorityLevel.HIGH],
            weights=[0.2, 0.5, 0.3]  # Most complaints are medium priority
        )[0].value
        
        eta_map = {
            PriorityLevel.LOW.value: random.randint(5, 14),
            PriorityLevel.MEDIUM.value: random.randint(2, 7),
            PriorityLevel.HIGH.value: random.randint(1, 3),
        }
        
        sentiment = "negative" if priority == "high" else random.choice(SyntheticDataGenerator.SENTIMENTS)
        
        complaint = Complaint(
            complaint_id=complaint_id,
            text=complaint_text,
            language=language,
            submitted_at=submitted_at,
            customer_id=SyntheticDataGenerator.generate_customer_id(),
            priority=priority,
            assigned_officer_id=officer_id,
            eta_days=eta_map[priority],
            category=category,
            source=random.choice(SyntheticDataGenerator.SOURCES),
            sentiment=sentiment,
            # Optional: 20% have audio recordings
            audio_file=f"audio_{complaint_id}.wav" if random.random() < 0.2 else None,
            # Optional: 10% have video recordings
            video_file=f"video_{complaint_id}.mp4" if random.random() < 0.1 else None,
            # Mark some as resolved
            status="resolved" if random.random() < 0.3 else "open",
            actual_resolution_days=random.randint(1, 14) if random.random() < 0.3 else None
        )
        
        return complaint
    
    @staticmethod
    def generate_officers(num_officers: int = 10) -> List[Officer]:
        """Generate multiple synthetic officers."""
        officers = []
        for i in range(num_officers):
            officer = SyntheticDataGenerator.generate_officer(
                officer_id=f"OFC_{i+1:04d}"
            )
            officers.append(officer)
        return officers
    
    @staticmethod
    def generate_complaints(num_complaints: int = 500, officers: List[Officer] = None) -> List[Complaint]:
        """Generate multiple synthetic complaints."""
        if officers is None:
            officers = []
        
        complaints = []
        officer_ids = [o.officer_id for o in officers] if officers else []
        
        for i in range(num_complaints):
            officer_id = random.choice(officer_ids) if officer_ids else None
            
            complaint = SyntheticDataGenerator.generate_complaint(
                complaint_id=f"CMP_{i+1:06d}",
                officer_id=officer_id
            )
            complaints.append(complaint)
        
        return complaints


def save_officers_to_json(officers: List[Officer], filepath: str):
    """Save officers to JSON file."""
    officers_list = [o.to_dict() for o in officers]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(officers_list, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(officers)} officers to {filepath}")


def save_complaints_to_json(complaints: List[Complaint], filepath: str):
    """Save complaints to JSON file."""
    complaints_list = [c.to_dict() for c in complaints]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(complaints_list, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(complaints)} complaints to {filepath}")


def load_officers_from_json(filepath: str) -> List[Officer]:
    """Load officers from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    officers = [Officer.from_dict(o) for o in data]
    print(f"[OK] Loaded {len(officers)} officers from {filepath}")
    return officers


def load_complaints_from_json(filepath: str) -> List[Complaint]:
    """Load complaints from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    complaints = [Complaint.from_dict(c) for c in data]
    print(f"[OK] Loaded {len(complaints)} complaints from {filepath}")
    return complaints


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Synthetic Data...")
    print("=" * 60)
    
    # Generate officers
    print("\n1. Generating Officers...")
    officers = SyntheticDataGenerator.generate_officers(num_officers=12)
    print(f"   Generated {len(officers)} officers")
    print(f"   Sample officer:\n   {officers[0].to_json()}")
    
    # Generate complaints
    print("\n2. Generating Complaints...")
    complaints = SyntheticDataGenerator.generate_complaints(num_complaints=500, officers=officers)
    print(f"   Generated {len(complaints)} complaints")
    print(f"   Sample complaint:\n   {complaints[0].to_json()}")
    
    # Save to files
    print("\n3. Saving to Files...")
    save_officers_to_json(officers, "../../../data/raw/officers.json")
    save_complaints_to_json(complaints, "../../../data/raw/complaints.json")
    
    # Verify by loading
    print("\n4. Verifying Data...")
    loaded_officers = load_officers_from_json("../../../data/raw/officers.json")
    loaded_complaints = load_complaints_from_json("../../../data/raw/complaints.json")
    
    print(f"   Officers verified: {len(loaded_officers)} == {len(officers)}")
    print(f"   Complaints verified: {len(loaded_complaints)} == {len(complaints)}")
    
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
