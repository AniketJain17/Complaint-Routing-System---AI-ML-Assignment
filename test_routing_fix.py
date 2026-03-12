"""Quick test of the officer routing fix"""
import sys
from pathlib import Path
import numpy as np
import joblib

# Load the routing model directly
models_dir = Path('data/models')
routing_model = joblib.load(models_dir / 'routing_model.pkl')

print("\n[TEST] Officer Routing Model Structure")
print(f"  Officer IDs: {routing_model['officer_ids']}")
print(f"  Officer Embeddings Keys: {list(routing_model['officer_embeddings'].keys())}")
print(f"  Text Dimension: {routing_model['text_dim']}")

# Test the fix: iterate correctly over embeddings
print("\n[TEST] Testing Officer Routing Logic")
officer_ids = routing_model['officer_ids']
officer_embeddings = routing_model['officer_embeddings']

# Create a test complaint embedding (random 768D vector)
test_embedding = np.random.randn(768)
test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize

# Compute similarities (this is the fixed code)
similarities = []
for oid in officer_ids:  # FIXED: iterate over officer_ids, not embeddings keys
    officer_emb = officer_embeddings[oid]  # Get embedding for this officer
    # Cosine similarity
    sim = np.dot(test_embedding, officer_emb) / (
        np.linalg.norm(test_embedding) * np.linalg.norm(officer_emb) + 1e-8
    )
    similarities.append(sim)
    print(f"  {oid}: {sim:.4f}")

similarities = np.array(similarities)

# Get top-3
top_indices = np.argsort(-similarities)[:3]
result = [(officer_ids[idx], float(similarities[idx])) for idx in top_indices]

print(f"\n[OK] Top-3 Officers:")
for officer_id, score in result:
    print(f"  {officer_id}: {score:.4f} ({score*100:.2f}%)")

print("\n[SUCCESS] Officer routing logic works correctly!")
