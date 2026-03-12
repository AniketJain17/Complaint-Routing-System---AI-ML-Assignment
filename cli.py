"""
Command-line interface for the complaint routing system.
Provides easy tools for making predictions and evaluating model performance.

Usage:
    python cli.py predict --text "complaint text" --language en
    python cli.py evaluate --test-data data/raw/complaints.json
    python cli.py batch --input-file complaints.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.inference.inference_pipeline import ComplaintRoutingInference
from src.data.data_loader import DataLoader


class ComplaintRoutingCLI:
    """Command-line interface for complaint routing system."""
    
    def __init__(self, models_dir: str = 'data/models'):
        """Initialize the CLI with the inference pipeline."""
        self.inference = ComplaintRoutingInference(models_dir=models_dir)
        self.data_loader = DataLoader('data')
    
    def predict_command(self, text: str, language: str = 'en', audio_path: str = None, video_path: str = None):
        """
        Predict routing, priority, and ETA for a single complaint.
        
        Args:
            text: Complaint text
            language: Language code (en, es, fr, de, pt, it, zh, ja)
            audio_path: Optional path to audio file
            video_path: Optional path to video file
        """
        complaint = {
            'text': text,
            'language': language
        }
        
        if audio_path:
            complaint['audio_path'] = audio_path
        if video_path:
            complaint['video_path'] = video_path
        
        print("\n[INFO] Processing complaint...")
        print(f"  Text: {text[:80]}...")
        print(f"  Language: {language}")
        
        result = self.inference.predict(complaint)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Suggested Officers:")
        for officer in result.assigned_officers:
            officer_id = officer['officer_id']
            confidence = officer['score']
            conf_pct = confidence * 100
            print(f"  - Officer {officer_id}: {conf_pct:.1f}%")
        
        print(f"\nPredicted Priority: {result.predicted_priority}")
        print(f"Predicted ETA: {result.predicted_eta_days} days")
        
        if result.similar_complaints:
            print(f"\nSimilar Complaints:")
            for complaint in result.similar_complaints:
                complaint_id = complaint['complaint_id']
                print(f"  - {complaint_id}")
        
        print(f"\nConfidence Scores:")
        print(f"  - priority: {result.priority_confidence*100:.1f}%")
        print(f"  - eta: {result.eta_confidence*100:.1f}%")
        
        print("="*60 + "\n")
        
        return result
    
    def evaluate_command(self, test_split: float = 0.15):
        """
        Evaluate model performance on test set.
        
        Args:
            test_split: Fraction of data to use as test set
        """
        print("\n[INFO] Loading evaluation data...")
        complaints = self.data_loader.get_all_complaints()
        
        # Split into test set (last 15%)
        test_size = int(len(complaints) * test_split)
        test_complaints = complaints[-test_size:]
        
        print(f"[INFO] Evaluating on {len(test_complaints)} test complaints...")
        
        # Make predictions
        predictions = self.inference.batch_predict([
            {
                'id': c['id'],
                'text': c['text'],
                'language': c['language']
            }
            for c in test_complaints
        ])
        
        # Calculate metrics
        officer_matches = 0
        priority_correct = 0
        eta_within_3days = 0
        
        for pred, actual in zip(predictions, test_complaints):
            if pred is None:
                continue
            
            # Officer routing: check if assigned officer is in actual
            assigned_officer_ids = [off_id for off_id, _ in pred.assigned_officers]
            if actual['assigned_officer_id'] in assigned_officer_ids:
                officer_matches += 1
            
            # Priority prediction
            if pred.predicted_priority == actual['priority']:
                priority_correct += 1
            
            # ETA within 3 days
            eta_diff = abs(pred.predicted_eta_days - actual['eta_days'])
            if eta_diff <= 3:
                eta_within_3days += 1
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        valid_preds = len([p for p in predictions if p is not None])
        
        if valid_preds > 0:
            print(f"\nDataset: {valid_preds} test complaints")
            
            print(f"\nOfficer Routing Performance:")
            officer_acc = officer_matches / valid_preds * 100
            print(f"  Top-3 Accuracy: {officer_acc:.1f}% ({officer_matches}/{valid_preds})")
            
            print(f"\nPriority Prediction Performance:")
            priority_acc = priority_correct / valid_preds * 100
            print(f"  Accuracy: {priority_acc:.1f}% ({priority_correct}/{valid_preds})")
            
            print(f"\nETA Prediction Performance:")
            eta_pct = eta_within_3days / valid_preds * 100
            print(f"  Within ±3 days: {eta_pct:.1f}% ({eta_within_3days}/{valid_preds})")
        
        print("="*60 + "\n")
    
    def batch_command(self, input_file: str, output_file: str = 'predictions.json'):
        """
        Make predictions for batch of complaints from CSV or JSON file.
        
        Args:
            input_file: Path to input file (JSON with 'text', 'language', 'id')
            output_file: Path to save predictions
        """
        print(f"\n[INFO] Loading complaints from {input_file}...")
        
        # Load complaints
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                complaints = data if isinstance(data, list) else [data]
            else:
                import csv
                reader = csv.DictReader(f)
                complaints = list(reader)
        
        print(f"[INFO] Loaded {len(complaints)} complaints")
        print(f"[INFO] Making predictions...")
        
        # Batch predict
        results = self.inference.batch_predict(complaints)
        
        # Save results
        output_data = []
        for result in results:
            if result:
                output_data.append({
                    'complaint_id': result.complaint_id,
                    'assigned_officers': [
                        {'officer_id': off, 'confidence': float(conf)}
                        for off, conf in result.assigned_officers
                    ],
                    'predicted_priority': result.predicted_priority,
                    'predicted_eta_days': result.predicted_eta_days,
                    'similar_complaints': result.similar_complaint_ids,
                    'confidence_scores': {k: float(v) for k, v in result.confidence_scores.items()}
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Saved {len(output_data)} predictions to {output_file}")
        print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Complaint Routing System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Make a single prediction
  python cli.py predict --text "My internet is down" --language en
  
  # Evaluate on test set
  python cli.py evaluate --test-split 0.15
  
  # Batch predict from file
  python cli.py batch --input complaints.json --output predictions.json
        """
    )
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict for a single complaint')
    predict_parser.add_argument('--text', required=True, help='Complaint text')
    predict_parser.add_argument('--language', default='en', help='Language code (default: en)')
    predict_parser.add_argument('--audio', help='Path to audio file (optional)')
    predict_parser.add_argument('--video', help='Path to video file (optional)')
    predict_parser.add_argument('--models-dir', default='data/models', help='Models directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models on test set')
    eval_parser.add_argument('--test-split', type=float, default=0.15, help='Test set fraction')
    eval_parser.add_argument('--models-dir', default='data/models', help='Models directory')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch predict from file')
    batch_parser.add_argument('--input', required=True, help='Input file (JSON)')
    batch_parser.add_argument('--output', default='predictions.json', help='Output file')
    batch_parser.add_argument('--models-dir', default='data/models', help='Models directory')
    
    args = parser.parse_args()
    
    # Execute command
    if not args.command:
        parser.print_help()
        return
    
    models_dir = getattr(args, 'models_dir', 'data/models')
    cli = ComplaintRoutingCLI(models_dir=models_dir)
    
    try:
        if args.command == 'predict':
            cli.predict_command(
                text=args.text,
                language=args.language,
                audio_path=args.audio,
                video_path=args.video
            )
        elif args.command == 'evaluate':
            cli.evaluate_command(test_split=args.test_split)
        elif args.command == 'batch':
            cli.batch_command(input_file=args.input, output_file=args.output)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
