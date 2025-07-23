# scripts/training/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import logging
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.json_generator import AdvancedHeadingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classifier():
    """Train the heading classifier"""
    
    # Load training data
    train_df = pd.read_csv("data/processed/training_features.csv")
    
    # Load validation data (handle if it doesn't exist)
    val_df = pd.DataFrame() # Initialize as empty
    try:
        val_df = pd.read_csv("data/processed/validation_features.csv")
    except FileNotFoundError:
        logger.warning("validation_features.csv not found. Proceeding without validation evaluation for now.")
    
    print(f"Training data: {len(train_df)} samples")
    if not val_df.empty:
        print(f"Validation data: {len(val_df)} samples")
    else:
        print("No validation data loaded.")
    
    # Check label distribution
    print(f"Training label distribution:")
    print(train_df['label'].value_counts())
    
    # Initialize classifier
    classifier = AdvancedHeadingClassifier()
    
    # Train model
    print("Training classifier...")
    accuracy = classifier.train(train_df)
    
    print(f"Training completed with accuracy: {accuracy:.4f}")
    
    # Evaluate on validation set if available
    val_accuracy = None
    if not val_df.empty:
        print("Evaluating on validation set...")
        val_predictions = classifier.predict(val_df)
        
        # Calculate validation accuracy
        val_accuracy = (val_predictions == val_df['label']).mean()
        print(f"Validation accuracy: {val_accuracy:.4f}")
    else:
        print("Skipping validation evaluation as no validation data is available.")
    
    # Save model
    model_path = "models/production/advanced_heading_classifier.pkl"
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training metadata
    metadata = {
        'training_samples': len(train_df),
        'validation_samples': len(val_df) if not val_df.empty else 0,
        'training_accuracy': float(accuracy),
        'validation_accuracy': float(val_accuracy) if val_accuracy is not None else None,
        'label_distribution': train_df['label'].value_counts().to_dict(),
        'feature_count': train_df.shape[1] - 1,  # Exclude label column
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open("models/production/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return classifier

if __name__ == "__main__":
    train_classifier()