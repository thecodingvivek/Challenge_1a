# scripts/training/prepare_data.py
import pandas as pd
from pathlib import Path
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_extractor import HybridPDFFeatureExtractor

def extract_features_for_training():
    """Extract features from all training data"""
    
    extractor = HybridPDFFeatureExtractor()
    
    # Process each split
    for split in ['training']:
        print(f"Processing {split} split...")
        
        pdf_dir = f"data/raw_pdfs/{split}"
        labels_dir = f"data/ground_truth/{split}"
        
        # Extract features
        df = extractor.process_directory(pdf_dir, labels_dir)
        
        if not df.empty:
            # Save features
            output_file = f"data/processed/{split}_features.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} feature rows to {output_file}")
        else:
            print(f"No features extracted for {split}")

if __name__ == "__main__":
    extract_features_for_training()