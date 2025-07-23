#!/usr/bin/env python3
"""
Quick test to verify training data is ready
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys

def test_training_data(csv_path: str):
    """Quick test of training data"""
    
    print(f"Testing training data: {csv_path}")
    print("="*50)
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} samples")
        
        # Check for label column
        if 'label' not in df.columns:
            print("❌ No 'label' column found")
            return False
        
        # Prepare features and labels
        exclude_columns = ['label', 'text', 'source_file', 'bbox', 'font_name']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        print(f"✅ Found {len(feature_columns)} feature columns")
        
        # Check for numeric features
        X = df[feature_columns].copy()
        y = df['label'].copy()
        
        # Convert to numeric and handle missing values
        for col in feature_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Processed features shape: {X.shape}")
        
        # Check labels
        label_counts = y.value_counts()
        print(f"✅ Label distribution: {dict(label_counts)}")
        
        min_samples = label_counts.min()
        if min_samples < 2:
            print(f"⚠️  Warning: Some classes have only {min_samples} samples")
        
        # Quick training test
        print("\n🧪 Quick training test...")
        
        if len(df) < 10:
            print("⚠️  Very small dataset, skipping train/test split")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            # Use stratify if possible
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                # Fallback without stratify
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
        
        # Test Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X_train, y_train)
            
            accuracy = rf.score(X_test, y_test)
            print(f"✅ Random Forest test accuracy: {accuracy:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Random Forest test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def main():
    """Test all available training data files"""
    
    # Test paths to check
    test_paths = [
        "data/processed/training_features.csv",
        "data/processed/training_features_cleaned.csv"
    ]
    
    success_count = 0
    
    for path in test_paths:
        if Path(path).exists():
            print(f"\n{'='*60}")
            success = test_training_data(path)
            if success:
                success_count += 1
                print(f"✅ {path} is ready for training!")
            else:
                print(f"❌ {path} has issues")
        else:
            print(f"⏭️  Skipping {path} (not found)")
    
    print(f"\n{'='*60}")
    if success_count > 0:
        print(f"✅ {success_count} file(s) ready for training")
        print("\nNext steps:")
        print("1. Run hyperparameter tuning: python scripts/training/hyperparameter_tuning.py")
        print("2. Train full model: python scripts/training/train_model.py")
    else:
        print("❌ No files ready for training")
        print("\nTroubleshooting steps:")
        print("1. Run data diagnosis: python scripts/data_preparation/diagnose_data.py")
        print("2. Prepare training data: python scripts/training/prepare_data.py")

if __name__ == "__main__":
    main()