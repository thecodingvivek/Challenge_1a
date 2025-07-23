#!/usr/bin/env python3
"""
Data Diagnostic and Cleanup Script
Identifies and fixes common data issues before training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

def diagnose_data(csv_path: str):
    """Diagnose issues in the training data"""
    
    print(f"Diagnosing data in: {csv_path}")
    print("="*60)
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Check for required columns
    required_columns = ['label']
    for col in required_columns:
        if col in df.columns:
            print(f"✅ Required column '{col}' found")
        else:
            print(f"❌ Required column '{col}' missing")
    
    # Label analysis
    if 'label' in df.columns:
        print(f"\n📊 Label Distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        # Check for class imbalance
        min_samples = label_counts.min()
        max_samples = label_counts.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        if min_samples < 5:
            print(f"⚠️  Classes with very few samples (min: {min_samples})")
            print("   Consider collecting more data or merging similar classes")
        
        if imbalance_ratio > 10:
            print(f"⚠️  High class imbalance (ratio: {imbalance_ratio:.1f})")
            print("   Consider using class weighting or resampling")
    
    # Column analysis
    print(f"\n📋 Column Analysis:")
    
    numeric_columns = []
    string_columns = []
    problematic_columns = []
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Count missing values
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Try to identify column type
        if col_type in ['int64', 'float64']:
            numeric_columns.append(col)
            status = "✅ Numeric"
        elif col_type == 'object':
            # Check if it can be converted to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
                status = "⚠️  Object but numeric"
            except:
                # Check for specific problematic patterns
                sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if sample_val.startswith('(') and sample_val.endswith(')'):
                    problematic_columns.append(col)
                    status = "❌ Tuple/Coordinate string"
                elif ',' in sample_val and len(sample_val) > 20:
                    problematic_columns.append(col)
                    status = "❌ Complex string"
                else:
                    string_columns.append(col)
                    status = "⚠️  String/Categorical"
        else:
            problematic_columns.append(col)
            status = f"❌ Unknown type: {col_type}"
        
        print(f"  {col:30} | {status:25} | Missing: {missing_pct:5.1f}%")
    
    # Summary
    print(f"\n📈 Column Summary:")
    print(f"  Numeric columns: {len(numeric_columns)}")
    print(f"  String columns: {len(string_columns)}")
    print(f"  Problematic columns: {len(problematic_columns)}")
    
    if problematic_columns:
        print(f"\n❌ Problematic columns that need fixing:")
        for col in problematic_columns:
            print(f"  - {col}")
    
    # Memory and performance analysis
    print(f"\n💾 Memory Analysis:")
    for col in df.columns:
        mem_usage = df[col].memory_usage(deep=True) / 1024 / 1024
        if mem_usage > 10:  # More than 10MB
            print(f"  Large column '{col}': {mem_usage:.2f} MB")
    
    # Check for infinite values
    if numeric_columns:
        inf_cols = []
        for col in numeric_columns:
            if col in df.columns:
                try:
                    if np.any(np.isinf(pd.to_numeric(df[col], errors='coerce'))):
                        inf_cols.append(col)
                except:
                    pass
        
        if inf_cols:
            print(f"\n⚠️  Columns with infinite values: {inf_cols}")
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': numeric_columns,
        'string_columns': string_columns, 
        'problematic_columns': problematic_columns,
        'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
        'missing_data': df.isnull().sum().to_dict()
    }

def clean_data(input_csv: str, output_csv: str = None):
    """Clean and prepare data for training"""
    
    print(f"\nCleaning data from: {input_csv}")
    
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_cleaned.csv')
    
    # Load data
    df = pd.read_csv(input_csv)
    original_shape = df.shape
    
    print(f"Original shape: {original_shape}")
    
    # Define columns to exclude
    exclude_columns = [
        'text', 'source_file', 'bbox', 'font_name', 
        'preliminary_label', 'heuristic_class'
    ]
    
    # Remove problematic columns
    columns_to_drop = []
    for col in df.columns:
        if col in exclude_columns:
            columns_to_drop.append(col)
            continue
            
        if col == 'label':
            continue
            
        # Check if column can be converted to numeric
        try:
            # Try converting to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # If more than 50% are NaN after conversion, it's probably not numeric
            nan_ratio = numeric_series.isnull().sum() / len(df)
            if nan_ratio > 0.5:
                columns_to_drop.append(col)
                print(f"Dropping non-numeric column: {col} (NaN ratio: {nan_ratio:.2f})")
            else:
                # Replace original column with numeric version
                df[col] = numeric_series.fillna(0)
                
        except Exception as e:
            columns_to_drop.append(col)
            print(f"Dropping problematic column: {col} ({e})")
    
    # Drop problematic columns
    df = df.drop(columns=columns_to_drop)
    
    # Handle missing values in remaining columns
    numeric_columns = [col for col in df.columns if col != 'label']
    for col in numeric_columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values in {col}")
            df[col] = df[col].fillna(0)
    
    # Handle infinite values
    for col in numeric_columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"Replacing {inf_count} infinite values in {col}")
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    # Remove rows with missing labels
    if 'label' in df.columns:
        missing_labels = df['label'].isnull().sum()
        if missing_labels > 0:
            print(f"Removing {missing_labels} rows with missing labels")
            df = df[df['label'].notna()]
    
    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows")
        df = df.drop_duplicates()
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows and {original_shape[1] - df.shape[1]} columns")
    
    # Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned data to: {output_csv}")
    
    return df

def main():
    """Main diagnostic function"""
    
    # Check for training features
    training_features_path = "data/processed/training_features.csv"
    
    if not Path(training_features_path).exists():
        print(f"Training features not found at {training_features_path}")
        print("Please run: python scripts/training/prepare_data.py first")
        return
    
    # Diagnose data
    diagnosis = diagnose_data(training_features_path)
    
    if diagnosis is None:
        return
    
    # Check if cleaning is needed
    needs_cleaning = (
        len(diagnosis['problematic_columns']) > 0 or
        any(count > len(diagnosis['label_distribution']) * 0.1 for count in diagnosis['missing_data'].values())
    )
    
    if needs_cleaning:
        print(f"\n🔧 Data needs cleaning. Running cleanup...")
        
        # Clean the data
        cleaned_df = clean_data(training_features_path)
        
        # Re-diagnose cleaned data
        print(f"\n🔍 Re-diagnosing cleaned data...")
        diagnose_data(training_features_path.replace('.csv', '_cleaned.csv'))
        
        print(f"\n✅ Use the cleaned data file for training:")
        print(f"   {training_features_path.replace('.csv', '_cleaned.csv')}")
        
    else:
        print(f"\n✅ Data looks good for training!")
    
    # Save diagnosis report
    report_path = "results/data_diagnosis_report.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(diagnosis, f, indent=2, default=str)
    
    print(f"\n📋 Diagnosis report saved to: {report_path}")

if __name__ == "__main__":
    main()