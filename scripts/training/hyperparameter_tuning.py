#!/usr/bin/env python3
"""
Fixed Hyperparameter Tuning Script
Handles data preprocessing and class imbalance issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, make_scorer, f1_score
import lightgbm as lgb
import json
import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.append('src')

def load_and_preprocess_data():
    """Load and preprocess training data"""
    
    print("Loading training data...")
    
    # Check if training features exist
    training_features_path = "data/processed/training_features.csv"
    if not Path(training_features_path).exists():
        print(f"Training features not found at {training_features_path}")
        print("Please run: python scripts/training/prepare_data.py first")
        return None, None
    
    # Load training data
    train_df = pd.read_csv(training_features_path)
    print(f"Loaded {len(train_df)} training samples")
    
    # Check for required columns
    if 'label' not in train_df.columns:
        print("Error: 'label' column not found in training data")
        return None, None
    
    # Print label distribution
    print(f"\nLabel distribution:")
    label_counts = train_df['label'].value_counts()
    print(label_counts)
    
    # Check for classes with very few samples
    min_samples = label_counts.min()
    if min_samples < 5:
        print(f"\nWarning: Some classes have only {min_samples} samples")
        print("Consider collecting more data or using stratified sampling")
    
    # Define columns to exclude from features
    exclude_columns = [
        'label', 'text', 'source_file', 'bbox', 'font_name', 
        'preliminary_label', 'heuristic_class', 'predicted_label'
    ]
    
    # Get feature columns (numeric only)
    feature_columns = []
    for col in train_df.columns:
        if col not in exclude_columns:
            # Check if column is numeric or can be converted
            try:
                # Try to convert a sample to check if it's numeric
                test_series = pd.to_numeric(train_df[col].dropna().head(10), errors='raise')
                feature_columns.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column: {col}")
    
    print(f"\nUsing {len(feature_columns)} numeric features")
    print(f"Feature columns: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"Feature columns: {feature_columns}")
    
    # Prepare features
    X = train_df[feature_columns].copy()
    
    # Convert all columns to numeric and handle any issues
    print("Converting columns to numeric...")
    for col in feature_columns:
        print(f"Processing column: {col} (dtype: {X[col].dtype})")
        
        # Convert to numeric, coercing errors to NaN
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill NaN values with 0
        X[col] = X[col].fillna(0)
        
        # Check if conversion was successful
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            print(f"Warning: Column {col} still has dtype {X[col].dtype}")
            # Force conversion to float
            X[col] = X[col].astype(float, errors='ignore')
    
    # Ensure all columns are numeric
    print("\nFinal data type check:")
    for col in X.columns:
        print(f"{col}: {X[col].dtype}")
        if X[col].dtype == 'object':
            print(f"Force converting {col} to float64")
            # Get unique non-null values to debug
            unique_vals = X[col].dropna().unique()[:5]
            print(f"Sample values: {unique_vals}")
            
            # Try to convert again, more aggressively
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    
    # Prepare labels
    y = train_df['label'].copy()
    
    # Remove rows with missing labels
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
    
    # Final verification that all data is numeric
    print("\nVerifying all data is numeric...")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Error: Column {col} is still not numeric: {X[col].dtype}")
            return None, None
    
    return X, y

def tune_random_forest(X, y):
    """Tune Random Forest hyperparameters"""
    
    print("\nTuning Random Forest...")
    
    # Check class distribution for cross-validation
    label_counts = pd.Series(y).value_counts()
    min_class_size = label_counts.min()
    
    # Adjust CV folds based on smallest class
    cv_folds = min(5, min_class_size - 1) if min_class_size > 1 else 2
    print(f"Using {cv_folds} CV folds (min class size: {min_class_size})")
    
    # Reduced parameter grid for faster tuning
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    
    # Create stratified CV
    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        print("Warning: Using simple train/test split due to insufficient data")
        cv = 2
    
    try:
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=1)  # Use single job for stability
        
        # Create scorer
        scorer = make_scorer(f1_score, average='macro', zero_division=0)
        
        # Perform grid search
        rf_search = GridSearchCV(
            rf, rf_params, 
            cv=cv, 
            scoring=scorer, 
            n_jobs=1,  # Use single job for stability
            verbose=1,
            error_score=0  # Return 0 for failed fits
        )
        
        print("Starting Random Forest grid search...")
        rf_search.fit(X, y)
        
        print(f"Best RF params: {rf_search.best_params_}")
        print(f"Best RF score: {rf_search.best_score_:.4f}")
        
        return rf_search.best_params_, rf_search.best_score_
        
    except Exception as e:
        print(f"Error during Random Forest tuning: {e}")
        return None, None

def tune_lightgbm(X, y):
    """Tune LightGBM hyperparameters"""
    
    print("\nTuning LightGBM...")
    
    try:
        # Encode labels for LightGBM
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Simple parameter grid
        lgb_params_list = [
            {
                'objective': 'multiclass',
                'num_class': len(le.classes_),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            {
                'objective': 'multiclass',
                'num_class': len(le.classes_),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 50,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.9,
                'bagging_freq': 10,
                'verbose': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        ]
        
        best_score = 0
        best_params = None
        
        for i, params in enumerate(lgb_params_list):
            print(f"Testing LightGBM configuration {i+1}/{len(lgb_params_list)}")
            
            try:
                # Simple train/validation split for LightGBM
                from sklearn.model_selection import train_test_split
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(stopping_rounds=10)]
                )
                
                # Evaluate
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                score = f1_score(y_val, y_pred_classes, average='macro', zero_division=0)
                
                print(f"LightGBM config {i+1} score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Error in LightGBM config {i+1}: {e}")
                continue
        
        if best_params:
            print(f"Best LightGBM score: {best_score:.4f}")
            return best_params, best_score
        else:
            print("No successful LightGBM configurations")
            return None, None
            
    except Exception as e:
        print(f"Error during LightGBM tuning: {e}")
        return None, None

def save_results(rf_results, lgb_results):
    """Save tuning results"""
    
    results = {
        'tuning_date': pd.Timestamp.now().isoformat(),
        'random_forest': {
            'best_params': rf_results[0] if rf_results[0] else None,
            'best_score': rf_results[1] if rf_results[1] else None
        },
        'lightgbm': {
            'best_params': lgb_results[0] if lgb_results[0] else None,
            'best_score': lgb_results[1] if lgb_results[1] else None
        }
    }
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results
    output_file = results_dir / "hyperparameter_tuning_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")

def tune_hyperparameters():
    """Main hyperparameter tuning function"""
    
    print("Starting hyperparameter tuning...")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return
    
    # Check data shape
    print(f"\nData shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Check for infinite or very large values - safer approach
    print("Checking for infinite values...")
    try:
        # Convert to numpy array and ensure it's float
        X_values = X.values.astype(float)
        
        # Check for infinite values
        inf_mask = np.isinf(X_values)
        if np.any(inf_mask):
            print("Warning: Infinite values detected, replacing with 0")
            X_values[inf_mask] = 0
            # Update the DataFrame
            X.iloc[:, :] = X_values
        
        # Check for very large values
        finite_values = X_values[np.isfinite(X_values)]
        if len(finite_values) > 0:
            max_val = np.max(np.abs(finite_values))
            if max_val > 1e6:
                print(f"Warning: Very large values detected (max: {max_val}), consider scaling")
        
    except Exception as e:
        print(f"Error checking infinite values: {e}")
        print("Proceeding without infinite value check...")
    
    # Tune Random Forest
    rf_results = tune_random_forest(X, y)
    
    # Tune LightGBM
    lgb_results = tune_lightgbm(X, y)
    
    # Save results
    save_results(rf_results, lgb_results)
    
    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)
    
    if rf_results[0]:
        print(f"Random Forest - Best Score: {rf_results[1]:.4f}")
        print(f"Random Forest - Best Params: {rf_results[0]}")
    else:
        print("Random Forest tuning failed")
    
    if lgb_results[0]:
        print(f"LightGBM - Best Score: {lgb_results[1]:.4f}")
        print(f"LightGBM - Best Params: {lgb_results[0]}")
    else:
        print("LightGBM tuning failed")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        tune_hyperparameters()
    except KeyboardInterrupt:
        print("\nTuning interrupted by user")
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()