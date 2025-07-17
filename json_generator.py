#!/usr/bin/env python3
"""
JSON Generator for PDF Document Structure Detection
Trains a model on extracted features and generates Adobe-compliant JSON output
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import joblib

from feature_extractor import PDFFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeadingClassifier:
    """Machine learning classifier for document structure detection"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        self.categorical_features = ['font_name', 'line_position', 'alignment_type']
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/inference"""
        df = df.copy()
        
        # Handle categorical features
        categorical_encoders = {}
        for col in self.categorical_features:
            if col in df.columns:
                if self.label_encoder is None:  # Training mode
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    categorical_encoders[col] = le
                else:  # Inference mode - use saved encoders
                    if hasattr(self, 'categorical_encoders') and col in self.categorical_encoders:
                        le = self.categorical_encoders[col]
                        # Handle unknown categories by mapping them to 0 (or a more suitable default)
                        df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                    else:
                        df[col] = 0  # Default encoding if encoder is not found
        
        if self.label_encoder is None:  # Training mode
            self.categorical_encoders = categorical_encoders
        
        # Handle boolean features
        boolean_columns = ['is_bold', 'has_terminal_punct', 'isolated_block', 'centered_text', 'has_numbers']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Define feature columns for training
        if self.feature_columns is None:
            self.feature_columns = [
                'font_size', 'relative_size', 'is_bold', 'x_offset', 'y_pos_norm',
                'text_len', 'block_spacing', 'capital_ratio', 'title_case_ratio',
                'stopword_ratio', 'has_terminal_punct', 'ngram_score',
                'isolated_block', 'centered_text', 'word_count', 'avg_word_length',
                'has_numbers', 'special_char_ratio', 'page'
            ]
            
            # Add categorical features
            for col in self.categorical_features:
                if col in df.columns:
                    self.feature_columns.append(col)
        
        # Select only available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        df_features = df[available_features].copy()
        
        # Fill missing values (e.g., if a feature column was added but is missing in some rows)
        df_features = df_features.fillna(0)
        
        # Scale numerical features
        numerical_features = [col for col in df_features.columns if col not in self.categorical_features]
        
        if self.scaler is None:  # Training mode
            self.scaler = StandardScaler()
            if numerical_features:
                df_features[numerical_features] = self.scaler.fit_transform(df_features[numerical_features])
        else:  # Inference mode
            if numerical_features:
                df_features[numerical_features] = self.scaler.transform(df_features[numerical_features])
        
        return df_features
    
    def train(self, df: pd.DataFrame):
        """Train the heading classifier"""
        if df.empty or 'label' not in df.columns:
            raise ValueError("Training data must contain 'label' column")
        
        # Filter out None labels
        df = df[df['label'].notna()].copy()
        
        if df.empty:
            raise ValueError("No valid labels found in training data")
        
        logger.info(f"Training on {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['label'])
        
        # --- START OF MODIFICATION ---
        # Handle the ValueError: The least populated class in y has only 1 member
        # Check if any class has less than 2 samples
        min_samples_per_class = df['label'].value_counts().min()

        if min_samples_per_class < 2:
            logger.warning(f"Warning: At least one class has only {min_samples_per_class} sample(s). "
                           "Stratified split is not possible for these classes. "
                           "Proceeding with a non-stratified split or adjusting strategy for rare classes.")
            
            # Option 1 (Simpler but less ideal for imbalanced data): Remove stratify
            # This is the direct fix for the error, but might lead to poor splits for rare classes.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Option 2 (More robust, but requires more data or careful handling):
            # If you want to keep stratification for other classes, you'd need to:
            # 1. Identify and separate the rare classes (e.g., 'Title' in your case).
            # 2. Split the remaining data with stratification.
            # 3. Add the rare class samples entirely to the training set (or strategically to test if needed).
            # This is more complex and usually points to a need for more data.
            # For this immediate fix, Option 1 is implemented.

        else:
            # If all classes have at least 2 samples, proceed with stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        # --- END OF MODIFICATION ---

        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(self.label_encoder.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Classification report
        target_names = self.label_encoder.classes_
        # Get the unique integer labels that the LabelEncoder knows about
        # This ensures all classes are considered for the report, even if not in y_test
        labels_in_encoder = self.label_encoder.transform(self.label_encoder.classes_)

        report = classification_report(y_test, y_pred_classes,
                                       target_names=target_names,
                                       labels=labels_in_encoder, # Add this line
                                       zero_division='warn') # Add this to handle classes with no true samples in y_test gracefully
        logger.info(f"Classification report:\n{report}")
        
        return accuracy
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if df.empty:
            return np.array([])
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Ensure the feature columns match those used during training
        # This is crucial for consistent prediction
        missing_cols = set(self.feature_columns) - set(X.columns)
        for c in missing_cols:
            X[c] = 0 # Add missing columns with a default value (e.g., 0)
        X = X[self.feature_columns] # Ensure order of columns is the same

        # Predict
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert back to labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels
    
    def save_model(self, filepath: str):
        """Save trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and preprocessors"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.categorical_encoders = model_data['categorical_encoders']
        
        logger.info(f"Model loaded from {filepath}")


class JSONGenerator:
    """Generate Adobe-compliant JSON output from PDF predictions"""
    
    def __init__(self, model_path: str = "heading_classifier.pkl"):
        self.feature_extractor = PDFFeatureExtractor()
        self.classifier = HeadingClassifier()
        self.model_path = model_path
        
        # Load model if it exists
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)
    
    def train_model(self, input_dir: str, output_dir: str):
        """Train the classifier on PDF-JSON pairs"""
        logger.info("Training heading classifier...")
        
        # Extract features from training data
        df = self.feature_extractor.process_directory(input_dir, output_dir)
        
        if df.empty:
            raise ValueError("No training data found")
        
        # Train classifier
        accuracy = self.classifier.train(df)
        
        # Save model
        self.classifier.save_model(self.model_path)
        
        return accuracy
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF and return Adobe-compliant JSON"""
        if self.classifier.model is None:
            raise ValueError("Model not loaded. Train model first or provide valid model path.")
        
        # Extract features
        blocks = self.feature_extractor.process_pdf(pdf_path)
        
        if not blocks:
            return {"title": "", "outline": []}
        
        # Convert to DataFrame
        df = pd.DataFrame(blocks)
        
        # Predict labels
        predictions = self.classifier.predict(df)
        
        # Add predictions to blocks
        for i, block in enumerate(blocks):
            block['predicted_label'] = predictions[i] if i < len(predictions) else 'Paragraph'
        
        # Generate JSON output
        return self._generate_json_output(blocks)
    
    def _generate_json_output(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Adobe-compliant JSON from predicted blocks"""
        # Sort blocks by page and y-position
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        title = ""
        outline = []
        
        for block in sorted_blocks:
            label = block.get('predicted_label', 'Paragraph')
            text = block['text'].strip()
            page = block['page']
            
            if label == 'Title' and not title:
                title = text
            elif label in ['H1', 'H2', 'H3']:
                outline.append({
                    "level": label,
                    "text": text,
                    "page": page
                })
        
        # If no title found, use the first heading or first significant text
        if not title and outline:
            title = outline[0]['text']
        elif not title and sorted_blocks:
            # Find first non-paragraph block or first block with good characteristics
            for block in sorted_blocks:
                text = block['text'].strip()
                if (len(text) > 10 and len(text) < 100 and 
                    block.get('font_size', 0) > 12 and
                    block.get('page', 1) == 1):
                    title = text
                    break
        
        return {
            "title": title,
            "outline": outline
        }
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in input directory and save JSON files to output directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            try:
                # Process PDF
                result = self.process_pdf(str(pdf_file))
                
                # Save JSON
                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processed {pdf_file.name} -> {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")


def main():
    """Main function for training and inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Document Structure Detection")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="Mode: train or predict")
    parser.add_argument("--input", required=True, help="Input directory containing PDFs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="heading_classifier.pkl", help="Model file path")
    
    args = parser.parse_args()
    
    generator = JSONGenerator(model_path=args.model)
    
    if args.mode == "train":
        # Training mode: input contains PDFs, output contains corresponding JSONs
        logger.info("Training mode: Processing PDF-JSON pairs...")
        accuracy = generator.train_model(args.input, args.output)
        logger.info(f"Training completed with accuracy: {accuracy:.4f}")
        
    elif args.mode == "predict":
        # Prediction mode: input contains PDFs, output will contain generated JSONs
        logger.info("Prediction mode: Generating JSON outputs...")
        generator.process_directory(args.input, args.output)
        logger.info("Prediction completed")


if __name__ == "__main__":
    main()