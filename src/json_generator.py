#!/usr/bin/env python3
"""
Enhanced JSON Generator with Advanced ML Models and Ensemble Methods
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
import joblib

from .feature_extractor import EnhancedPDFFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedHeadingClassifier:
    """Advanced heading classifier with ensemble methods and heuristic integration"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_selector = None
        self.feature_columns = None
        self.categorical_encoders = {}
        self.class_weights = None
        
        # Feature categories
        self.heuristic_features = [
            'title_score', 'h1_score', 'h2_score', 'h3_score', 
            'heading_likelihood', 'heuristic_confidence'
        ]
        
        self.contextual_features = [
            'font_size_z_score', 'font_size_percentile', 'is_largest_font',
            'font_size_deviation', 'relative_position', 'page_position_normalized',
            'prev_font_size_diff', 'next_font_size_diff', 'text_length_z_score',
            'is_short_text', 'is_medium_text', 'is_long_text'
        ]
        
        self.linguistic_features = [
            'word_count', 'avg_word_length', 'sentence_count', 'avg_sentence_length',
            'unique_word_ratio', 'stopword_ratio', 'alpha_ratio', 'digit_ratio',
            'punct_ratio', 'bigram_diversity', 'flesch_score'
        ]
        
        self.visual_features = [
            'font_size', 'is_bold', 'block_width', 'block_height', 'aspect_ratio',
            'starts_with_number', 'ends_with_punct', 'has_colon', 'parentheses_count'
        ]
        
        self.positional_features = [
            'page', 'is_first_page', 'is_last_page', 'block_index', 'heading_rank',
            'heading_percentile'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for training/inference"""
        df = df.copy()
        
        # Handle categorical features
        categorical_features = ['font_name', 'preliminary_label', 'heuristic_class']
        
        for col in categorical_features:
            if col in df.columns:
                if col not in self.categorical_encoders:  # Training mode
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.categorical_encoders[col] = le
                else:  # Inference mode
                    le = self.categorical_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        
        # Handle boolean features
        boolean_columns = [
            'is_bold', 'is_largest_font', 'is_common_font_size', 'is_first_page',
            'is_last_page', 'is_short_text', 'is_medium_text', 'is_long_text',
            'starts_with_number', 'ends_with_punct', 'has_colon'
        ]
        
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Define comprehensive feature set
        if self.feature_columns is None:
            self.feature_columns = (
                self.heuristic_features + self.contextual_features + 
                self.linguistic_features + self.visual_features + 
                self.positional_features + categorical_features
            )
        
        # Select available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        df_features = df[available_features].copy()
        
        # Fill missing values
        df_features = df_features.fillna(0)
        
        # Feature scaling
        if self.scaler is None:  # Training mode
            self.scaler = RobustScaler()
            # Don't scale categorical and boolean features
            numerical_features = [
                col for col in df_features.columns 
                if col not in categorical_features + boolean_columns
            ]
            if numerical_features:
                df_features[numerical_features] = self.scaler.fit_transform(
                    df_features[numerical_features]
                )
        else:  # Inference mode
            numerical_features = [
                col for col in df_features.columns 
                if col not in categorical_features + boolean_columns
            ]
            if numerical_features:
                df_features[numerical_features] = self.scaler.transform(
                    df_features[numerical_features]
                )
        
        return df_features
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        
        # Interaction features
        if 'font_size_z_score' in df.columns and 'is_short_text' in df.columns:
            df['font_size_text_interaction'] = df['font_size_z_score'] * df['is_short_text']
        
        if 'heading_likelihood' in df.columns and 'page_position_normalized' in df.columns:
            df['heading_position_interaction'] = df['heading_likelihood'] * (1 - df['page_position_normalized'])
        
        if 'title_score' in df.columns and 'is_first_page' in df.columns:
            df['title_page_interaction'] = df['title_score'] * df['is_first_page']
        
        # Ratio features
        if 'word_count' in df.columns and 'sentence_count' in df.columns:
            df['words_per_sentence'] = df['word_count'] / np.maximum(df['sentence_count'], 1)
        
        if 'char_count' in df.columns and 'word_count' in df.columns:
            df['chars_per_word'] = df['char_count'] / np.maximum(df['word_count'], 1)
        
        # Ranking features
        if 'heading_likelihood' in df.columns:
            df['heading_likelihood_rank'] = df['heading_likelihood'].rank(ascending=False)
            df['heading_likelihood_percentile'] = df['heading_likelihood'].rank(pct=True)
        
        return df
    
    def train(self, df: pd.DataFrame):
        """Train ensemble of models"""
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
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y), y=y
        )
        self.class_weights = dict(zip(np.unique(y), class_weights))
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Handle stratified split
        min_samples_per_class = pd.Series(y).value_counts().min()
        
        if min_samples_per_class < 2:
            logger.warning("Some classes have only 1 sample. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Train individual models
        self._train_individual_models(X_train, y_train)
        
        # Create ensemble
        self._create_ensemble(X_train, y_train)
        
        # Evaluate on test set
        self._evaluate_models(X_test, y_test)
        
        return self._get_best_model_accuracy(X_test, y_test)
    
    def _train_individual_models(self, X_train, y_train):
        """Train individual models"""
        
        # LightGBM
        lgb_params = {
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
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        self.models['lgb'] = lgb.train(
            lgb_params, train_data, num_boost_round=100, 
            valid_sets=[train_data], callbacks=[lgb.early_stopping(10)]
        )
        
        # XGBoost
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, class_weight='balanced'
        )
        self.models['xgb'].fit(X_train, y_train)
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight='balanced'
        )
        self.models['rf'].fit(X_train, y_train)
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # Logistic Regression
        self.models['lr'] = LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000
        )
        self.models['lr'].fit(X_train, y_train)
        
        # Neural Network
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
        self.models['nn'].fit(X_train, y_train)
        
        logger.info(f"Trained {len(self.models)} individual models")
    
    def _create_ensemble(self, X_train, y_train):
        """Create ensemble model"""
        
        # Create voting classifier with sklearn models
        sklearn_models = [
            ('rf', self.models['rf']),
            ('gb', self.models['gb']),
            ('lr', self.models['lr']),
            ('nn', self.models['nn'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=sklearn_models, voting='soft'
        )
        self.ensemble_model.fit(X_train, y_train)
        
        logger.info("Created ensemble model")
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            try:
                if name in ['lgb']:
                    y_pred = model.predict(X_test)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    y_pred_classes = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred_classes)
                results[name] = accuracy
                
                logger.info(f"{name.upper()} accuracy: {accuracy:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = 0
        
        # Evaluate ensemble
        try:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            results['ensemble'] = ensemble_accuracy
            
            logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            # Detailed classification report for ensemble
            target_names = self.label_encoder.classes_
            report = classification_report(
                y_test, y_pred_ensemble, target_names=target_names, zero_division=0
            )
            logger.info(f"Ensemble classification report:\n{report}")
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            results['ensemble'] = 0
        
        self.model_results = results
    
    def _get_best_model_accuracy(self, X_test, y_test):
        """Get best model accuracy"""
        if hasattr(self, 'model_results'):
            best_model = max(self.model_results, key=self.model_results.get)
            return self.model_results[best_model]
        return 0
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict labels using ensemble of models"""
        if not self.models and self.ensemble_model is None:
            raise ValueError("No models trained. Call train() first.")
        
        if df.empty:
            return np.array([])
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Feature selection
        X_selected = self.feature_selector.transform(X)
        
        # Get predictions from ensemble
        if self.ensemble_model is not None:
            predictions = self.ensemble_model.predict(X_selected)
        else:
            # Fallback to best individual model
            best_model_name = max(self.model_results, key=self.model_results.get)
            model = self.models[best_model_name]
            
            if best_model_name == 'lgb':
                pred_proba = model.predict(X_selected)
                predictions = np.argmax(pred_proba, axis=1)
            else:
                predictions = model.predict(X_selected)
        
        # Convert back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels
    
    def predict_with_confidence(self, df: pd.DataFrame) -> tuple:
        """Predict labels with confidence scores"""
        if not self.models and self.ensemble_model is None:
            raise ValueError("No models trained. Call train() first.")
        
        if df.empty:
            return np.array([]), np.array([])
        
        # Prepare features
        X = self.prepare_features(df)
        X_selected = self.feature_selector.transform(X)
        
        # Get predictions and probabilities
        if self.ensemble_model is not None:
            predictions = self.ensemble_model.predict(X_selected)
            probabilities = self.ensemble_model.predict_proba(X_selected)
        else:
            # Fallback to best individual model
            best_model_name = max(self.model_results, key=self.model_results.get)
            model = self.models[best_model_name]
            
            if best_model_name == 'lgb':
                probabilities = model.predict(X_selected)
                predictions = np.argmax(probabilities, axis=1)
            else:
                predictions = model.predict(X_selected)
                probabilities = model.predict_proba(X_selected)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Convert back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, confidence_scores
    
    def save_model(self, filepath: str):
        """Save trained models and preprocessors"""
        if not self.models and self.ensemble_model is None:
            raise ValueError("No models to save")
        
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders,
            'class_weights': self.class_weights
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models and preprocessors"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.ensemble_model = model_data['ensemble_model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_columns = model_data['feature_columns']
        self.categorical_encoders = model_data['categorical_encoders']
        self.class_weights = model_data['class_weights']
        
        logger.info(f"Models loaded from {filepath}")


class EnhancedJSONGenerator:
    """Enhanced JSON Generator with advanced ML and contextual analysis"""
    
    def __init__(self, model_path: str = "models/production/advanced_heading_classifier.pkl"):
        self.feature_extractor = EnhancedPDFFeatureExtractor()
        self.classifier = AdvancedHeadingClassifier()
        self.model_path = model_path
        
        # Load model if it exists
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)
    
    def train_model(self, input_dir: str, output_dir: str):
        """Train the advanced classifier"""
        logger.info("Training advanced heading classifier...")
        
        # Extract features
        df = self.feature_extractor.process_directory(input_dir, output_dir)
        
        if df.empty:
            raise ValueError("No training data found")
        
        # Train classifier
        accuracy = self.classifier.train(df)
        
        # Save model
        self.classifier.save_model(self.model_path)
        
        return accuracy
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF with advanced analysis"""
        if not self.classifier.models and self.classifier.ensemble_model is None:
            raise ValueError("Model not loaded. Train model first or provide valid model path.")
        
        # Extract features
        blocks = self.feature_extractor.process_pdf(pdf_path)
        
        if not blocks:
            return {"title": "", "outline": []}
        
        # Convert to DataFrame
        df = pd.DataFrame(blocks)
        
        # Get predictions with confidence
        predictions, confidence_scores = self.classifier.predict_with_confidence(df)
        
        # Add predictions to blocks
        for i, block in enumerate(blocks):
            block['predicted_label'] = predictions[i] if i < len(predictions) else 'Paragraph'
            block['confidence'] = confidence_scores[i] if i < len(confidence_scores) else 0.0
        
        # Generate JSON output with advanced post-processing
        return self._generate_enhanced_json_output(blocks)
    
    def _generate_enhanced_json_output(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate JSON output with advanced post-processing"""
        
        # Sort blocks by page and position
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        # Post-process predictions using contextual rules
        sorted_blocks = self._post_process_predictions(sorted_blocks)
        
        title = ""
        outline = []
        
        # Extract title and outline
        for block in sorted_blocks:
            label = block.get('predicted_label', 'Paragraph')
            text = block['text'].strip()
            page = block['page']
            confidence = block.get('confidence', 0.0)
            
            # Only include high-confidence predictions
            if confidence < 0.3:
                continue
            
            if label == 'Title' and not title:
                title = text
            elif label in ['H1', 'H2', 'H3']:
                outline.append({
                    "level": label,
                    "text": text,
                    "page": page
                })
        
        # Fallback title selection
        if not title:
            title = self._select_fallback_title(sorted_blocks)
        
        # Post-process outline
        outline = self._post_process_outline(outline)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _post_process_predictions(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process predictions using contextual rules"""
        
        # Rule 1: If multiple titles predicted, keep only the first high-confidence one
        title_found = False
        for block in blocks:
            if block.get('predicted_label') == 'Title':
                if title_found or block.get('confidence', 0) < 0.5:
                    block['predicted_label'] = 'H1'
                else:
                    title_found = True
        
        # Rule 2: Ensure logical heading hierarchy
        prev_level = 0
        for block in blocks:
            label = block.get('predicted_label', 'Paragraph')
            if label in ['H1', 'H2', 'H3']:
                level = int(label[1])
                # Don't allow jumping more than one level
                if level > prev_level + 1:
                    block['predicted_label'] = f'H{prev_level + 1}'
                    level = prev_level + 1
                prev_level = level
        
        # Rule 3: Remove low-confidence headings that are too similar to nearby text
        for i, block in enumerate(blocks):
            if block.get('predicted_label') in ['H1', 'H2', 'H3']:
                if block.get('confidence', 0) < 0.4:
                    # Check if similar to nearby paragraphs
                    similar_nearby = False
                    for j in range(max(0, i-2), min(len(blocks), i+3)):
                        if j != i and blocks[j].get('predicted_label') == 'Paragraph':
                            if self._text_similarity(block['text'], blocks[j]['text']) > 0.7:
                                similar_nearby = True
                                break
                    
                    if similar_nearby:
                        block['predicted_label'] = 'Paragraph'
        
        return blocks
    
    def _select_fallback_title(self, blocks: List[Dict[str, Any]]) -> str:
        """Select fallback title when no title is found"""
        
        # Look for first page, large font, short text
        candidates = []
        
        for block in blocks:
            if (block['page'] == 1 and 
                block.get('font_size_z_score', 0) > 1 and
                block.get('is_short_text', False)):
                
                score = (block.get('font_size_z_score', 0) + 
                        block.get('title_score', 0) + 
                        (3 if block.get('is_first_page', False) else 0))
                
                candidates.append((score, block['text']))
        
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        # Final fallback: first heading or first significant text
        for block in blocks:
            if block.get('predicted_label') in ['H1', 'H2', 'H3']:
                return block['text']
        
        # Last resort: first block with reasonable characteristics
        for block in blocks:
            text = block['text'].strip()
            if 10 <= len(text) <= 100 and block['page'] == 1:
                return text
        
        return ""
    
    def _post_process_outline(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process outline to ensure quality"""
        
        # Remove duplicates
        seen = set()
        unique_outline = []
        
        for item in outline:
            key = (item['level'], item['text'].lower().strip())
            if key not in seen:
                seen.add(key)
                unique_outline.append(item)
        
        # Sort by page and level
        unique_outline.sort(key=lambda x: (x['page'], x['level']))
        
        return unique_outline
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in input directory"""
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
    
    parser = argparse.ArgumentParser(description="Advanced PDF Document Structure Detection")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="Mode: train or predict")
    parser.add_argument("--input", required=True, help="Input directory containing PDFs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="models/production/advanced_heading_classifier.pkl", 
                        help="Model file path")
    
    args = parser.parse_args()
    
    generator = EnhancedJSONGenerator(model_path=args.model)
    
    if args.mode == "train":
        logger.info("Training mode: Processing PDF-JSON pairs...")
        accuracy = generator.train_model(args.input, args.output)
        logger.info(f"Training completed with accuracy: {accuracy:.4f}")
        
    elif args.mode == "predict":
        logger.info("Prediction mode: Generating JSON outputs...")
        generator.process_directory(args.input, args.output)
        logger.info("Prediction completed")


if __name__ == "__main__":
    main()