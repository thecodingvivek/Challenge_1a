#!/usr/bin/env python3
"""
JSON Generator:
    Combines advanced ML and heuristic approaches for maximum accuracy
"""

import os
import sys
import json
import logging
import fitz
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyOptimizedClassifier:
    """Advanced ensemble classifier for maximum accuracy"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.trained = False
    
    def train_advanced_models(self, df: pd.DataFrame) -> float:
        """Train ensemble of models for maximum accuracy"""
        
        logger.info("Training advanced ensemble models...")
        
        # Prepare features and labels
        X = df.drop(['label'], axis=1)
        y = df['label']
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with optimized parameters
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
                learning_rate_init=0.01, alpha=0.01
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=2, random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
        }
        
        # Train models and calculate weights
        accuracies = {}
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies[name] = accuracy
                self.models[name] = model
                logger.info(f"{name} accuracy: {accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Calculate ensemble weights based on accuracy
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            self.ensemble_weights = {name: acc/total_accuracy for name, acc in accuracies.items()}
        
        # Calculate overall ensemble accuracy
        ensemble_predictions = self.predict_ensemble(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        
        logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        self.trained = True
        
        return ensemble_accuracy
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        if not self.models:
            return np.array([])
        
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Failed to predict with {name}: {e}")
        
        if not predictions:
            return np.array([])
        
        # Weighted voting
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(predictions))
            ensemble_pred += weight * pred
        
        return np.round(ensemble_pred).astype(int)
    
    def predict_proba_ensemble(self, X):
        """Get prediction probabilities"""
        if not self.models:
            return np.array([])
        
        probabilities = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities[name] = proba
            except Exception as e:
                logger.warning(f"Failed to get probabilities from {name}: {e}")
        
        if not probabilities:
            return np.array([])
        
        # Weighted average of probabilities
        ensemble_proba = None
        for name, proba in probabilities.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(probabilities))
            if ensemble_proba is None:
                ensemble_proba = weight * proba
            else:
                ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'model_components': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_columns),
            'training_samples': 'unknown'
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data.get('models', {})
        self.ensemble_weights = model_data.get('ensemble_weights', {})
        self.scaler = model_data.get('scaler', StandardScaler())
        self.label_encoder = model_data.get('label_encoder', LabelEncoder())
        self.feature_columns = model_data.get('feature_columns', [])
        self.trained = model_data.get('trained', False)
        
        logger.info(f"Model loaded from {filepath}")

class JSONGenerator:
    """JSON generator with unified feature processing"""
    
    def __init__(self, model_path: str = "models/production/ultra_accuracy_optimized_classifier.pkl"):
        self.feature_extractor = FeatureExtractor()
        self.classifier = AccuracyOptimizedClassifier()
        self.model_path = model_path
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                self.classifier.load_model(model_path)
                logger.info("Pre-trained model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF with features"""
        
        try:
            # Extract blocks using extractor
            doc = fitz.open(pdf_path)
            blocks = self.feature_extractor.extract_text_blocks(doc)
            doc.close()
            
            # Compute features
            blocks_with_features = self.feature_extractor.compute_ultra_features(blocks)
            
            # Convert to DataFrame for ML processing
            if self.classifier.trained and self.classifier.models:
                df_features = self._blocks_to_dataframe(blocks_with_features)
                if not df_features.empty:
                    predictions = self._predict_with_ml(df_features, blocks_with_features)
                else:
                    predictions = self._apply_ultra_heuristics(blocks_with_features)
            else:
                predictions = self._apply_ultra_heuristics(blocks_with_features)
            
            # Generate structured output
            result = self._generate_structured_output(predictions, blocks_with_features)
            
            # Add metadata
            result.update({
                'total_blocks': len(blocks),
                'processing_method': 'ultra_enhanced_ml' if self.classifier.models else 'ultra_heuristic'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {"title": "", "outline": [], "error": str(e)}
    
    def _blocks_to_dataframe(self, blocks: List[Dict]) -> pd.DataFrame:
        """Convert blocks with features to DataFrame for ML processing"""
        try:
            feature_data = []
            for block in blocks:
                row = {}
                # Add numeric features only, ensuring proper data types
                for k, v in block.items():
                    if k not in ['text', 'lines', 'bbox', 'spans']:
                        # Convert to appropriate numeric type
                        if isinstance(v, bool):
                            row[k] = int(v)  # Convert boolean to int
                        elif isinstance(v, (int, float)):
                            row[k] = float(v)  # Ensure all numeric features are float
                        elif isinstance(v, str):
                            # Skip string features except for specific ones we want to encode
                            continue
                        else:
                            # For any other type, try to convert to float or skip
                            try:
                                row[k] = float(v)
                            except (ValueError, TypeError):
                                continue
                
                feature_data.append(row)
            
            return pd.DataFrame(feature_data)
        except Exception as e:
            logger.error(f"Error converting blocks to DataFrame: {e}")
            return pd.DataFrame()
    
    def _predict_with_ml(self, df_features: pd.DataFrame, blocks: List[Dict]) -> List[str]:
        """Optimized ML predictions with intelligent fallback"""
        
        try:
            # PERFORMANCE: Batch processing optimization
            X = df_features.copy()
            
            # PERFORMANCE: Faster column handling
            missing_cols = set(self.classifier.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            
            # PERFORMANCE: Efficient reordering
            X = X.reindex(columns=self.classifier.feature_columns, fill_value=0)
            
            # PERFORMANCE: Vectorized data type conversion
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
            
            # PERFORMANCE: Single scaling operation
            X_scaled = self.classifier.scaler.transform(X)
            
            # PERFORMANCE: Optimized ensemble prediction
            predictions = self._fast_ensemble_predict(X_scaled)
            probabilities = self._fast_ensemble_proba(X_scaled)
            
            # Convert back to labels
            labels = self.classifier.label_encoder.inverse_transform(predictions)
            
            # ACCURACY: Document-type aware confidence adjustment
            doc_stats = self._compute_document_statistics(blocks)
            doc_type = self._detect_document_type(blocks, doc_stats)
            
            final_predictions = []
            
            for i, (label, block) in enumerate(zip(labels, blocks)):
                confidence = np.max(probabilities[i]) if probabilities is not None and len(probabilities) > i else 0.5
                
                # ACCURACY: Dynamic confidence thresholds by document type - VERY HIGH THRESHOLDS TO PREFER HEURISTICS
                confidence_thresholds = {
                    'academic': 0.8,      # Very high - prefer heuristics
                    'technical': 0.85,    # Very high - prefer heuristics
                    'business': 0.85,     # Very high - prefer heuristics
                    'form': 0.75,         # Very high - prefer heuristics
                    'invitation': 0.9,    # Very high - prefer heuristics
                    'general': 0.8        # Very high - prefer heuristics
                }
                
                threshold = confidence_thresholds.get(doc_type, 0.4)
                
                # ACCURACY: Smart ML/heuristic hybrid decision
                if confidence < threshold or self._should_use_heuristic(label, block, doc_type):
                    heuristic_label = self._get_heuristic_label_fast(block, doc_stats, doc_type)
                    final_predictions.append(heuristic_label)
                else:
                    final_predictions.append(label)
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # PERFORMANCE: Fast fallback to optimized heuristics
            doc_stats = self._compute_document_statistics(blocks)
            doc_type = self._detect_document_type(blocks, doc_stats)
            return [self._get_heuristic_label_fast(block, doc_stats, doc_type) for block in blocks]
    
    def _fast_ensemble_predict(self, X):
        """Optimized ensemble prediction - RESTORED FULL ENSEMBLE FOR ACCURACY"""
        if not self.classifier.models:
            return np.array([])
        
        # ACCURACY PRIORITY: Use all models, not just top 3
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, model in self.classifier.models.items():
            if model:
                try:
                    pred = model.predict(X)
                    weight = self.classifier.ensemble_weights.get(name, 1.0)
                    ensemble_pred += weight * pred
                    total_weight += weight
                except Exception:
                    continue
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return np.round(ensemble_pred).astype(int)
    
    def _fast_ensemble_proba(self, X):
        """Optimized ensemble probability prediction - RESTORED FULL ENSEMBLE"""
        if not self.classifier.models:
            return np.array([])
        
        # ACCURACY PRIORITY: Use all models with probability support
        prob_models = [(name, model) for name, model in self.classifier.models.items() 
                      if hasattr(model, 'predict_proba')]
        
        if not prob_models:
            return np.array([])
        
        ensemble_proba = None
        total_weight = 0
        
        for name, model in prob_models:  # Use all models, not just top 3
            try:
                proba = model.predict_proba(X)
                weight = self.classifier.ensemble_weights.get(name, 1.0)
                if ensemble_proba is None:
                    ensemble_proba = weight * proba
                else:
                    ensemble_proba += weight * proba
                total_weight += weight
            except Exception:
                continue
        
        if total_weight > 0 and ensemble_proba is not None:
            ensemble_proba /= total_weight
        
        return ensemble_proba
    
    def _should_use_heuristic(self, ml_label: str, block: Dict, doc_type: str) -> bool:
        """Smart decision on when to use heuristics over ML - IMPROVED SELECTIVITY"""
        
        text = block.get('text', '').strip().lower()
        
        # ACCURACY: More selective heuristic usage - only for clear cases
        # Reduced from always using heuristics for titles
        if ml_label == 'Title':
            # Only use heuristics if ML confidence is very low or text has strong title patterns
            font_size = block.get('avg_font_size', 12)
            word_count = len(text.split())
            if (font_size < 14 and word_count > 15):  # Probably not a title
                return True
        
        # ACCURACY: Use heuristics for very obvious patterns only
        strong_obvious_patterns = {
            'academic': ['abstract', 'introduction', 'conclusion', 'references', 'methodology'],
            'technical': ['overview', 'installation', 'configuration', 'api reference'],
            'business': ['executive summary', 'recommendations', 'next steps', 'conclusion'],
            'form': ['personal information', 'contact details', 'declaration', 'signature']
        }
        
        # Only use heuristics if text exactly matches these strong patterns
        if any(text.strip().lower() == pattern for pattern in strong_obvious_patterns.get(doc_type, [])):
            return True
        
        # ACCURACY: Use heuristics for very clear numbered sections only
        if (text.startswith(('1.', '2.', '3.')) and 
            len(text.split()) <= 5 and  # Very short numbered items
            any(char.isalpha() for char in text)):  # Has actual content
            return True
        
        return False
    
    def _apply_ultra_heuristics(self, blocks: List[Dict]) -> List[str]:
        """Apply heuristic predictions with optimized pattern recognition"""
        
        predictions = []
        
        # PERFORMANCE OPTIMIZATION: Pre-compute document-level statistics once
        doc_stats = self._compute_document_statistics(blocks)
        
        # ACCURACY OPTIMIZATION: Detect document type early for specialized processing
        doc_type = self._detect_document_type(blocks, doc_stats)
        
        for i, block in enumerate(blocks):
            text = block.get('text', '').strip()
            
            if not text:
                predictions.append('Paragraph')
                continue
            
            # OPTIMIZATION: Use document-type-aware classification
            prediction = self._classify_block_optimized(block, blocks, i, doc_stats, doc_type)
            predictions.append(prediction)
        
        return predictions
    
    def _compute_document_statistics(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Pre-compute document-level statistics for optimization"""
        
        font_sizes = [b.get('avg_font_size', 12) for b in blocks if b.get('avg_font_size')]
        
        return {
            'avg_font_size': np.mean(font_sizes) if font_sizes else 12,
            'max_font_size': max(font_sizes) if font_sizes else 12,
            'min_font_size': min(font_sizes) if font_sizes else 12,
            'unique_font_sizes': sorted(set(font_sizes), reverse=True) if font_sizes else [12],
            'total_blocks': len(blocks),
            'all_text_lower': ' '.join(b.get('text', '').lower() for b in blocks),
            'has_numbers': any(any(c.isdigit() for c in b.get('text', '')) for b in blocks),
            'avg_text_length': np.mean([len(b.get('text', '')) for b in blocks])
        }
    
    def _detect_document_type(self, blocks: List[Dict], doc_stats: Dict[str, Any]) -> str:
        """Fast document type detection for specialized processing"""
        
        all_text = doc_stats['all_text_lower']
        
        # Academic/Research papers
        if any(term in all_text for term in ['abstract', 'introduction', 'methodology', 'references', 'conclusion']):
            return 'academic'
        
        # Technical/Manual documents
        if any(term in all_text for term in ['api', 'implementation', 'configuration', 'installation', 'documentation']):
            return 'technical'
        
        # Invitations/Flyers
        if any(term in all_text for term in ['invitation', 'party', 'rsvp', 'join us', 'address:', 'date:']):
            return 'invitation'
        
        # Forms/Applications
        if any(term in all_text for term in ['application', 'form', 'please fill', 'submit', 'signature']):
            return 'form'
        
        # Business/Official documents
        if any(term in all_text for term in ['company', 'organization', 'department', 'office', 'proposal']):
            return 'business'
        
        return 'general'
    
    def _classify_block_optimized(self, block: Dict, all_blocks: List[Dict], index: int, 
                                 doc_stats: Dict[str, Any], doc_type: str) -> str:
        """Optimized block classification with document-type awareness"""
        
        text = block.get('text', '').strip()
        font_size = block.get('avg_font_size', 12)
        
        # OPTIMIZATION: Fast title detection using pre-computed stats
        if self._is_title_optimized(block, index, doc_stats, doc_type):
            return 'Title'
        
        # OPTIMIZATION: Fast heading detection with type-specific patterns
        heading_level = self._get_heading_level_optimized(block, doc_stats, doc_type)
        if heading_level > 0:
            return f'H{heading_level}'
        
        return 'Paragraph'
    
    def _is_title_optimized(self, block: Dict, index: int, doc_stats: Dict[str, Any], doc_type: str) -> bool:
        """Optimized title detection with document-type awareness"""
        
        text = block.get('text', '').strip()
        if not text or len(text) < 5:
            return False
        
        text_lower = text.lower()
        font_size = block.get('avg_font_size', 12)
        word_count = len(text.split())
        
        # ACCURACY BOOST: Strong title patterns by document type
        strong_patterns = {
            'academic': ['abstract', 'a study of', 'analysis of', 'research on', 'investigation'],
            'technical': ['user guide', 'documentation', 'api reference', 'manual', 'specification'],
            'business': ['proposal for', 'report on', 'overview of', 'strategy', 'plan'],
            'form': ['application for', 'request form', 'enrollment', 'registration'],
            'invitation': ['you are invited', 'join us', 'celebration', 'party invitation'],
            'general': ['overview', 'summary', 'introduction', 'guide to']
        }
        
        if any(pattern in text_lower for pattern in strong_patterns.get(doc_type, [])):
            return True
        
        # PERFORMANCE: Quick position-based filtering
        if index > doc_stats['total_blocks'] * 0.3:  # Not in first 30%
            return False
        
        # ACCURACY: Font size analysis with document type consideration - IMPROVED THRESHOLDS
        font_threshold = {
            'academic': 1.3,      # Increased from 1.2
            'technical': 1.35,    # Increased from 1.25
            'business': 1.4,      # Increased from 1.3
            'form': 1.25,         # Increased from 1.15
            'invitation': 1.5,    # Increased from 1.4
            'general': 1.3        # Increased from 1.25
        }.get(doc_type, 1.3)
        
        if font_size > doc_stats['avg_font_size'] * font_threshold:
            # ACCURACY: Length and content validation - STRICTER CRITERIA
            if (4 <= word_count <= 15 and  # Stricter word count range
                not text.endswith(':') and
                not text.endswith('.') and  # Titles usually don't end with periods
                not any(avoid in text_lower for avoid in ['page', 'figure', 'table', 'section', 'paragraph']) and
                text[0].isupper()):  # Titles should start with capital letter
                return True
        
        # ACCURACY: First block special handling
        if index == 0 and font_size >= doc_stats['avg_font_size'] * 1.1:
            if word_count >= 3 and not text_lower.startswith(('the ', 'a ', 'an ')):
                return True
        
        return False
    
    def _get_heading_level_optimized(self, block: Dict, doc_stats: Dict[str, Any], doc_type: str) -> int:
        """Optimized heading level detection with document-type patterns"""
        
        text = block.get('text', '').strip()
        if not text:
            return 0
        
        font_size = block.get('avg_font_size', 12)
        text_lower = text.lower()
        word_count = len(text.split())
        
        # PERFORMANCE: Quick elimination of obvious non-headings
        if (word_count > 20 or 
            text.endswith('.') and word_count > 8 or
            len(text) > 200):
            return 0
        
        # ACCURACY: Document-type specific heading patterns
        heading_patterns = {
            'academic': {
                'strong': ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references', 'discussion'],
                'numbered': r'^\d+\.?\s+[A-Z]',
                'level_threshold': 1.15    
            },
            'technical': {
                'strong': ['overview', 'installation', 'configuration', 'api', 'examples', 'troubleshooting', 'requirements'],
                'numbered': r'^\d+\.?\d*\.?\s+[A-Z]',
                'level_threshold': 1.12   
            },
            'business': {
                'strong': ['executive summary', 'background', 'objectives', 'recommendations', 'next steps', 'conclusion'],
                'numbered': r'^\d+\.\s+[A-Z]',
                'level_threshold': 1.2    
            },
            'form': {
                'strong': ['personal information', 'contact details', 'education', 'experience', 'declaration'],
                'numbered': r'^\d+\.\s+',
                'level_threshold': 1.1   
            }
        }
        
        patterns = heading_patterns.get(doc_type, heading_patterns['technical'])
        
        # ACCURACY: Strong semantic patterns
        if any(pattern in text_lower for pattern in patterns['strong']):
            if font_size >= doc_stats['avg_font_size'] * patterns['level_threshold']:
                return self._determine_level_by_font(font_size, doc_stats)
        
        # ACCURACY: Numbered headings
        import re
        if re.match(patterns['numbered'], text):
            return self._determine_level_by_font(font_size, doc_stats)
        
        # ACCURACY: Font-based detection with improved thresholds
        if font_size > doc_stats['avg_font_size'] * patterns['level_threshold']:
            # Additional pattern checks
            if any([
                text.isupper() and word_count <= 8,
                text.endswith(':') and word_count <= 6,
                block.get('flags', 0) & 16,  # Bold flag
                text.startswith(('Chapter', 'Section', 'Part', 'Appendix'))
            ]):
                return self._determine_level_by_font(font_size, doc_stats)
        
        return 0
    
    def _determine_level_by_font(self, font_size: float, doc_stats: Dict[str, Any]) -> int:
        """Determine heading level based on font size distribution"""
        
        unique_sizes = doc_stats['unique_font_sizes']
        
        if len(unique_sizes) >= 3:
            if font_size >= unique_sizes[0]:
                return 1
            elif font_size >= unique_sizes[1]:
                return 2
            else:
                return 3
        elif font_size >= doc_stats['avg_font_size'] * 1.3:
            return 1
        elif font_size >= doc_stats['avg_font_size'] * 1.15:
            return 2
        else:
            return 3
    
    def _get_heuristic_label_fast(self, block: Dict, doc_stats: Dict[str, Any], doc_type: str) -> str:
        """Fast heuristic labeling with document-type awareness"""
        
        text = block.get('text', '').strip()
        if not text:
            return 'Paragraph'
        
        font_size = block.get('avg_font_size', 12)
        text_lower = text.lower()
        word_count = len(text.split())
        
        # PERFORMANCE: Quick title patterns
        title_indicators = {
            'academic': ['abstract', 'a study of', 'analysis of'],
            'technical': ['user guide', 'documentation', 'api reference'],
            'business': ['executive summary', 'proposal', 'overview'],
            'form': ['application for', 'request form'],
            'invitation': ['you are invited', 'join us'],
            'general': ['introduction', 'overview']
        }
        
        if any(pattern in text_lower for pattern in title_indicators.get(doc_type, [])):
            if font_size >= doc_stats['avg_font_size'] * 1.1:
                return 'Title'
        
        # PERFORMANCE: Quick heading detection
        if font_size >= doc_stats['avg_font_size'] * 1.08:
            # Strong heading patterns
            if any([
                text.isupper() and word_count <= 8,
                text.startswith(('1.', '2.', '3.', 'Chapter', 'Section')),
                text.endswith(':') and word_count <= 6,
                block.get('flags', 0) & 16  # Bold
            ]):
                # Determine level quickly
                if font_size >= doc_stats['avg_font_size'] * 1.3:
                    return 'H1'
                elif font_size >= doc_stats['avg_font_size'] * 1.15:
                    return 'H2'
                else:
                    return 'H3'
        
        # PERFORMANCE: Title detection for large fonts
        if (font_size >= doc_stats['avg_font_size'] * 1.2 and 
            3 <= word_count <= 15 and
            not text.endswith('.') and
            not any(avoid in text_lower for avoid in ['page', 'figure', 'table'])):
            return 'Title'
        
        return 'Paragraph'
    
    def _generate_structured_output(self, predictions: List[str], blocks: List[Dict]) -> Dict[str, Any]:
        """Generate optimized structured JSON output with enhanced accuracy"""
        
        title = ""
        outline = []
        
        # PERFORMANCE OPTIMIZATION: Pre-compute document-level statistics once
        doc_stats = self._compute_document_statistics(blocks)
        doc_type = self._detect_document_type(blocks, doc_stats)
        
        # ACCURACY: Multi-strategy title extraction - IMPROVED WITH ORIGINAL PATTERNS
        title_candidates = []
        
        # Strategy 1: ML predictions
        title_indices = [i for i, pred in enumerate(predictions) if pred == 'Title']
        for idx in title_indices:
            title_candidates.append((blocks[idx].get('text', '').strip(), idx, 'ml', 0.8))
        
        # Strategy 2: Original heuristic patterns (high-confidence)
        for i, block in enumerate(blocks[:8]):  # Check first 8 blocks
            text = block.get('text', '').strip()
            font_size = block.get('avg_font_size', 12)
            
            # Original title detection logic - for comparison with 52.7% accuracy system
            if text and len(text) >= 5:
                word_count = len(text.split())
                
                # Strong title indicators from original system
                if (font_size >= doc_stats['avg_font_size'] * 1.2 and 
                    3 <= word_count <= 12 and  # Shorter titles preferred
                    i <= 3 and  # Must be in first 4 blocks
                    not text.lower().startswith(('the ', 'a ', 'an ')) and
                    text[0].isupper() and
                    not text.endswith('.') and
                    not any(avoid in text.lower() for avoid in ['page', 'figure', 'table'])):
                    
                    confidence = 0.9 if i == 0 else (0.8 if i == 1 else 0.7)
                    title_candidates.append((text, i, 'original_heuristic', confidence))
        
        # Strategy 3: Document-specific title patterns
        for i, block in enumerate(blocks[:5]):  # Check first 5 blocks only
            text = block.get('text', '').strip()
            if self._is_title_optimized(block, i, doc_stats, doc_type):
                confidence = 0.9 if i == 0 else 0.7  # Higher confidence for first block
                title_candidates.append((text, i, 'doc_specific', confidence))
        
        # ACCURACY: Choose best title candidate with preference for shorter, earlier titles
        if title_candidates:
            # Sort by confidence, then by position, then by length (shorter preferred)
            title_candidates.sort(key=lambda x: (-x[3], x[1], len(x[0])))
            title = title_candidates[0][0]
        
        # ACCURACY: Enhanced heading extraction with post-processing - RESTORED ORIGINAL PATTERNS
        heading_candidates = []
        
        # Strategy 1: ML predictions
        for i, (pred, block) in enumerate(zip(predictions, blocks)):
            if pred.startswith('H'):
                try:
                    level = int(pred[1]) if len(pred) > 1 and pred[1].isdigit() else 1
                    text = block.get('text', '').strip()
                    
                    # ACCURACY: Validate heading quality
                    if self._is_valid_heading(text, level, doc_type):
                        confidence = 0.8 if self.classifier.trained else 0.6
                        heading_candidates.append({
                            'text': text,
                            'level': level,
                            'page': block.get('page', 1),
                            'confidence': confidence,
                            'position': i
                        })
                except:
                    continue
        
        # Strategy 2: Original heuristic detection for missing headings
        for i, block in enumerate(blocks):
            text = block.get('text', '').strip()
            if text and len(text) >= 3:
                font_size = block.get('avg_font_size', 12)
                word_count = len(text.split())
                
                # Original heading patterns that achieved 52.7% accuracy
                is_heading = False
                level = 3  # default
                
                # Check for obvious heading patterns - MORE AGGRESSIVE LIKE ORIGINAL
                if (word_count <= 20 and  # Increased from 15
                    (font_size >= doc_stats['avg_font_size'] * 1.05 or  # Lower threshold
                     text.isupper() or 
                     text.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', 
                                     'Chapter', 'Section', 'Part', 'Appendix', 'Figure', 'Table')) or
                     text.endswith((':', '?')) or
                     block.get('flags', 0) & 16 or  # Bold flag
                     any(word in text.upper() for word in ['OVERVIEW', 'INTRODUCTION', 'CONCLUSION', 
                                                           'SUMMARY', 'ABSTRACT', 'REFERENCES']))):
                    
                    is_heading = True
                    if font_size >= doc_stats['avg_font_size'] * 1.25:
                        level = 1
                    elif font_size >= doc_stats['avg_font_size'] * 1.15:
                        level = 2
                    else:
                        level = 3
                
                # Add if not already detected and is valid
                if is_heading and self._is_valid_heading(text, level, doc_type):
                    # Check if not already added
                    already_exists = any(h['text'].lower() == text.lower() and abs(h['position'] - i) < 2 
                                       for h in heading_candidates)
                    if not already_exists:
                        heading_candidates.append({
                            'text': text,
                            'level': level,
                            'page': block.get('page', 1),
                            'confidence': 0.7,
                            'position': i
                        })
        
        # ACCURACY: Fallback heading detection if still none found
        if not heading_candidates:
            for i, block in enumerate(blocks):
                heading_level = self._get_heading_level_optimized(block, doc_stats, doc_type)
                if heading_level > 0:
                    text = block.get('text', '').strip()
                    if self._is_valid_heading(text, heading_level, doc_type):
                        heading_candidates.append({
                            'text': text,
                            'level': heading_level,
                            'page': block.get('page', 1),
                            'confidence': 0.6,
                            'position': i
                        })
        
        # ACCURACY: Post-process headings for consistency
        outline = self._post_process_headings(heading_candidates, doc_type)
        
        return {
            'title': title,
            'outline': outline
        }
    
    def _is_valid_heading(self, text: str, level: int, doc_type: str) -> bool:
        """Validate if text is a legitimate heading"""
        
        if not text or len(text) < 2:
            return False
        
        word_count = len(text.split())
        
        # ACCURACY: Length validation by document type - MORE PERMISSIVE
        max_words = {
            'academic': 20,        # Increased from 12
            'technical': 18,       # Increased from 10
            'business': 25,        # Increased from 15
            'form': 15,           # Increased from 8
            'invitation': 12,      # Increased from 6
            'general': 18         # Increased from 10
        }.get(doc_type, 18)
        
        if word_count > max_words:
            return False
        
        # ACCURACY: Content validation
        invalid_patterns = [
            'lorem ipsum', 'copyright', 'all rights reserved',
            'page ', 'figure ', 'table ', 'www.', '.com', '.org',
            'email:', 'tel:', 'phone:', 'fax:'
        ]
        
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in invalid_patterns):
            return False
        
        # ACCURACY: Structure validation
        if text.count('.') > 3 or text.count(',') > 2:
            return False
        
        return True
    
    def _post_process_headings(self, heading_candidates: List[Dict], doc_type: str) -> List[Dict]:
        """Post-process headings for better structure and accuracy"""
        
        if not heading_candidates:
            return []
        
        # ACCURACY: Sort by position in document
        heading_candidates.sort(key=lambda x: x.get('position', float('inf')))
        
        # ACCURACY: Level consistency adjustment
        processed_headings = []
        prev_level = 0
        
        for heading in heading_candidates:
            current_level = heading['level']
            
            # ACCURACY: Prevent level jumps greater than 1
            if prev_level > 0 and current_level > prev_level + 1:
                current_level = prev_level + 1
            
            # ACCURACY: Remove duplicate headings
            if processed_headings:
                last_heading = processed_headings[-1]
                # Safe position comparison - use default if key missing
                last_pos = last_heading.get('position', float('inf'))
                curr_pos = heading.get('position', float('inf'))
                if (last_heading['text'].lower() == heading['text'].lower() or
                    (last_pos != float('inf') and curr_pos != float('inf') and abs(last_pos - curr_pos) < 3)):
                    continue
            
            processed_headings.append({
                'text': heading['text'],
                'level': current_level,
                'page': heading['page'],
                'confidence': heading['confidence'],
                'position': heading.get('position', len(processed_headings))
            })
            
            prev_level = current_level
        
        # ACCURACY: Final validation - remove obvious false positives
        final_headings = []
        for heading in processed_headings:
            if len(heading['text'].split()) >= 1:  # At least one word
                final_headings.append(heading)
        
        return final_headings
    
    def train_model(self, training_csv_path: str) -> float:
        """Train the model"""
        
        if not os.path.exists(training_csv_path):
            raise FileNotFoundError(f"Training data not found: {training_csv_path}")
        
        # Load training data
        df = pd.read_csv(training_csv_path)
        logger.info(f"Loaded training data: {len(df)} samples")
        
        # Train classifier
        accuracy = self.classifier.train_advanced_models(df)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.classifier.save_model(self.model_path)
        
        return accuracy

