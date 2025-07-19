#!/usr/bin/env python3
"""
Enhanced Feature Extractor for PDF Document Structure Detection
Combines visual, spatial, linguistic, and contextual features with heuristic ranking
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFFeatureExtractor:
    """Enhanced PDF Feature Extractor with contextual analysis and heuristic ranking"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Common title/heading patterns
        self.title_patterns = [
            r'^(chapter|section|part|appendix)\s+\d+',
            r'^\d+\.\s+',
            r'^\d+\.\d+\s+',
            r'^\d+\.\d+\.\d+\s+',
            r'^[ivxlcdm]+\.\s+',  # Roman numerals
            r'^[a-z]\)\s+',
            r'^[A-Z]\)\s+',
            r'^\([a-z]\)\s+',
            r'^\([A-Z]\)\s+',
            r'^\([0-9]+\)\s+',
            r'^bullet\s+',
            r'^•\s+',
            r'^-\s+',
            r'^\*\s+',
        ]
        
        # Common heading keywords
        self.heading_keywords = {
            'introduction', 'conclusion', 'abstract', 'summary', 'overview',
            'background', 'methodology', 'method', 'approach', 'results',
            'discussion', 'analysis', 'findings', 'recommendations', 'future',
            'references', 'bibliography', 'appendix', 'acknowledgments',
            'objectives', 'goals', 'scope', 'purpose', 'requirements',
            'specifications', 'implementation', 'evaluation', 'testing',
            'validation', 'review', 'related', 'work', 'literature'
        }
        
        # Title indicators
        self.title_indicators = {
            'report', 'proposal', 'document', 'manual', 'guide', 'handbook',
            'specification', 'standard', 'policy', 'procedure', 'protocol',
            'framework', 'strategy', 'plan', 'analysis', 'study', 'research',
            'review', 'assessment', 'evaluation', 'survey', 'white', 'paper'
        }
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text blocks with enhanced metadata"""
        blocks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_dict = page.get_text("dict")
                page_height = page.rect.height
                page_width = page.rect.width
                
                # Extract blocks from page
                for block in page_dict.get("blocks", []):
                    if "lines" in block:  # Text block
                        block_text = ""
                        block_bbox = block["bbox"]
                        font_info = []
                        
                        # Collect text and font info from all lines
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                                font_info.append({
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"]
                                })
                            block_text += line_text + " "
                        
                        block_text = block_text.strip()
                        if not block_text or len(block_text) < 3:
                            continue
                            
                        # Get dominant font info
                        if font_info:
                            font_sizes = [f["size"] for f in font_info]
                            font_names = [f["font"] for f in font_info]
                            font_flags = [f["flags"] for f in font_info]
                            
                            dominant_size = max(set(font_sizes), key=font_sizes.count)
                            dominant_font = max(set(font_names), key=font_names.count)
                            dominant_flags = max(set(font_flags), key=font_flags.count)
                        else:
                            dominant_size = 12
                            dominant_font = "default"
                            dominant_flags = 0
                        
                        block_info = {
                            "text": block_text,
                            "page": page_num + 1,
                            "bbox": block_bbox,
                            "font_size": dominant_size,
                            "font_name": dominant_font,
                            "font_flags": dominant_flags,
                            "page_height": page_height,
                            "page_width": page_width
                        }
                        
                        blocks.append(block_info)
                        
            doc.close()
            return blocks
            
        except Exception as e:
            logger.error(f"Error extracting blocks from {pdf_path}: {e}")
            return []
    
    def compute_contextual_features(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute contextual features by analyzing document structure"""
        if not blocks:
            return []
        
        # Sort blocks by page and position
        sorted_blocks = sorted(blocks, key=lambda x: (x["page"], x["bbox"][1]))
        
        # Compute global statistics
        all_font_sizes = [block["font_size"] for block in blocks]
        font_size_stats = {
            'mean': np.mean(all_font_sizes),
            'std': np.std(all_font_sizes),
            'min': np.min(all_font_sizes),
            'max': np.max(all_font_sizes)
        }
        
        # Analyze font size distribution
        font_size_counts = Counter(all_font_sizes)
        most_common_size = font_size_counts.most_common(1)[0][0]
        
        # Compute text length statistics
        text_lengths = [len(block["text"]) for block in blocks]
        text_length_stats = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'median': np.median(text_lengths)
        }
        
        enriched_blocks = []
        
        for i, block in enumerate(sorted_blocks):
            features = block.copy()
            
            # Basic contextual features
            features["block_index"] = i
            features["total_blocks"] = len(blocks)
            features["relative_position"] = i / len(blocks)
            
            # Font size analysis
            features["font_size_z_score"] = (block["font_size"] - font_size_stats['mean']) / max(font_size_stats['std'], 1)
            features["font_size_percentile"] = sum(1 for size in all_font_sizes if size <= block["font_size"]) / len(all_font_sizes)
            features["is_largest_font"] = block["font_size"] == font_size_stats['max']
            features["is_common_font_size"] = block["font_size"] == most_common_size
            features["font_size_deviation"] = abs(block["font_size"] - most_common_size)
            
            # Position analysis
            features["is_first_page"] = block["page"] == 1
            features["is_last_page"] = block["page"] == max(b["page"] for b in blocks)
            features["page_position_normalized"] = (block["bbox"][1] / block["page_height"])
            
            # Surrounding context analysis
            prev_block = sorted_blocks[i-1] if i > 0 else None
            next_block = sorted_blocks[i+1] if i < len(sorted_blocks) - 1 else None
            
            if prev_block:
                features["prev_font_size_diff"] = block["font_size"] - prev_block["font_size"]
                features["prev_text_length"] = len(prev_block["text"])
                features["prev_page_diff"] = block["page"] - prev_block["page"]
            else:
                features["prev_font_size_diff"] = 0
                features["prev_text_length"] = 0
                features["prev_page_diff"] = 0
            
            if next_block:
                features["next_font_size_diff"] = next_block["font_size"] - block["font_size"]
                features["next_text_length"] = len(next_block["text"])
                features["next_page_diff"] = next_block["page"] - block["page"]
            else:
                features["next_font_size_diff"] = 0
                features["next_text_length"] = 0
                features["next_page_diff"] = 0
            
            # Text length analysis
            features["text_length_z_score"] = (len(block["text"]) - text_length_stats['mean']) / max(text_length_stats['std'], 1)
            features["is_short_text"] = len(block["text"]) < text_length_stats['mean'] - text_length_stats['std']
            features["is_medium_text"] = (text_length_stats['mean'] - text_length_stats['std']) <= len(block["text"]) <= (text_length_stats['mean'] + text_length_stats['std'])
            features["is_long_text"] = len(block["text"]) > text_length_stats['mean'] + text_length_stats['std']
            
            enriched_blocks.append(features)
        
        return enriched_blocks
    
    def compute_heuristic_scores(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute heuristic scores for heading likelihood"""
        
        for block in blocks:
            text = block["text"].lower().strip()
            original_text = block["text"].strip()
            
            # Initialize scores
            title_score = 0
            h1_score = 0
            h2_score = 0
            h3_score = 0
            
            # Pattern matching scores
            for pattern in self.title_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    if pattern in [r'^\d+\.\s+', r'^[ivxlcdm]+\.\s+']:
                        h1_score += 3
                    elif pattern in [r'^\d+\.\d+\s+']:
                        h2_score += 3
                    elif pattern in [r'^\d+\.\d+\.\d+\s+']:
                        h3_score += 3
                    else:
                        h1_score += 1
            
            # Keyword matching
            text_words = set(re.findall(r'\b\w+\b', text))
            
            # Title keyword matching
            title_keyword_count = len(text_words.intersection(self.title_indicators))
            if title_keyword_count > 0:
                title_score += title_keyword_count * 2
            
            # Heading keyword matching
            heading_keyword_count = len(text_words.intersection(self.heading_keywords))
            if heading_keyword_count > 0:
                h1_score += heading_keyword_count
                h2_score += heading_keyword_count * 0.5
            
            # Position-based scoring
            if block["page"] == 1:
                title_score += 3
                h1_score += 2
            
            if block["page_position_normalized"] < 0.2:  # Top of page
                title_score += 2
                h1_score += 1
            
            # Font size based scoring
            if block["font_size_z_score"] > 2:
                title_score += 4
                h1_score += 3
            elif block["font_size_z_score"] > 1:
                h1_score += 2
                h2_score += 1
            elif block["font_size_z_score"] > 0:
                h2_score += 1
                h3_score += 0.5
            
            # Text characteristics
            if block["is_short_text"]:
                title_score += 2
                h1_score += 2
                h2_score += 1
                h3_score += 1
            
            # Capitalization scoring
            if original_text.isupper():
                title_score += 2
                h1_score += 1
            elif original_text.istitle():
                title_score += 1
                h1_score += 1
                h2_score += 1
            
            # Centering and formatting
            if block.get("centered_text", False):
                title_score += 2
                h1_score += 1
            
            if block.get("is_bold", False):
                title_score += 1
                h1_score += 1
                h2_score += 0.5
            
            # Assign scores
            block["title_score"] = title_score
            block["h1_score"] = h1_score
            block["h2_score"] = h2_score
            block["h3_score"] = h3_score
            
            # Compute overall heading likelihood
            block["heading_likelihood"] = max(title_score, h1_score, h2_score, h3_score)
            
            # Determine most likely class based on heuristics
            scores = {
                "Title": title_score,
                "H1": h1_score,
                "H2": h2_score,
                "H3": h3_score,
                "Paragraph": 0
            }
            
            block["heuristic_class"] = max(scores, key=scores.get)
            block["heuristic_confidence"] = max(scores.values()) / max(sum(scores.values()), 1)
        
        return blocks
    
    def compute_enhanced_features(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute enhanced visual and linguistic features"""
        
        for block in blocks:
            text = block["text"]
            
            # Enhanced visual features
            bbox = block["bbox"]
            block["block_width"] = bbox[2] - bbox[0]
            block["block_height"] = bbox[3] - bbox[1]
            block["aspect_ratio"] = block["block_width"] / max(block["block_height"], 1)
            
            # Enhanced linguistic features
            sentences = sent_tokenize(text)
            block["sentence_count"] = len(sentences)
            block["avg_sentence_length"] = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            
            # Word analysis
            words = text.split()
            if words:
                block["word_count"] = len(words)
                block["avg_word_length"] = sum(len(word) for word in words) / len(words)
                block["unique_word_ratio"] = len(set(words)) / len(words)
            else:
                block["word_count"] = 0
                block["avg_word_length"] = 0
                block["unique_word_ratio"] = 0
            
            # Character analysis
            block["char_count"] = len(text)
            block["alpha_ratio"] = sum(c.isalpha() for c in text) / max(len(text), 1)
            block["digit_ratio"] = sum(c.isdigit() for c in text) / max(len(text), 1)
            block["punct_ratio"] = sum(c in '.,!?;:' for c in text) / max(len(text), 1)
            block["space_ratio"] = sum(c.isspace() for c in text) / max(len(text), 1)
            
            # N-gram analysis
            try:
                tokens = word_tokenize(text.lower())
                if tokens:
                    stop_count = sum(1 for token in tokens if token in self.stop_words)
                    block["stopword_ratio"] = stop_count / len(tokens)
                    
                    # Bigram analysis
                    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
                    if bigrams:
                        bigram_counts = Counter(bigrams)
                        block["bigram_diversity"] = len(bigram_counts) / len(bigrams)
                    else:
                        block["bigram_diversity"] = 0
                else:
                    block["stopword_ratio"] = 0
                    block["bigram_diversity"] = 0
            except:
                block["stopword_ratio"] = 0
                block["bigram_diversity"] = 0
            
            # Formatting features
            block["starts_with_number"] = bool(re.match(r'^\d', text))
            block["ends_with_punct"] = text.strip().endswith(('.', '!', '?', ':', ';'))
            block["has_colon"] = ':' in text
            block["parentheses_count"] = text.count('(') + text.count(')')
            block["quote_count"] = text.count('"') + text.count("'")
            
            # Readability metrics (simplified)
            if block["sentence_count"] > 0 and block["word_count"] > 0:
                # Flesch reading ease approximation
                avg_words_per_sentence = block["word_count"] / block["sentence_count"]
                avg_syllables_per_word = block["avg_word_length"] * 0.5  # Rough approximation
                
                flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
                block["flesch_score"] = max(0, min(100, flesch_score))
            else:
                block["flesch_score"] = 50  # Neutral
        
        return blocks
    
    def rank_heading_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank blocks by heading likelihood and assign preliminary labels"""
        
        # Sort blocks by heading likelihood
        sorted_blocks = sorted(blocks, key=lambda x: x["heading_likelihood"], reverse=True)
        
        # Assign ranking
        for i, block in enumerate(sorted_blocks):
            block["heading_rank"] = i + 1
            block["heading_percentile"] = (len(sorted_blocks) - i) / len(sorted_blocks)
        
        # Determine preliminary labels based on ranking and scores
        for block in blocks:
            if block["heading_likelihood"] > 3:
                # High likelihood - use heuristic classification
                block["preliminary_label"] = block["heuristic_class"]
            elif block["heading_likelihood"] > 1:
                # Medium likelihood - likely a heading but unsure of level
                if block["font_size_z_score"] > 1:
                    block["preliminary_label"] = "H1"
                elif block["font_size_z_score"] > 0:
                    block["preliminary_label"] = "H2"
                else:
                    block["preliminary_label"] = "H3"
            else:
                # Low likelihood - probably paragraph
                block["preliminary_label"] = "Paragraph"
        
        return blocks
    
    def process_pdf(self, pdf_path: str, label_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a single PDF with enhanced feature extraction"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text blocks
        blocks = self.extract_text_blocks(pdf_path)
        if not blocks:
            logger.warning(f"No text blocks found in {pdf_path}")
            return []
        
        # Compute contextual features
        blocks = self.compute_contextual_features(blocks)
        
        # Compute heuristic scores
        blocks = self.compute_heuristic_scores(blocks)
        
        # Compute enhanced features
        blocks = self.compute_enhanced_features(blocks)
        
        # Rank heading candidates
        blocks = self.rank_heading_candidates(blocks)
        
        # Match with labels if available
        if label_path:
            blocks = self.match_with_labels(blocks, label_path)
        
        return blocks
    
    def match_with_labels(self, blocks: List[Dict[str, Any]], label_file: str) -> List[Dict[str, Any]]:
        """Match extracted blocks with ground truth labels"""
        if not os.path.exists(label_file):
            # No labels available (inference mode)
            for block in blocks:
                block["label"] = None
            return blocks
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # Create label mapping
            label_map = {}
            
            # Add title
            if "title" in labels and labels["title"]:
                label_map[labels["title"]] = "Title"
            
            # Add headings
            if "outline" in labels:
                for item in labels["outline"]:
                    if "text" in item:
                        label_map[item["text"]] = item["level"]
            
            # Match blocks with labels
            for block in blocks:
                text = block["text"].strip()
                
                # Try exact match first
                if text in label_map:
                    block["label"] = label_map[text]
                else:
                    # Try fuzzy matching
                    best_match = None
                    best_score = 0
                    
                    for label_text, label_type in label_map.items():
                        # Simple similarity based on common words
                        text_words = set(text.lower().split())
                        label_words = set(label_text.lower().split())
                        
                        if text_words and label_words:
                            intersection = text_words.intersection(label_words)
                            union = text_words.union(label_words)
                            similarity = len(intersection) / len(union)
                            
                            if similarity > best_score and similarity > 0.5:
                                best_score = similarity
                                best_match = label_type
                    
                    block["label"] = best_match if best_match else "Paragraph"
            
            return blocks
            
        except Exception as e:
            logger.error(f"Error matching labels from {label_file}: {e}")
            for block in blocks:
                block["label"] = None
            return blocks
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Process all PDFs in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else None
        
        all_blocks = []
        
        for pdf_file in input_path.glob("*.pdf"):
            # Find corresponding label file
            label_file = None
            if output_path:
                label_file = output_path / f"{pdf_file.stem}.json"
                if not label_file.exists():
                    label_file = None
            
            blocks = self.process_pdf(str(pdf_file), str(label_file) if label_file else None)
            
            # Add source file info
            for block in blocks:
                block["source_file"] = pdf_file.name
            
            all_blocks.extend(blocks)
        
        if not all_blocks:
            logger.warning("No blocks extracted from any PDF")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_blocks)
        
        logger.info(f"Processed {len(df)} blocks from {len(list(input_path.glob('*.pdf')))} PDFs")
        
        return df


def main():
    """Example usage"""
    extractor = EnhancedPDFFeatureExtractor()
    
    # Process training data
    df = extractor.process_directory("input/", "output/")
    
    if not df.empty:
        print(f"Extracted features from {len(df)} text blocks")
        print(f"Columns: {list(df.columns)}")
        
        if "label" in df.columns:
            print(f"Label distribution:\n{df['label'].value_counts()}")
        
        if "preliminary_label" in df.columns:
            print(f"Preliminary label distribution:\n{df['preliminary_label'].value_counts()}")
        
        # Save for inspection
        df.to_csv("enhanced_features.csv", index=False)
        print("Enhanced features saved to enhanced_features.csv")
    else:
        print("No features extracted")


if __name__ == "__main__":
    main()