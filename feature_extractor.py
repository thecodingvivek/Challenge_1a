#!/usr/bin/env python3
"""
Feature Extractor for PDF Document Structure Detection
Extracts visual, spatial, and linguistic features from PDF text blocks
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

class PDFFeatureExtractor:
    """Extract features from PDF text blocks for heading classification"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text blocks from PDF with detailed formatting info"""
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
                        if not block_text:
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
    
    def compute_visual_features(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute visual and spatial features for each block"""
        if not blocks:
            return []
            
        # Calculate page-level statistics
        page_stats = {}
        for block in blocks:
            page_num = block["page"]
            if page_num not in page_stats:
                page_stats[page_num] = {
                    "font_sizes": [],
                    "y_positions": [],
                    "blocks": []
                }
            page_stats[page_num]["font_sizes"].append(block["font_size"])
            page_stats[page_num]["y_positions"].append(block["bbox"][1])
            page_stats[page_num]["blocks"].append(block)
        
        # Calculate global statistics
        all_font_sizes = [block["font_size"] for block in blocks]
        avg_font_size = np.mean(all_font_sizes)
        
        enriched_blocks = []
        
        for i, block in enumerate(blocks):
            features = block.copy()
            
            # Basic visual features
            features["font_size"] = block["font_size"]
            features["relative_size"] = block["font_size"] / avg_font_size
            features["font_name"] = block["font_name"]
            features["is_bold"] = bool(block["font_flags"] & 2**4)  # Bold flag
            
            # Spatial features
            bbox = block["bbox"]
            features["x_offset"] = bbox[0]
            features["y_pos_norm"] = bbox[1] / block["page_height"]
            features["text_len"] = len(block["text"])
            features["block_width"] = bbox[2] - bbox[0]
            features["block_height"] = bbox[3] - bbox[1]
            
            # Block spacing (distance from previous block on same page)
            page_blocks = page_stats[block["page"]]["blocks"]
            page_blocks_sorted = sorted(page_blocks, key=lambda x: x["bbox"][1])
            
            block_spacing = 0
            for j, pb in enumerate(page_blocks_sorted):
                if pb["bbox"] == block["bbox"] and j > 0:
                    prev_block = page_blocks_sorted[j-1]
                    block_spacing = block["bbox"][1] - prev_block["bbox"][3]
                    break
            
            features["block_spacing"] = max(0, block_spacing)
            
            # Alignment features
            center_x = (bbox[0] + bbox[2]) / 2
            page_center = block["page_width"] / 2
            left_margin = bbox[0]
            
            if abs(center_x - page_center) < 50:
                features["alignment_type"] = "Center"
                features["centered_text"] = True
            elif left_margin < 100:
                features["alignment_type"] = "Left"
                features["centered_text"] = False
            else:
                features["alignment_type"] = "Right"
                features["centered_text"] = False
            
            # Position on page
            page_height = block["page_height"]
            if bbox[1] < page_height * 0.3:
                features["line_position"] = "Top"
            elif bbox[1] > page_height * 0.7:
                features["line_position"] = "Bottom"
            else:
                features["line_position"] = "Middle"
            
            # Isolation (whitespace before/after)
            features["isolated_block"] = block_spacing > 20
            
            enriched_blocks.append(features)
        
        return enriched_blocks
    
    def compute_linguistic_features(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute linguistic and NLP features"""
        enriched_blocks = []
        
        for block in blocks:
            features = block.copy()
            text = block["text"]
            
            # Basic text statistics
            features["capital_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            features["title_case_ratio"] = sum(1 for word in text.split() if word.istitle()) / max(len(text.split()), 1)
            features["has_terminal_punct"] = text.strip().endswith(('.', '!', '?', ':', ';'))
            
            # Stopword analysis
            try:
                tokens = word_tokenize(text.lower())
                stop_count = sum(1 for token in tokens if token in self.stop_words)
                features["stopword_ratio"] = stop_count / max(len(tokens), 1)
            except:
                features["stopword_ratio"] = 0
            
            # N-gram score (heuristic for title-like text)
            words = text.split()
            word_count = len(words)
            
            if word_count <= 10 and word_count > 0:
                # Short text bonus
                ngram_score = 1.0
            elif word_count <= 20:
                ngram_score = 0.7
            else:
                ngram_score = 0.3
                
            # Bonus for title-case words
            if features["title_case_ratio"] > 0.5:
                ngram_score += 0.3
                
            features["ngram_score"] = min(ngram_score, 1.0)
            
            # Additional text features
            features["word_count"] = word_count
            features["avg_word_length"] = sum(len(word) for word in words) / max(word_count, 1)
            features["has_numbers"] = bool(re.search(r'\d', text))
            features["special_char_ratio"] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            
            # Proximity to previous heading (placeholder)
            features["proximity_to_prev_heading"] = None
            
            enriched_blocks.append(features)
        
        return enriched_blocks
    
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
            if "title" in labels:
                label_map[labels["title"]] = "Title"
            
            # Add headings
            if "outline" in labels:
                for item in labels["outline"]:
                    label_map[item["text"]] = item["level"]
            
            # Match blocks with labels
            labeled_blocks = []
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
                
                labeled_blocks.append(block)
            
            return labeled_blocks
            
        except Exception as e:
            logger.error(f"Error matching labels from {label_file}: {e}")
            for block in blocks:
                block["label"] = None
            return blocks
    
    def process_pdf(self, pdf_path: str, label_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a single PDF and return feature-enriched blocks"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text blocks
        blocks = self.extract_text_blocks(pdf_path)
        if not blocks:
            logger.warning(f"No text blocks found in {pdf_path}")
            return []
        
        # Compute visual features
        blocks = self.compute_visual_features(blocks)
        
        # Compute linguistic features
        blocks = self.compute_linguistic_features(blocks)
        
        # Match with labels if available
        if label_path:
            blocks = self.match_with_labels(blocks, label_path)
        
        return blocks
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Process all PDFs in a directory and return combined DataFrame"""
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
        
        # Select relevant features for training
        feature_columns = [
            "font_size", "relative_size", "font_name", "is_bold",
            "x_offset", "y_pos_norm", "text_len", "block_spacing",
            "capital_ratio", "title_case_ratio", "stopword_ratio",
            "has_terminal_punct", "line_position", "ngram_score",
            "isolated_block", "alignment_type", "centered_text",
            "word_count", "avg_word_length", "has_numbers", "special_char_ratio",
            "page", "text"
        ]
        
        if "label" in df.columns:
            feature_columns.append("label")
        
        # Keep only available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        df = df[available_columns]
        
        pdf_count = len(list(input_path.glob('*.pdf')))
        logger.info(f"Processed {len(df)} blocks from {pdf_count} PDFs")
        
        return df


def main():
    """Example usage"""
    extractor = PDFFeatureExtractor()
    
    # Process training data
    df = extractor.process_directory("input/", "output/")
    
    if not df.empty:
        print(f"Extracted features from {len(df)} text blocks")
        print(f"Columns: {list(df.columns)}")
        
        if "label" in df.columns:
            print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Save for inspection
        df.to_csv("extracted_features.csv", index=False)
        print("Features saved to extracted_features.csv")
    else:
        print("No features extracted")


if __name__ == "__main__":
    main()