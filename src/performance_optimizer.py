#!/usr/bin/env python3
"""
Performance Optimizer for PDF Structure Detection
Optimizes speed and memory usage while maintaining accuracy
"""

import os
import gc
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities for PDF processing"""
    
    def __init__(self, max_workers: int = 2, memory_limit_mb: int = 512):
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.cache_size = 128
        self.batch_size = 5  # Process PDFs in batches
        
        # Performance metrics
        self.processing_times = []
        self.memory_usage = []
        
    def optimize_pdf_processing(self, pdf_paths: List[str], 
                               processor_func: callable) -> List[Dict[str, Any]]:
        """Optimize PDF processing with batching and memory management"""
        
        results = []
        
        # Process in batches to manage memory
        for batch_start in range(0, len(pdf_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pdf_paths))
            batch_paths = pdf_paths[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}: "
                       f"{len(batch_paths)} PDFs")
            
            # Process batch
            batch_results = self._process_batch(batch_paths, processor_func)
            results.extend(batch_results)
            
            # Memory cleanup
            gc.collect()
            
            # Log progress
            progress = (batch_end / len(pdf_paths)) * 100
            logger.info(f"Progress: {progress:.1f}%")
        
        return results
    
    def _process_batch(self, pdf_paths: List[str], 
                      processor_func: callable) -> List[Dict[str, Any]]:
        """Process a batch of PDFs with optional parallelization"""
        
        results = []
        
        # For small batches or single-threaded processing
        if len(pdf_paths) <= 2 or self.max_workers == 1:
            for pdf_path in pdf_paths:
                start_time = time.time()
                
                try:
                    result = processor_func(pdf_path)
                    processing_time = time.time() - start_time
                    
                    self.processing_times.append(processing_time)
                    results.append(result)
                    
                    logger.debug(f"Processed {Path(pdf_path).name} in {processing_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
                    results.append({"title": "", "outline": []})
        
        else:
            # Parallel processing for larger batches
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(processor_func, pdf_path): pdf_path 
                    for pdf_path in pdf_paths
                }
                
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path}: {e}")
                        results.append({"title": "", "outline": []})
        
        return results
    
    @lru_cache(maxsize=128)
    def cached_feature_computation(self, text_hash: str, 
                                  feature_type: str) -> Dict[str, Any]:
        """Cache expensive feature computations"""
        # This would be used for expensive computations
        # that are repeated across similar text blocks
        pass
    
    def optimize_memory_usage(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize memory usage by removing unnecessary data"""
        
        # Remove intermediate computation results
        optimized_blocks = []
        
        for block in blocks:
            optimized_block = {
                'text': block['text'],
                'page': block['page'],
                'bbox': block['bbox'],
                'font_size': block['font_size'],
                'predicted_label': block.get('predicted_label', 'Paragraph'),
                'confidence': block.get('confidence', 0.0)
            }
            
            optimized_blocks.append(optimized_block)
        
        return optimized_blocks
    
    def fast_feature_extraction(self, blocks: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fast feature extraction with vectorized operations"""
        
        if not blocks:
            return pd.DataFrame()
        
        # Convert to numpy arrays for vectorized operations
        texts = [block['text'] for block in blocks]
        font_sizes = np.array([block['font_size'] for block in blocks])
        pages = np.array([block['page'] for block in blocks])
        
        # Vectorized computations
        text_lengths = np.array([len(text) for text in texts])
        word_counts = np.array([len(text.split()) for text in texts])
        
        # Font size statistics
        font_mean = np.mean(font_sizes)
        font_std = np.std(font_sizes)
        font_z_scores = (font_sizes - font_mean) / max(font_std, 1)
        
        # Create DataFrame with essential features
        df = pd.DataFrame({
            'text': texts,
            'font_size': font_sizes,
            'page': pages,
            'text_length': text_lengths,
            'word_count': word_counts,
            'font_z_score': font_z_scores,
            'is_short': text_lengths < 50,
            'is_first_page': pages == 1,
            'large_font': font_z_scores > 1
        })
        
        return df
    
    def lightweight_heuristics(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply lightweight heuristics for fast processing"""
        
        # Precompile regex patterns
        import re
        
        title_pattern = re.compile(r'^(chapter|section|part)\s+\d+', re.IGNORECASE)
        number_pattern = re.compile(r'^\d+\.\s')
        
        for block in blocks:
            text = block['text'].strip()
            
            # Fast heuristic scoring
            score = 0
            
            # Font size bonus
            if block.get('font_z_score', 0) > 1:
                score += 3
            
            # Position bonus
            if block.get('is_first_page', False):
                score += 2
            
            # Length bonus
            if block.get('is_short', False):
                score += 1
            
            # Pattern matching
            if title_pattern.match(text):
                score += 3
            elif number_pattern.match(text):
                score += 2
            
            # Assign preliminary label
            if score >= 5:
                block['fast_label'] = 'H1'
            elif score >= 3:
                block['fast_label'] = 'H2'
            elif score >= 1:
                block['fast_label'] = 'H3'
            else:
                block['fast_label'] = 'Paragraph'
            
            block['fast_score'] = score
        
        return blocks
    
    def memory_efficient_processing(self, pdf_path: str) -> Iterator[Dict[str, Any]]:
        """Process PDF with memory-efficient streaming"""
        
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            # Process pages one at a time
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_dict = page.get_text("dict")
                
                # Extract blocks from this page only
                page_blocks = []
                
                for block in page_dict.get("blocks", []):
                    if "lines" in block:
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                        
                        if block_text.strip():
                            page_blocks.append({
                                'text': block_text.strip(),
                                'page': page_num + 1,
                                'bbox': block["bbox"],
                                'font_size': 12  # Simplified
                            })
                
                # Yield blocks for this page
                for block in page_blocks:
                    yield block
                
                # Clean up page
                page = None
                gc.collect()
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error in memory-efficient processing: {e}")
            return
    
    def adaptive_processing(self, pdf_path: str, 
                           time_limit: float = 10.0) -> Dict[str, Any]:
        """Adaptive processing that adjusts based on time constraints"""
        
        start_time = time.time()
        
        # Start with fast heuristics
        try:
            # Quick text extraction
            blocks = list(self.memory_efficient_processing(pdf_path))
            
            if not blocks:
                return {"title": "", "outline": []}
            
            # Apply lightweight heuristics
            blocks = self.lightweight_heuristics(blocks)
            
            # Check time remaining
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            
            if remaining < 2.0:
                # Fast path - use heuristics only
                return self._generate_fast_output(blocks)
            
            else:
                # Normal path - can do more processing
                return self._generate_enhanced_output(blocks, remaining)
                
        except Exception as e:
            logger.error(f"Error in adaptive processing: {e}")
            return {"title": "", "outline": []}
    
    def _generate_fast_output(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate output using fast heuristics only"""
        
        # Sort blocks
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        title = ""
        outline = []
        
        # Find title (first high-scoring block on page 1)
        for block in sorted_blocks:
            if (block['page'] == 1 and 
                block.get('fast_score', 0) >= 4 and
                not title):
                title = block['text']
                break
        
        # Find headings
        for block in sorted_blocks:
            label = block.get('fast_label', 'Paragraph')
            if label in ['H1', 'H2', 'H3']:
                outline.append({
                    "level": label,
                    "text": block['text'],
                    "page": block['page']
                })
        
        return {"title": title, "outline": outline}
    
    def _generate_enhanced_output(self, blocks: List[Dict[str, Any]], 
                                 time_budget: float) -> Dict[str, Any]:
        """Generate output with enhanced processing within time budget"""
        
        # This would include more sophisticated processing
        # but still within time constraints
        
        return self._generate_fast_output(blocks)  # Simplified for now
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor and report performance metrics"""
        
        metrics = {
            'total_files_processed': len(self.processing_times),
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'min_processing_time': min(self.processing_times) if self.processing_times else 0,
            'files_exceeding_10s': sum(1 for t in self.processing_times if t > 10),
            'success_rate': len(self.processing_times) / max(len(self.processing_times), 1) * 100
        }
        
        return metrics
    
    def optimize_model_inference(self, model, X: np.ndarray) -> np.ndarray:
        """Optimize model inference for speed"""
        
        # Batch prediction for better throughput
        batch_size = 32
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)
        
        return np.array(predictions)
    
    def cleanup_resources(self):
        """Clean up resources and caches"""
        
        # Clear caches
        self.cached_feature_computation.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        # Reset metrics
        self.processing_times = []
        self.memory_usage = []
        
        logger.info("Resources cleaned up")


class FastPDFProcessor:
    """Fast PDF processor optimized for the hackathon constraints"""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer(max_workers=2, memory_limit_mb=512)
        
        # Precompiled patterns for speed
        import re
        self.title_patterns = [
            re.compile(r'^(chapter|section|part)\s+\d+', re.IGNORECASE),
            re.compile(r'^\d+\.\s'),
            re.compile(r'^\d+\.\d+\s'),
            re.compile(r'^[ivxlcdm]+\.\s', re.IGNORECASE)
        ]
        
        # Common title words
        self.title_words = {
            'report', 'proposal', 'manual', 'guide', 'handbook',
            'specification', 'standard', 'analysis', 'study'
        }
        
        # Common heading words
        self.heading_words = {
            'introduction', 'conclusion', 'summary', 'overview',
            'background', 'methodology', 'results', 'discussion'
        }
    
    def process_pdf_fast(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF with maximum speed optimization"""
        
        start_time = time.time()
        
        try:
            # Fast text extraction
            blocks = self._extract_blocks_fast(pdf_path)
            
            if not blocks:
                return {"title": "", "outline": []}
            
            # Fast feature computation
            blocks = self._compute_features_fast(blocks)
            
            # Fast classification
            blocks = self._classify_fast(blocks)
            
            # Generate output
            result = self._generate_output_fast(blocks)
            
            processing_time = time.time() - start_time
            logger.info(f"Fast processing completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fast processing: {e}")
            return {"title": "", "outline": []}
    
    def _extract_blocks_fast(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fast block extraction with minimal parsing"""
        
        blocks = []
        
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            # Limit pages for speed (first 20 pages for structure)
            max_pages = min(20, len(doc))
            
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                
                # Use simpler text extraction
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        block_text = ""
                        font_size = 12  # Default
                        
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                                font_size = span.get("size", 12)
                        
                        if block_text.strip() and len(block_text.strip()) > 2:
                            blocks.append({
                                'text': block_text.strip(),
                                'page': page_num + 1,
                                'bbox': block["bbox"],
                                'font_size': font_size
                            })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error in fast block extraction: {e}")
        
        return blocks
    
    def _compute_features_fast(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast feature computation with vectorized operations"""
        
        if not blocks:
            return blocks
        
        # Vectorized computations
        font_sizes = np.array([b['font_size'] for b in blocks])
        text_lengths = np.array([len(b['text']) for b in blocks])
        pages = np.array([b['page'] for b in blocks])
        
        # Statistics
        font_mean = np.mean(font_sizes)
        font_std = np.std(font_sizes)
        length_mean = np.mean(text_lengths)
        
        # Add features
        for i, block in enumerate(blocks):
            block['font_z_score'] = (font_sizes[i] - font_mean) / max(font_std, 1)
            block['is_large_font'] = font_sizes[i] > font_mean + font_std
            block['is_short'] = text_lengths[i] < 50
            block['is_medium'] = 50 <= text_lengths[i] <= 200
            block['is_first_page'] = pages[i] == 1
            block['position_score'] = 1.0 / pages[i]  # Earlier pages score higher
        
        return blocks
    
    def _classify_fast(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast classification using optimized heuristics"""
        
        for block in blocks:
            text = block['text'].strip()
            text_lower = text.lower()
            
            # Initialize scores
            title_score = 0
            h1_score = 0
            h2_score = 0
            h3_score = 0
            
            # Font size scoring
            if block.get('font_z_score', 0) > 2:
                title_score += 4
                h1_score += 3
            elif block.get('font_z_score', 0) > 1:
                h1_score += 2
                h2_score += 1
            elif block.get('font_z_score', 0) > 0:
                h2_score += 1
                h3_score += 0.5
            
            # Position scoring
            if block.get('is_first_page', False):
                title_score += 3
                h1_score += 2
            
            # Length scoring
            if block.get('is_short', False):
                title_score += 2
                h1_score += 1
                h2_score += 1
            
            # Pattern scoring (fast)
            for pattern in self.title_patterns:
                if pattern.match(text):
                    h1_score += 3
                    break
            
            # Word scoring (fast)
            text_words = set(text_lower.split())
            
            title_matches = len(text_words.intersection(self.title_words))
            if title_matches > 0:
                title_score += title_matches * 2
            
            heading_matches = len(text_words.intersection(self.heading_words))
            if heading_matches > 0:
                h1_score += heading_matches
            
            # Assign label
            scores = {
                'Title': title_score,
                'H1': h1_score,
                'H2': h2_score,
                'H3': h3_score
            }
            
            max_score = max(scores.values())
            if max_score > 2:
                block['predicted_label'] = max(scores, key=scores.get)
                block['confidence'] = min(max_score / 10, 1.0)
            else:
                block['predicted_label'] = 'Paragraph'
                block['confidence'] = 0.0
        
        return blocks
    
    def _generate_output_fast(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fast output generation"""
        
        # Sort blocks
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        title = ""
        outline = []
        
        # Find title
        for block in sorted_blocks:
            if (block.get('predicted_label') == 'Title' and 
                block.get('confidence', 0) > 0.3 and
                not title):
                title = block['text']
                break
        
        # Find headings
        for block in sorted_blocks:
            label = block.get('predicted_label', 'Paragraph')
            confidence = block.get('confidence', 0)
            
            if label in ['H1', 'H2', 'H3'] and confidence > 0.2:
                outline.append({
                    "level": label,
                    "text": block['text'],
                    "page": block['page']
                })
        
        # Fallback title
        if not title and outline:
            title = outline[0]['text']
        elif not title:
            # Find first reasonable text on page 1
            for block in sorted_blocks:
                if (block['page'] == 1 and 
                    10 <= len(block['text']) <= 100 and
                    block.get('font_z_score', 0) > 0):
                    title = block['text']
                    break
        
        return {"title": title, "outline": outline}