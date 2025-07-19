#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for PDF Structure Detection
Tests the system against various document types and validates performance
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import numpy as np
from collections import defaultdict
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Import our components
from src.enhanced_json_generator import EnhancedJSONGenerator
from src.enhanced_feature_extractor import EnhancedPDFFeatureExtractor
from src.advanced_contextual_analyzer import AdvancedContextualAnalyzer
from src.performance_optimizer import PerformanceOptimizer, FastPDFProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation of PDF structure detection system"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.document_types = {
            'academic': ['paper', 'research', 'journal', 'thesis'],
            'technical': ['manual', 'specification', 'guide', 'documentation'],
            'business': ['report', 'proposal', 'analysis', 'strategy'],
            'legal': ['contract', 'agreement', 'terms', 'policy'],
            'education': ['textbook', 'course', 'curriculum', 'syllabus']
        }
        
    def evaluate_system(self, test_pdfs_dir: str, expected_outputs_dir: str = None) -> Dict[str, Any]:
        """Evaluate the complete system"""
        
        logger.info("Starting comprehensive evaluation...")
        
        # Initialize components
        try:
            # ML-based system
            ml_generator = EnhancedJSONGenerator()
            
            # Heuristic-based system
            feature_extractor = EnhancedPDFFeatureExtractor()
            contextual_analyzer = AdvancedContextualAnalyzer()
            
            # Performance optimizer
            performance_optimizer = PerformanceOptimizer()
            fast_processor = FastPDFProcessor()
            
            # Find test files
            test_files = self._find_test_files(test_pdfs_dir, expected_outputs_dir)
            
            if not test_files:
                logger.error("No test files found")
                return {}
            
            logger.info(f"Found {len(test_files)} test files")
            
            # Run comprehensive tests
            results = {
                'ml_system': self._test_ml_system(test_files, ml_generator),
                'heuristic_system': self._test_heuristic_system(test_files, feature_extractor, contextual_analyzer),
                'fast_system': self._test_fast_system(test_files, fast_processor),
                'performance_analysis': self._analyze_performance(test_files),
                'document_type_analysis': self._analyze_by_document_type(test_files),
                'summary': self._generate_summary()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {'error': str(e)}
    
    def _find_test_files(self, test_dir: str, expected_dir: str = None) -> List[Dict[str, Any]]:
        """Find test files and their expected outputs"""
        
        test_path = Path(test_dir)
        expected_path = Path(expected_dir) if expected_dir else None
        
        test_files = []
        
        for pdf_file in test_path.glob("*.pdf"):
            test_file = {
                'pdf_path': str(pdf_file),
                'name': pdf_file.stem,
                'expected_output': None,
                'document_type': self._classify_document_type(pdf_file.name)
            }
            
            # Look for expected output
            if expected_path:
                expected_file = expected_path / f"{pdf_file.stem}.json"
                if expected_file.exists():
                    try:
                        with open(expected_file, 'r', encoding='utf-8') as f:
                            test_file['expected_output'] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load expected output for {pdf_file.name}: {e}")
            
            test_files.append(test_file)
        
        return test_files
    
    def _classify_document_type(self, filename: str) -> str:
        """Classify document type based on filename"""
        
        filename_lower = filename.lower()
        
        for doc_type, keywords in self.document_types.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
        
        return 'unknown'
    
    def _test_ml_system(self, test_files: List[Dict[str, Any]], generator: EnhancedJSONGenerator) -> Dict[str, Any]:
        """Test ML-based system"""
        
        logger.info("Testing ML-based system...")
        
        results = {
            'total_files': len(test_files),
            'successful_processes': 0,
            'failed_processes': 0,
            'processing_times': [],
            'accuracy_metrics': {},
            'detailed_results': []
        }
        
        for test_file in test_files:
            try:
                start_time = time.time()
                
                # Process with ML system
                try:
                    output = generator.process_pdf(test_file['pdf_path'])
                    processing_time = time.time() - start_time
                    
                    results['successful_processes'] += 1
                    results['processing_times'].append(processing_time)
                    
                    # Calculate accuracy if expected output available
                    if test_file['expected_output']:
                        accuracy = self._calculate_accuracy(output, test_file['expected_output'])
                        results['detailed_results'].append({
                            'file': test_file['name'],
                            'processing_time': processing_time,
                            'accuracy': accuracy,
                            'output': output
                        })
                    else:
                        results['detailed_results'].append({
                            'file': test_file['name'],
                            'processing_time': processing_time,
                            'output': output
                        })
                    
                except Exception as e:
                    logger.error(f"ML system failed on {test_file['name']}: {e}")
                    results['failed_processes'] += 1
                    results['detailed_results'].append({
                        'file': test_file['name'],
                        'error': str(e)
                    })
                
            except Exception as e:
                logger.error(f"Error testing ML system on {test_file['name']}: {e}")
                results['failed_processes'] += 1
        
        # Calculate summary metrics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['max_processing_time'] = max(results['processing_times'])
            results['files_exceeding_10s'] = sum(1 for t in results['processing_times'] if t > 10)
        
        return results
    
    def _test_heuristic_system(self, test_files: List[Dict[str, Any]], 
                              feature_extractor: EnhancedPDFFeatureExtractor,
                              contextual_analyzer: AdvancedContextualAnalyzer) -> Dict[str, Any]:
        """Test heuristic-based system"""
        
        logger.info("Testing heuristic-based system...")
        
        results = {
            'total_files': len(test_files),
            'successful_processes': 0,
            'failed_processes': 0,
            'processing_times': [],
            'detailed_results': []
        }
        
        for test_file in test_files:
            try:
                start_time = time.time()
                
                # Extract features
                blocks = feature_extractor.process_pdf(test_file['pdf_path'])
                
                if blocks:
                    # Analyze context
                    context = contextual_analyzer.analyze_document_context(blocks)
                    
                    # Enhance detection
                    blocks = contextual_analyzer.enhance_heading_detection(blocks, context)
                    blocks = contextual_analyzer.calculate_contextual_scores(blocks, context)
                    blocks = contextual_analyzer.validate_document_structure(blocks, context)
                    
                    # Generate output
                    output = self._generate_output_from_blocks(blocks, context)
                    
                    processing_time = time.time() - start_time
                    
                    results['successful_processes'] += 1
                    results['processing_times'].append(processing_time)
                    
                    # Calculate accuracy if expected output available
                    if test_file['expected_output']:
                        accuracy = self._calculate_accuracy(output, test_file['expected_output'])
                        results['detailed_results'].append({
                            'file': test_file['name'],
                            'processing_time': processing_time,
                            'accuracy': accuracy,
                            'output': output,
                            'context': {
                                'document_type': context.document_type,
                                'structure_complexity': context.structure_complexity,
                                'language': context.language
                            }
                        })
                    else:
                        results['detailed_results'].append({
                            'file': test_file['name'],
                            'processing_time': processing_time,
                            'output': output,
                            'context': {
                                'document_type': context.document_type,
                                'structure_complexity': context.structure_complexity,
                                'language': context.language
                            }
                        })
                
                else:
                    results['failed_processes'] += 1
                    results['detailed_results'].append({
                        'file': test_file['name'],
                        'error': 'No blocks extracted'
                    })
                
            except Exception as e:
                logger.error(f"Error testing heuristic system on {test_file['name']}: {e}")
                results['failed_processes'] += 1
                results['detailed_results'].append({
                    'file': test_file['name'],
                    'error': str(e)
                })
        
        # Calculate summary metrics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['max_processing_time'] = max(results['processing_times'])
            results['files_exceeding_10s'] = sum(1 for t in results['processing_times'] if t > 10)
        
        return results
    
    def _test_fast_system(self, test_files: List[Dict[str, Any]], 
                         fast_processor: FastPDFProcessor) -> Dict[str, Any]:
        """Test fast processing system"""
        
        logger.info("Testing fast processing system...")
        
        results = {
            'total_files': len(test_files),
            'successful_processes': 0,
            'failed_processes': 0,
            'processing_times': [],
            'detailed_results': []
        }
        
        for test_file in test_files:
            try:
                start_time = time.time()
                
                output = fast_processor.process_pdf_fast(test_file['pdf_path'])
                processing_time = time.time() - start_time
                
                results['successful_processes'] += 1
                results['processing_times'].append(processing_time)
                
                # Calculate accuracy if expected output available
                if test_file['expected_output']:
                    accuracy = self._calculate_accuracy(output, test_file['expected_output'])
                    results['detailed_results'].append({
                        'file': test_file['name'],
                        'processing_time': processing_time,
                        'accuracy': accuracy,
                        'output': output
                    })
                else:
                    results['detailed_results'].append({
                        'file': test_file['name'],
                        'processing_time': processing_time,
                        'output': output
                    })
                
            except Exception as e:
                logger.error(f"Error testing fast system on {test_file['name']}: {e}")
                results['failed_processes'] += 1
                results['detailed_results'].append({
                    'file': test_file['name'],
                    'error': str(e)
                })
        
        # Calculate summary metrics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['max_processing_time'] = max(results['processing_times'])
            results['files_exceeding_10s'] = sum(1 for t in results['processing_times'] if t > 10)
        
        return results
    
    def _analyze_performance(self, test_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        
        logger.info("Analyzing performance characteristics...")
        
        performance_analysis = {
            'time_constraint_compliance': {},
            'memory_usage': {},
            'scalability': {},
            'robustness': {}
        }
        
        # Analyze time constraints
        # This would include analysis of processing times vs document complexity
        
        # Analyze memory usage
        # This would include memory profiling
        
        # Analyze scalability
        # This would include batch processing analysis
        
        # Analyze robustness
        # This would include error handling analysis
        
        return performance_analysis
    
    def _analyze_by_document_type(self, test_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by document type"""
        
        logger.info("Analyzing performance by document type...")
        
        type_analysis = defaultdict(lambda: {
            'count': 0,
            'avg_processing_time': 0,
            'success_rate': 0,
            'avg_accuracy': 0
        })
        
        for test_file in test_files:
            doc_type = test_file['document_type']
            type_analysis[doc_type]['count'] += 1
        
        return dict(type_analysis)
    
    def _calculate_accuracy(self, predicted: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        
        metrics = {
            'title_similarity': 0.0,
            'heading_precision': 0.0,
            'heading_recall': 0.0,
            'heading_f1': 0.0,
            'exact_match': 0.0
        }
        
        # Title similarity
        pred_title = predicted.get('title', '').strip().lower()
        actual_title = actual.get('title', '').strip().lower()
        
        if pred_title and actual_title:
            pred_words = set(pred_title.split())
            actual_words = set(actual_title.split())
            
            if pred_words and actual_words:
                intersection = pred_words.intersection(actual_words)
                union = pred_words.union(actual_words)
                metrics['title_similarity'] = len(intersection) / len(union)
        
        # Heading metrics
        pred_headings = set()
        actual_headings = set()
        
        for item in predicted.get('outline', []):
            pred_headings.add((item['level'], item['text'].strip().lower()))
        
        for item in actual.get('outline', []):
            actual_headings.add((item['level'], item['text'].strip().lower()))
        
        if pred_headings or actual_headings:
            intersection = pred_headings.intersection(actual_headings)
            
            if pred_headings:
                metrics['heading_precision'] = len(intersection) / len(pred_headings)
            
            if actual_headings:
                metrics['heading_recall'] = len(intersection) / len(actual_headings)
            
            if metrics['heading_precision'] + metrics['heading_recall'] > 0:
                metrics['heading_f1'] = (2 * metrics['heading_precision'] * metrics['heading_recall']) / \
                                       (metrics['heading_precision'] + metrics['heading_recall'])
        
        # Exact match
        if (pred_title == actual_title and 
            len(pred_headings.symmetric_difference(actual_headings)) == 0):
            metrics['exact_match'] = 1.0
        
        return metrics
    
    def _generate_output_from_blocks(self, blocks: List[Dict[str, Any]], context) -> Dict[str, Any]:
        """Generate output from blocks with context"""
        
        # Sort blocks
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        title = ""
        outline = []
        
        # Find title using contextual scores
        for block in sorted_blocks:
            if (block['page'] == 1 and 
                block.get('contextual_title_score', 0) > 2 and
                not title):
                title = block['text'].strip()
                break
        
        # Find headings using contextual scores
        for block in sorted_blocks:
            h1_score = block.get('contextual_h1_score', 0)
            h2_score = block.get('contextual_h2_score', 0)
            h3_score = block.get('contextual_h3_score', 0)
            
            if h1_score > 3:
                outline.append({
                    "level": "H1",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
            elif h2_score > 3:
                outline.append({
                    "level": "H2",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
            elif h3_score > 3:
                outline.append({
                    "level": "H3",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
        
        # Fallback title
        if not title and outline:
            title = outline[0]['text']
        
        return {"title": title, "outline": outline}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary"""
        
        return {
            'evaluation_complete': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'recommendations': [
                "Use ML system for best accuracy when available",
                "Fall back to enhanced heuristics for reliability",
                "Use fast processing for time-critical scenarios",
                "Consider document type for optimal processing"
            ]
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_file}")


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Comprehensive PDF Structure Detection Evaluation")
    parser.add_argument("--test-pdfs", required=True, help="Directory containing test PDFs")
    parser.add_argument("--expected-outputs", help="Directory containing expected JSON outputs")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_system(args.test_pdfs, args.expected_outputs)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for system_name, system_results in results.items():
        if isinstance(system_results, dict) and 'total_files' in system_results:
            print(f"\n{system_name.upper()}:")
            print(f"  Total files: {system_results['total_files']}")
            print(f"  Successful: {system_results['successful_processes']}")
            print(f"  Failed: {system_results['failed_processes']}")
            if 'avg_processing_time' in system_results:
                print(f"  Avg processing time: {system_results['avg_processing_time']:.2f}s")
            if 'files_exceeding_10s' in system_results:
                print(f"  Files exceeding 10s: {system_results['files_exceeding_10s']}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()