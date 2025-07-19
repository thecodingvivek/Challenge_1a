#!/usr/bin/env python3
"""
API Wrapper for PDF Structure Detection System
Provides a clean, easy-to-use interface for integration
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our components
from .enhanced_json_generator import EnhancedJSONGenerator
from .enhanced_feature_extractor import EnhancedPDFFeatureExtractor
from .advanced_contextual_analyzer import AdvancedContextualAnalyzer
from .performance_optimizer import PerformanceOptimizer, FastPDFProcessor
from .monitoring_system import PerformanceMonitor, EnhancedLogger, get_global_monitor
from .config_manager import ConfigManager, SystemConfig, load_preset_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of PDF processing operation"""
    success: bool
    title: str
    outline: List[Dict[str, Any]]
    processing_time: float
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

@dataclass
class BatchProcessingResult:
    """Result of batch PDF processing operation"""
    total_files: int
    successful_files: int
    failed_files: int
    results: List[ProcessingResult]
    total_time: float
    avg_time_per_file: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'results': [r.to_dict() for r in self.results],
            'total_time': self.total_time,
            'avg_time_per_file': self.avg_time_per_file
        }

class PDFStructureDetector:
    """Main API for PDF structure detection"""
    
    def __init__(self, 
                 config: Optional[Union[str, SystemConfig]] = None,
                 model_path: Optional[str] = None,
                 enable_monitoring: bool = True,
                 enable_caching: bool = True):
        """
        Initialize PDF Structure Detector
        
        Args:
            config: Configuration (file path, SystemConfig object, or None for defaults)
            model_path: Path to trained model (optional)
            enable_monitoring: Enable performance monitoring
            enable_caching: Enable result caching
        """
        
        # Load configuration
        if isinstance(config, str):
            self.config_manager = ConfigManager(config)
        elif isinstance(config, SystemConfig):
            self.config_manager = ConfigManager()
            self.config_manager.config = config
        else:
            # Use hackathon preset by default
            self.config_manager = ConfigManager()
            self.config_manager.config = load_preset_config('hackathon')
        
        self.config = self.config_manager.get_config()
        
        # Override model path if provided
        if model_path:
            self.config.model.model_path = model_path
        
        # Initialize monitoring
        self.monitoring_enabled = enable_monitoring
        if enable_monitoring:
            self.performance_monitor = get_global_monitor()
        else:
            self.performance_monitor = None
        
        self.logger = EnhancedLogger("PDFStructureDetector", self.performance_monitor)
        
        # Initialize components
        self._initialize_components()
        
        # Caching
        self.caching_enabled = enable_caching
        self.cache = {} if enable_caching else None
        
        self.logger.info("PDF Structure Detector initialized", 
                        config_preset="hackathon",
                        monitoring_enabled=enable_monitoring,
                        caching_enabled=enable_caching)
    
    def _initialize_components(self):
        """Initialize processing components"""
        
        try:
            # Try to initialize ML-based generator
            if (self.config.processing.enable_ml_models and 
                os.path.exists(self.config.model.model_path)):
                
                self.ml_generator = EnhancedJSONGenerator(self.config.model.model_path)
                self.has_ml_model = True
                self.logger.info("ML model loaded", model_path=self.config.model.model_path)
            else:
                self.ml_generator = None
                self.has_ml_model = False
                self.logger.warning("No ML model available, using heuristics only")
            
            # Initialize heuristic components
            self.feature_extractor = EnhancedPDFFeatureExtractor()
            
            if self.config.processing.enable_contextual_analysis:
                self.contextual_analyzer = AdvancedContextualAnalyzer()
            else:
                self.contextual_analyzer = None
            
            # Initialize performance optimizer
            if self.config.processing.use_fast_mode:
                self.fast_processor = FastPDFProcessor()
            else:
                self.fast_processor = None
            
            self.performance_optimizer = PerformanceOptimizer(
                max_workers=self.config.processing.max_workers,
                memory_limit_mb=self.config.processing.memory_limit_mb
            )
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def process_pdf(self, 
                   pdf_path: Union[str, Path], 
                   output_format: str = "dict",
                   include_metadata: bool = None,
                   timeout: Optional[float] = None) -> Union[ProcessingResult, Dict[str, Any], str]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            output_format: Output format ("dict", "json", "result_object")
            include_metadata: Include processing metadata (overrides config)
            timeout: Processing timeout in seconds (overrides config)
        
        Returns:
            Processing result in requested format
        """
        
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            error_msg = f"PDF file not found: {pdf_path}"
            self.logger.error(error_msg)
            
            if output_format == "result_object":
                return ProcessingResult(
                    success=False,
                    title="",
                    outline=[],
                    processing_time=0.0,
                    error_message=error_msg
                )
            elif output_format == "json":
                return json.dumps({"error": error_msg})
            else:
                return {"error": error_msg}
        
        # Check cache
        cache_key = str(pdf_path.absolute())
        if self.caching_enabled and cache_key in self.cache:
            self.logger.info("Returning cached result", file=pdf_path.name)
            return self._format_output(self.cache[cache_key], output_format)
        
        # Start monitoring
        if self.monitoring_enabled:
            measurement = self.performance_monitor.start_processing_measurement(str(pdf_path))
        
        start_time = time.time()
        processing_timeout = timeout or self.config.processing.time_limit_seconds
        
        try:
            self.logger.push_context(file=pdf_path.name)
            self.logger.info("Starting PDF processing")
            
            # Choose processing method
            if self.config.processing.use_fast_mode and self.fast_processor:
                result = self._process_with_fast_mode(pdf_path, processing_timeout)
            elif self.has_ml_model and self.ml_generator:
                result = self._process_with_ml_model(pdf_path, processing_timeout)
            else:
                result = self._process_with_heuristics(pdf_path, processing_timeout)
            
            processing_time = time.time() - start_time
            
            # Add metadata if requested
            metadata = None
            if include_metadata or (include_metadata is None and self.config.output.include_metadata):
                metadata = {
                    'file_name': pdf_path.name,
                    'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'processing_method': self._get_processing_method(),
                    'processing_time': processing_time,
                    'config_preset': 'hackathon'
                }
            
            # Create result object
            processing_result = ProcessingResult(
                success=True,
                title=result.get('title', ''),
                outline=result.get('outline', []),
                processing_time=processing_time,
                confidence_score=result.get('avg_confidence'),
                metadata=metadata
            )
            
            # End monitoring
            if self.monitoring_enabled:
                self.performance_monitor.end_processing_measurement(
                    measurement, result, success=True
                )
            
            # Cache result
            if self.caching_enabled:
                self.cache[cache_key] = processing_result
            
            self.logger.performance("PDF processing completed", processing_time)
            self.logger.pop_context()
            
            return self._format_output(processing_result, output_format)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Error processing PDF: {error_msg}")
            
            # End monitoring with error
            if self.monitoring_enabled:
                self.performance_monitor.end_processing_measurement(
                    measurement, {}, success=False, error_message=error_msg
                )
            
            processing_result = ProcessingResult(
                success=False,
                title="",
                outline=[],
                processing_time=processing_time,
                error_message=error_msg
            )
            
            self.logger.pop_context()
            
            return self._format_output(processing_result, output_format)
    
    def _process_with_ml_model(self, pdf_path: Path, timeout: float) -> Dict[str, Any]:
        """Process PDF using ML model"""
        self.logger.debug("Using ML model processing")
        return self.ml_generator.process_pdf(str(pdf_path))
    
    def _process_with_heuristics(self, pdf_path: Path, timeout: float) -> Dict[str, Any]:
        """Process PDF using enhanced heuristics"""
        self.logger.debug("Using heuristic processing")
        
        # Extract features
        blocks = self.feature_extractor.process_pdf(str(pdf_path))
        
        if not blocks:
            return {"title": "", "outline": []}
        
        # Apply contextual analysis if available
        if self.contextual_analyzer:
            context = self.contextual_analyzer.analyze_document_context(blocks)
            blocks = self.contextual_analyzer.enhance_heading_detection(blocks, context)
            blocks = self.contextual_analyzer.calculate_contextual_scores(blocks, context)
            blocks = self.contextual_analyzer.validate_document_structure(blocks, context)
        
        # Generate output
        return self._generate_output_from_blocks(blocks)
    
    def _process_with_fast_mode(self, pdf_path: Path, timeout: float) -> Dict[str, Any]:
        """Process PDF using fast mode"""
        self.logger.debug("Using fast mode processing")
        return self.fast_processor.process_pdf_fast(str(pdf_path))
    
    def _generate_output_from_blocks(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate output from blocks"""
        
        # Sort blocks
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        title = ""
        outline = []
        
        # Find title
        for block in sorted_blocks:
            if (block['page'] == 1 and 
                block.get('title_score', 0) > self.config.heuristic.title_score_threshold and
                not title):
                title = block['text'].strip()
                break
        
        # Find headings
        for block in sorted_blocks:
            h1_score = block.get('h1_score', 0)
            h2_score = block.get('h2_score', 0)
            h3_score = block.get('h3_score', 0)
            
            if h1_score > self.config.heuristic.h1_score_threshold:
                outline.append({
                    "level": "H1",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
            elif h2_score > self.config.heuristic.h2_score_threshold:
                outline.append({
                    "level": "H2",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
            elif h3_score > self.config.heuristic.h3_score_threshold:
                outline.append({
                    "level": "H3",
                    "text": block['text'].strip(),
                    "page": block['page']
                })
        
        # Fallback title
        if not title and outline:
            title = outline[0]['text']
        
        return {"title": title, "outline": outline}
    
    def _get_processing_method(self) -> str:
        """Get description of processing method used"""
        if self.config.processing.use_fast_mode:
            return "fast_heuristics"
        elif self.has_ml_model:
            return "ml_ensemble"
        else:
            return "enhanced_heuristics"
    
    def _format_output(self, result: ProcessingResult, format: str) -> Union[ProcessingResult, Dict[str, Any], str]:
        """Format output in requested format"""
        
        if format == "result_object":
            return result
        elif format == "json":
            return result.to_json()
        else:  # dict
            return result.to_dict()
    
    def process_batch(self, 
                     pdf_paths: List[Union[str, Path]], 
                     output_format: str = "dict",
                     max_workers: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> BatchProcessingResult:
        """
        Process multiple PDF files in batch
        
        Args:
            pdf_paths: List of PDF file paths
            output_format: Output format for individual results
            max_workers: Maximum number of parallel workers
            progress_callback: Callback function for progress updates
        
        Returns:
            Batch processing result
        """
        
        start_time = time.time()
        workers = max_workers or self.config.processing.max_workers
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} files", 
                        max_workers=workers)
        
        results = []
        successful_count = 0
        failed_count = 0
        
        if workers == 1 or len(pdf_paths) == 1:
            # Sequential processing
            for i, pdf_path in enumerate(pdf_paths):
                result = self.process_pdf(pdf_path, output_format="result_object")
                results.append(result)
                
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(pdf_paths), result)
        
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_path = {
                    executor.submit(self.process_pdf, pdf_path, "result_object"): pdf_path 
                    for pdf_path in pdf_paths
                }
                
                completed = 0
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            successful_count += 1
                        else:
                            failed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error in batch processing {pdf_path}: {e}")
                        error_result = ProcessingResult(
                            success=False,
                            title="",
                            outline=[],
                            processing_time=0.0,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        failed_count += 1
                    
                    completed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed, len(pdf_paths), results[-1])
        
        total_time = time.time() - start_time
        avg_time = total_time / len(pdf_paths) if pdf_paths else 0
        
        self.logger.info(f"Batch processing completed", 
                        total_files=len(pdf_paths),
                        successful=successful_count,
                        failed=failed_count,
                        total_time=f"{total_time:.2f}s")
        
        # Convert results to requested format
        if output_format != "result_object":
            formatted_results = [self._format_output(r, output_format) for r in results]
        else:
            formatted_results = results
        
        return BatchProcessingResult(
            total_files=len(pdf_paths),
            successful_files=successful_count,
            failed_files=failed_count,
            results=formatted_results,
            total_time=total_time,
            avg_time_per_file=avg_time
        )
    
    def process_directory(self, 
                         input_dir: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         pattern: str = "*.pdf",
                         save_outputs: bool = True) -> BatchProcessingResult:
        """
        Process all PDFs in a directory
        
        Args:
            input_dir: Input directory containing PDFs
            output_dir: Output directory for JSON files (optional)
            pattern: File pattern to match (default: "*.pdf")
            save_outputs: Save individual JSON outputs
        
        Returns:
            Batch processing result
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else None
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")
        
        # Find PDF files
        pdf_files = list(input_path.glob(pattern))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_path}")
            return BatchProcessingResult(0, 0, 0, [], 0.0, 0.0)
        
        # Process files
        batch_result = self.process_batch(pdf_files, output_format="result_object")
        
        # Save outputs if requested
        if save_outputs and output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            
            for pdf_file, result in zip(pdf_files, batch_result.results):
                if result.success:
                    output_file = output_path / f"{pdf_file.stem}.json"
                    
                    output_data = {
                        "title": result.title,
                        "outline": result.outline
                    }
                    
                    if self.config.output.include_metadata and result.metadata:
                        output_data["metadata"] = result.metadata
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, 
                                indent=self.config.output.indent,
                                ensure_ascii=self.config.output.ensure_ascii)
                    
                    self.logger.debug(f"Saved output: {output_file}")
        
        return batch_result
    
    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics"""
        if self.monitoring_enabled:
            return self.performance_monitor.get_detailed_report()
        return None
    
    def clear_cache(self):
        """Clear result cache"""
        if self.caching_enabled:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration"""
        self.config_manager.update_config(config_updates)
        self.config = self.config_manager.get_config()
        self.logger.info("Configuration updated")

# Convenience functions for quick usage

def process_single_pdf(pdf_path: Union[str, Path], 
                      model_path: Optional[str] = None,
                      fast_mode: bool = False) -> Dict[str, Any]:
    """
    Quick function to process a single PDF
    
    Args:
        pdf_path: Path to PDF file
        model_path: Optional path to trained model
        fast_mode: Use fast processing mode
    
    Returns:
        Processing result as dictionary
    """
    
    # Create config
    config = load_preset_config('hackathon')
    config.processing.use_fast_mode = fast_mode
    
    if model_path:
        config.model.model_path = model_path
    
    # Process PDF
    detector = PDFStructureDetector(config=config, enable_monitoring=False)
    return detector.process_pdf(pdf_path, output_format="dict")

def process_pdf_batch(pdf_paths: List[Union[str, Path]], 
                     output_dir: Optional[str] = None,
                     max_workers: int = 2) -> BatchProcessingResult:
    """
    Quick function to process multiple PDFs
    
    Args:
        pdf_paths: List of PDF file paths
        output_dir: Optional output directory for JSON files
        max_workers: Maximum number of parallel workers
    
    Returns:
        Batch processing result
    """
    
    config = load_preset_config('hackathon')
    config.processing.max_workers = max_workers
    
    detector = PDFStructureDetector(config=config, enable_monitoring=False)
    
    if output_dir:
        # Save individual outputs
        result = detector.process_batch(pdf_paths, max_workers=max_workers)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for pdf_path, processing_result in zip(pdf_paths, result.results):
            if processing_result.success:
                output_file = output_path / f"{Path(pdf_path).stem}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "title": processing_result.title,
                        "outline": processing_result.outline
                    }, f, indent=2, ensure_ascii=False)
        
        return result
    
    else:
        return detector.process_batch(pdf_paths, max_workers=max_workers)

def main():
    """Example usage of the API"""
    
    # Simple usage
    result = process_single_pdf("example.pdf", fast_mode=True)
    print("Simple result:", json.dumps(result, indent=2))
    
    # Advanced usage
    config = load_preset_config('hackathon')
    detector = PDFStructureDetector(config=config)
    
    # Process single file
    result = detector.process_pdf("example.pdf", output_format="result_object")
    print(f"Success: {result.success}")
    print(f"Title: {result.title}")
    print(f"Headings: {len(result.outline)}")
    
    # Get performance metrics
    metrics = detector.get_performance_metrics()
    if metrics:
        print(f"Processing time: {metrics['system_metrics']['avg_processing_time']:.2f}s")

if __name__ == "__main__":
    main()