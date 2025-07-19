# scripts/evaluation/benchmark_performance.py
import time
import psutil
import pandas as pd
from pathlib import Path
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api_wrapper import PDFStructureDetector

def benchmark_system():
    """Benchmark system performance"""
    
    # Load test files
    test_dir = Path("data/raw_pdfs/test")
    test_files = list(test_dir.glob("*.pdf"))
    
    if not test_files:
        print("No test files found")
        return
    
    # Initialize detector
    detector = PDFStructureDetector(enable_monitoring=True)
    
    # Benchmark metrics
    benchmark_results = []
    
    for pdf_file in test_files:
        print(f"Benchmarking {pdf_file.name}...")
        
        # Measure system resources before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        start_time = time.time()
        
        # Process PDF
        result = detector.process_pdf(str(pdf_file), output_format="result_object")
        
        end_time = time.time()
        
        # Measure system resources after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = memory_after - memory_before
        file_size = pdf_file.stat().st_size / 1024 / 1024  # MB
        
        benchmark_result = {
            'file_name': pdf_file.name,
            'file_size_mb': file_size,
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': memory_after,
            'cpu_usage': cpu_after,
            'success': result.success,
            'title_length': len(result.title) if result.title else 0,
            'heading_count': len(result.outline),
            'time_per_mb': processing_time / file_size if file_size > 0 else 0
        }
        
        benchmark_results.append(benchmark_result)
        
        print(f"  Time: {processing_time:.2f}s, Memory: {memory_used:.1f}MB")
    
    # Save benchmark results
    df = pd.DataFrame(benchmark_results)
    df.to_csv("results/performance_metrics/benchmark_results.csv", index=False)
    
    # Print summary statistics
    print(f"\nBenchmark Summary:")
    print(f"Files processed: {len(benchmark_results)}")
    print(f"Average time: {df['processing_time'].mean():.2f}s")
    print(f"Max time: {df['processing_time'].max():.2f}s")
    print(f"Average memory: {df['memory_used_mb'].mean():.1f}MB")
    print(f"Max memory: {df['peak_memory_mb'].max():.1f}MB")
    print(f"Success rate: {df['success'].mean()*100:.1f}%")
    print(f"Files exceeding 10s: {(df['processing_time'] > 10).sum()}")
    
    return df

if __name__ == "__main__":
    benchmark_system()