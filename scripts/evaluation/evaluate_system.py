# scripts/evaluation/evaluate_system.py
import sys

from comprehensive_evaluation import ComprehensiveEvaluator

def run_comprehensive_evaluation():
    """Run comprehensive system evaluation"""
    
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate on test set
    results = evaluator.evaluate_system(
        test_pdfs_dir="data/raw_pdfs/test",
        expected_outputs_dir="data/ground_truth/test"
    )
    
    # Save results
    import json
    with open("results/evaluation_reports/comprehensive_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("Evaluation Results:")
    print("="*50)
    
    for system_name, system_results in results.items():
        if isinstance(system_results, dict) and 'total_files' in system_results:
            print(f"\n{system_name.upper()}:")
            print(f"  Success rate: {system_results['successful_processes']}/{system_results['total_files']}")
            if 'avg_processing_time' in system_results:
                print(f"  Avg time: {system_results['avg_processing_time']:.2f}s")

if __name__ == "__main__":
    run_comprehensive_evaluation()