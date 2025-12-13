"""
Complexity Analysis Script

Analyzes computational complexity of models:
- Parameter count
- Training time
- Inference time  
- Model size
- Accuracy vs complexity trade-offs

Usage:
    python scripts/analyze_complexity.py --models lgbm tabpfn
    python scripts/analyze_complexity.py --dataset covtype --models lgbm xgb tabpfn
"""

import os
import sys
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.complexity import ComplexityAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze computational complexity of models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["covtype", "heloc", "higgs", "all"],
        help="Dataset to analyze (default: all)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "tabpfn"],
        help="Models to compare (default: lgbm tabpfn)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/analysis/complexity_*.csv)"
    )
    
    args = parser.parse_args()
    
    analyzer = ComplexityAnalyzer()
    
    print("=" * 70)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    if args.dataset == "all":
        # Generate summary across all datasets
        print("\nAnalyzing all datasets...")
        df = analyzer.generate_summary_report(model_types=args.models)
        
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        print(df.to_string(index=False))
        
    else:
        # Analyze specific dataset
        print(f"\nAnalyzing dataset: {args.dataset}")
        df = analyzer.compare_models(
            dataset_name=args.dataset,
            model_types=args.models,
            save_report=True
        )
        
        print("\n" + "=" * 70)
        print(f"COMPLEXITY COMPARISON: {args.dataset.upper()}")
        print("=" * 70)
        print(df.to_string(index=False))
    
    # Print key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    if "tabpfn" in args.models or "baseline" in args.models:
        baseline_type = "tabpfn" if "tabpfn" in args.models else "baseline"
        
        for model_type in args.models:
            if model_type != baseline_type:
                # Compare with baseline
                baseline_rows = df[df["model_type"] == baseline_type]
                model_rows = df[df["model_type"] == model_type]
                
                if len(baseline_rows) > 0 and len(model_rows) > 0:
                    baseline_params = baseline_rows["n_parameters"].mean()
                    model_params = model_rows["n_parameters"].mean()
                    
                    baseline_time = baseline_rows["train_time"].mean()
                    model_time = model_rows["train_time"].mean()
                    
                    baseline_acc = baseline_rows["val_accuracy"].mean()
                    model_acc = model_rows["val_accuracy"].mean()
                    
                    print(f"\n{model_type.upper()} vs {baseline_type.upper()}:")
                    
                    if model_params > 0 and baseline_params > 0:
                        ratio = model_params / baseline_params
                        print(f"  Parameters: {ratio:.2f}x {'more' if ratio > 1 else 'fewer'}")
                    
                    if model_time > 0 and baseline_time > 0:
                        speedup = baseline_time / model_time
                        print(f"  Training: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
                    
                    if model_acc > 0 and baseline_acc > 0:
                        acc_diff = model_acc - baseline_acc
                        print(f"  Accuracy: {acc_diff:+.4f} ({model_acc:.4f} vs {baseline_acc:.4f})")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: results/analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()
