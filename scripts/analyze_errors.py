"""
Error Analysis Script

Analyzes prediction errors:
- Find misclassified samples
- Confusion matrices
- High-confidence errors
- Error patterns
- Feature analysis of errors

Usage:
    python scripts/analyze_errors.py --dataset covtype --model lgbm
    python scripts/analyze_errors.py --dataset heloc --model lgbm --compare tabpfn
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.error_analysis import ErrorAnalyzer
from data_utils import get_data_loader


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction errors"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["covtype", "heloc", "higgs"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type to analyze (e.g., lgbm, xgb, tabpfn)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare with another model (e.g., tabpfn)"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of error examples to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer()
    
    print("=" * 70)
    print(f"ERROR ANALYSIS: {args.dataset.upper()} - {args.model.upper()}")
    print("=" * 70)
    
    # Analyze primary model
    print(f"\nAnalyzing {args.model}...")
    result = analyzer.analyze_model_from_metrics(
        dataset_name=args.dataset,
        model_type=args.model,
        save_report=True
    )
    
    if result is None:
        print(f"\nERROR: No metrics found for {args.dataset} {args.model}")
        print("Please train the model first with:")
        print(f"  python src/train.py --dataset {args.dataset} --model {args.model}")
        return
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples:     {result['n_samples']}")
    print(f"Correct:           {result['n_samples'] - result['n_errors']} ({(1-result['error_rate'])*100:.2f}%)")
    print(f"Errors:            {result['n_errors']} ({result['error_rate']*100:.2f}%)")
    print(f"Accuracy:          {result['accuracy']:.4f}")
    
    # Show high-confidence errors
    if 'confident_errors' in result and len(result['confident_errors']) > 0:
        print("\n" + "=" * 70)
        print(f"TOP {min(args.show_examples, len(result['confident_errors']))} HIGH-CONFIDENCE ERRORS")
        print("=" * 70)
        print("💡 Analysis: These are the model's most 'confident mistakes'")
        print("   - High confidence but wrong prediction")
        print("   - May indicate:")
        print("     • Systematic bias in the model")
        print("     • Confusing patterns in the data")
        print("     • Label noise or edge cases")
        print()
        
        errors_df = pd.DataFrame(result['confident_errors'])
        for idx, row in errors_df.head(args.show_examples).iterrows():
            print(f"\nExample {idx + 1}:")
            print(f"  Sample index:    {int(row['sample_index'])}")
            print(f"  True label:      {int(row['true_label'])}")
            print(f"  Predicted:       {int(row['predicted_label'])}")
            print(f"  Confidence:      {row['confidence']:.2%}")
            
        # Add interpretation
        print("\n" + "-" * 70)
        print("📊 Interpretation:")
        avg_conf = errors_df.head(args.show_examples)['confidence'].mean()
        print(f"   Average confidence on these errors: {avg_conf:.2%}")
        
        # Check if there's a pattern in errors
        true_labels = errors_df.head(args.show_examples)['true_label'].values
        pred_labels = errors_df.head(args.show_examples)['predicted_label'].values
        most_common_true = pd.Series(true_labels).mode()[0] if len(true_labels) > 0 else None
        most_common_pred = pd.Series(pred_labels).mode()[0] if len(pred_labels) > 0 else None
        
        if most_common_true is not None and most_common_pred is not None:
            print(f"   Pattern detected: Model often confuses class {int(most_common_true)} → {int(most_common_pred)}")
            print(f"   → This suggests the model may struggle to distinguish these classes")
    
    # Compare with another model
    if args.compare:
        print("\n" + "=" * 70)
        print(f"COMPARISON WITH {args.compare.upper()}")
        print("=" * 70)
        
        compare_result = analyzer.analyze_model_from_metrics(
            dataset_name=args.dataset,
            model_type=args.compare,
            save_report=True
        )
        
        if compare_result:
            print(f"\n{args.model.upper()} Accuracy: {result['accuracy']:.4f}")
            print(f"{args.compare.upper()} Accuracy: {compare_result['accuracy']:.4f}")
            print(f"Difference: {result['accuracy'] - compare_result['accuracy']:+.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: results/analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()
