"""
Computational complexity analysis tools.

Analyzes and compares:
- Model parameters
- Training time
- Inference time
- Model size (disk/memory)
- FLOPs (optional)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class ComplexityAnalyzer:
    """Analyze computational complexity of models."""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "results/metrics"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def count_lightgbm_params(self, model) -> int:
        """
        Count parameters in a LightGBM model.
        
        Approximation: sum of nodes across all trees.
        Each node has: feature_id, threshold, left_child, right_child.
        """
        try:
            # Get boosters
            boosters = model.model.booster_.dump_model()
            total_nodes = 0
            
            # Count tree structures
            for tree in boosters['tree_info']:
                total_nodes += self._count_tree_nodes(tree['tree_structure'])
            
            # Each node stores ~4-5 parameters
            return total_nodes * 5
        except Exception as e:
            print(f"Warning: Could not count LightGBM params: {e}")
            return -1
    
    def _count_tree_nodes(self, tree_structure: Dict) -> int:
        """Recursively count nodes in a tree."""
        if 'leaf_value' in tree_structure:
            return 1  # Leaf node
        
        count = 1  # Current node
        if 'left_child' in tree_structure:
            count += self._count_tree_nodes(tree_structure['left_child'])
        if 'right_child' in tree_structure:
            count += self._count_tree_nodes(tree_structure['right_child'])
        
        return count
    
    def count_xgboost_params(self, model) -> int:
        """Count parameters in an XGBoost model."""
        try:
            # Get number of trees and average depth
            n_trees = model.model.n_estimators
            # Estimate: binary tree with depth d has ~2^d nodes
            # XGBoost default max_depth is 6
            max_depth = model.model.max_depth
            nodes_per_tree = 2 ** (max_depth + 1) - 1
            return n_trees * nodes_per_tree * 5
        except Exception as e:
            print(f"Warning: Could not count XGBoost params: {e}")
            return -1
    
    def count_random_forest_params(self, model) -> int:
        """Count parameters in a Random Forest model."""
        try:
            n_trees = model.model.n_estimators
            # Estimate based on tree depth
            total_nodes = sum(
                tree.tree_.node_count for tree in model.model.estimators_
            )
            return total_nodes * 5
        except Exception as e:
            print(f"Warning: Could not count RF params: {e}")
            return -1
    
    def count_tabpfn_params(self) -> int:
        """
        Return TabPFN parameter count.
        
        TabPFN is a pretrained Transformer. Parameter count from paper:
        TabPFN v2: ~12M parameters
        """
        return 12_000_000
    
    def count_model_params(self, model, model_type: str) -> int:
        """Count parameters for any model type."""
        model_type = model_type.lower()
        
        if model_type in ["lgbm", "lightgbm"]:
            return self.count_lightgbm_params(model)
        elif model_type in ["xgb", "xgboost"]:
            return self.count_xgboost_params(model)
        elif model_type in ["rf", "random_forest"]:
            return self.count_random_forest_params(model)
        elif model_type in ["baseline", "tabpfn"]:
            return self.count_tabpfn_params()
        else:
            return -1
    
    def get_model_size(self, dataset_name: str, model_type: str) -> float:
        """Get model file size in MB."""
        model_path = os.path.join(
            self.models_dir, 
            f"{dataset_name}_{model_type}_model.pkl"
        )
        
        if not os.path.exists(model_path):
            return -1
        
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def load_metrics(self, dataset_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Load saved metrics from JSON file."""
        import json
        
        metrics_path = os.path.join(
            self.results_dir,
            f"{dataset_name}_{model_type}_metrics.json"
        )
        
        if not os.path.exists(metrics_path):
            return None
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def analyze_model(self, dataset_name: str, model_type: str) -> Dict[str, Any]:
        """
        Comprehensive complexity analysis for a single model.
        
        Returns dict with:
        - n_parameters: number of model parameters
        - model_size_mb: disk size
        - train_time: training time (if available)
        - inference_time: prediction time (if available)
        - accuracy: validation accuracy
        """
        # Load metrics if available
        metrics = self.load_metrics(dataset_name, model_type)
        
        # Load model to count parameters
        model_path = os.path.join(
            self.models_dir,
            f"{dataset_name}_{model_type}_model.pkl"
        )
        
        result = {
            "dataset": dataset_name,
            "model_type": model_type,
            "n_parameters": -1,
            "model_size_mb": self.get_model_size(dataset_name, model_type),
        }
        
        # Add metrics if available
        if metrics:
            result.update({
                "train_time": metrics.get("train_time", -1),
                "inference_time": metrics.get("inference_time", -1),
                "val_accuracy": metrics.get("val_accuracy", -1),
                "cv_mean": metrics.get("cv_mean", -1),
                "cv_std": metrics.get("cv_std", -1),
            })
        
        # Count parameters
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                result["n_parameters"] = self.count_model_params(model, model_type)
            except Exception as e:
                print(f"Warning: Could not load model {model_path}: {e}")
        
        return result
    
    def compare_models(
        self, 
        dataset_name: str, 
        model_types: list,
        save_report: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.
        
        Returns DataFrame with comparison metrics.
        """
        results = []
        
        for model_type in model_types:
            analysis = self.analyze_model(dataset_name, model_type)
            results.append(analysis)
        
        df = pd.DataFrame(results)
        
        # Calculate speedup/compression ratios if baseline exists
        if "tabpfn" in model_types or "baseline" in model_types:
            baseline_idx = (
                model_types.index("tabpfn") 
                if "tabpfn" in model_types 
                else model_types.index("baseline")
            )
            
            baseline = results[baseline_idx]
            
            # Add relative metrics
            for i, row in enumerate(results):
                if i != baseline_idx:
                    # Speedup (baseline_time / model_time)
                    if baseline.get("train_time", -1) > 0 and row.get("train_time", -1) > 0:
                        df.loc[i, "train_speedup"] = baseline["train_time"] / row["train_time"]
                    
                    # Compression (baseline_params / model_params)
                    if baseline.get("n_parameters", -1) > 0 and row.get("n_parameters", -1) > 0:
                        df.loc[i, "param_ratio"] = row["n_parameters"] / baseline["n_parameters"]
        
        if save_report:
            report_path = os.path.join(
                "results/analysis",
                f"{dataset_name}_complexity_comparison.csv"
            )
            df.to_csv(report_path, index=False)
            print(f"Complexity comparison saved to: {report_path}")
        
        return df
    
    def generate_summary_report(self, model_types: list = ["lgbm", "tabpfn"]) -> pd.DataFrame:
        """Generate summary report across all datasets."""
        datasets = ["covtype", "heloc", "higgs"]
        all_results = []
        
        for dataset in datasets:
            for model_type in model_types:
                result = self.analyze_model(dataset, model_type)
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        
        # Save summary
        summary_path = os.path.join("results/analysis", "complexity_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Summary report saved to: {summary_path}")
        
        return df
