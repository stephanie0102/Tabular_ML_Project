"""
Error analysis tools.

Analyzes prediction errors:
- Find misclassified samples
- Analyze error patterns
- Visualize error characteristics
- Compare model disagreements
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import json


class ErrorAnalyzer:
    """Analyze prediction errors and model behavior."""
    
    def __init__(self, results_dir: str = "results/metrics", output_dir: str = "results/analysis"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_metrics(self, dataset_name: str, model_type: str) -> Optional[Dict]:
        """Load saved metrics."""
        metrics_path = os.path.join(
            self.results_dir,
            f"{dataset_name}_{model_type}_metrics.json"
        )
        
        if not os.path.exists(metrics_path):
            return None
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def find_errors(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Find prediction errors.
        
        Returns dict with:
        - error_indices: indices of misclassified samples
        - correct_indices: indices of correct predictions
        - confidence: prediction confidence (if y_proba provided)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        error_indices = np.where(y_true != y_pred)[0]
        correct_indices = np.where(y_true == y_pred)[0]
        
        result = {
            "error_indices": error_indices,
            "correct_indices": correct_indices,
            "n_errors": len(error_indices),
            "n_correct": len(correct_indices),
            "error_rate": len(error_indices) / len(y_true),
        }
        
        if y_proba is not None:
            # Confidence = max probability
            confidence = np.max(y_proba, axis=1)
            result["confidence"] = confidence
            result["error_confidence"] = confidence[error_indices]
            result["correct_confidence"] = confidence[correct_indices]
        
        return result
    
    def analyze_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_plot: bool = True,
        plot_name: str = "confusion_matrix.png"
    ) -> np.ndarray:
        """
        Analyze confusion matrix.
        
        Returns confusion matrix and optionally saves visualization.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if save_plot:
            plt.figure(figsize=(8, 6))
            #sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd')
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', annot_kws={"color": "black"})
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plot_path = os.path.join(self.output_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {plot_path}")
        
        if len(cm) == 2:  # 二分类
            TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
        return cm
    
    def find_high_confidence_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Find errors with highest prediction confidence.
        
        These are "confident mistakes" - interesting to analyze.
        """
        errors = self.find_errors(y_true, y_pred, y_proba)
        error_indices = errors["error_indices"]
        
        if len(error_indices) == 0:
            return pd.DataFrame()
        
        # Get confidence for errors
        confidence = np.max(y_proba[error_indices], axis=1)
        
        # Sort by confidence (descending)
        sorted_idx = np.argsort(confidence)[::-1][:top_k]
        top_error_indices = error_indices[sorted_idx]
        
        # Create report
        report = pd.DataFrame({
            "sample_index": top_error_indices,
            "true_label": y_true[top_error_indices],
            "predicted_label": y_pred[top_error_indices],
            "confidence": confidence[sorted_idx],
        })
        
        return report
    
    def analyze_model_from_metrics(
        self,
        dataset_name: str,
        model_type: str,
        save_report: bool = True
    ) -> Optional[Dict]:
        """
        Analyze errors from saved metrics file.
        """
        metrics = self.load_metrics(dataset_name, model_type)
        
        if metrics is None:
            print(f"No metrics found for {dataset_name} {model_type}")
            return None
        
        # Extract predictions
        y_true = np.array(metrics.get("val_true_labels", []))
        y_pred = np.array(metrics.get("val_predictions", []))
        
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"No validation predictions found")
            return None
        
        # Optional probabilities
        y_proba = metrics.get("val_probabilities")
        if y_proba is not None:
            y_proba = np.array(y_proba)
        
        # Find errors
        error_analysis = self.find_errors(y_true, y_pred, y_proba)
        
        # Confusion matrix
        cm = self.analyze_confusion(
            y_true, y_pred,
            save_plot=True,
            plot_name=f"{dataset_name}_{model_type}_confusion.png"
        )
        
        # High-confidence errors (if probabilities available)
        confident_errors = None
        if y_proba is not None:
            confident_errors = self.find_high_confidence_errors(
                y_true, y_pred, y_proba, top_k=5
            )
        
        result = {
            "dataset": dataset_name,
            "model_type": model_type,
            "n_samples": len(y_true),
            "n_errors": error_analysis["n_errors"],
            "error_rate": error_analysis["error_rate"],
            "accuracy": 1 - error_analysis["error_rate"],
            "confusion_matrix": cm.tolist(),
        }
        
        if confident_errors is not None and len(confident_errors) > 0:
            result["confident_errors"] = confident_errors.to_dict('records')
        
        if save_report:
            report_path = os.path.join(
                self.output_dir,
                f"{dataset_name}_{model_type}_error_analysis.json"
            )
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Error analysis saved to: {report_path}")
        
        return result

