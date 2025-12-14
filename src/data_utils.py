"""
Data loading and preprocessing utilities
for the three datasets: CoverType, HELOC, and HIGGS.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

# Data path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
# Consistent dataset identifiers (used as an extra feature in the unified model).
DATASET_ID_MAP = {
    'covtype': 0,
    'heloc': 1,
    'higgs': 2,
}

# Label offsets to put all tasks into a single unified label space.
LABEL_OFFSET_MAP = {
    'covtype': 0,   # 7 classes -> 0..6
    'heloc': 7,     # 2 classes -> 7..8
    'higgs': 9,     # 2 classes -> 9..10
}

# Number of classes per dataset (used when mapping predictions back).
DATASET_CLASS_COUNTS = {
    'covtype': 7,
    'heloc': 2,
    'higgs': 2,
}


class DataLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        
    def load_train_data(self):
        """Load training data (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def load_test_data(self):
        """Load test data (to be implemented by subclasses)."""
        raise NotImplementedError


class CovTypeDataLoader(DataLoader):
    """CoverType dataset loader – 7-class classification problem."""
    
    def __init__(self, data_dir=DATA_DIR):
        super().__init__(data_dir)
        self.name = 'covtype'
        self.num_classes = 7
        self.id_start = 1  # IDs start from 1
        self.feature_cols = None
        
    def load_train_data(self):
        """Load CoverType training data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'covtype_train.csv'))
        
        # Feature columns (all except Cover_Type)
        self.feature_cols = [col for col in df.columns if col != 'Cover_Type']
        X = df[self.feature_cols].values
        y = df['Cover_Type'].values - 1
        
        return X, y, self.feature_cols
    
    def load_test_data(self):
        """Load CoverType test data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'covtype_test.csv'))
        
        # Use the feature columns determined from training
        if self.feature_cols is not None:
            X = df[self.feature_cols].values
        else:
            X = df.values
            self.feature_cols = df.columns.tolist()
        
        return X, self.feature_cols


class HELOCDataLoader(DataLoader):
    """HELOC dataset loader – binary classification problem."""
    
    def __init__(self, data_dir=DATA_DIR):
        super().__init__(data_dir)
        self.name = 'heloc'
        self.num_classes = 2
        self.id_start = 3501  # IDs start from 3501
        self.feature_cols = None
        
    def load_train_data(self):
        """Load HELOC training data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'heloc_train.csv'))
        
        # Target column (RiskPerformance)
        if 'RiskPerformance' in df.columns:
            target_col = 'RiskPerformance'
        else:
            # Fallback: assume the first column is the target
            target_col = df.columns[0]
        
        self.feature_cols = [col for col in df.columns if col != target_col]
        X = df[self.feature_cols].values
        y = df[target_col].values
        
        # Map labels to numeric values (Good=1, Bad=0)
        if y.dtype == object:
            y = np.where(y == 'Good', 1, 0)
        
        # Handle missing values (negative values indicate missing)
        X = self._handle_missing(X)
        
        return X, y, self.feature_cols
    
    def load_test_data(self):
        """Load HELOC test data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'heloc_test.csv'))
        
        # Use the feature columns determined from training
        if self.feature_cols is not None:
            X = df[self.feature_cols].values
        else:
            X = df.values
            self.feature_cols = df.columns.tolist()
        
        X = self._handle_missing(X)
        return X, self.feature_cols
    
    def _handle_missing(self, X):
        """
        Handle missing values.

        In this dataset, negative values indicate missingness.
        We replace them with the median of non-missing values in that column.
        """
        X = X.astype(float)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            mask = col < 0  # negative values = missing
            if mask.any():
                valid_values = col[~mask]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    X[mask, col_idx] = median_val
        return X


class HIGGSDataLoader(DataLoader):
    """HIGGS dataset loader – binary classification problem."""
    
    def __init__(self, data_dir=DATA_DIR):
        super().__init__(data_dir)
        self.name = 'higgs'
        self.num_classes = 2
        self.id_start = 4547  # IDs start from 4547
        self.feature_cols = None  # store feature names to keep consistency
        
    def load_train_data(self):
        """Load HIGGS training data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'higgs_train.csv'))
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        exclude_patterns = ['eventid', 'weight', 'label', 'id']
        
        # Find label column
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Cannot find label column in HIGGS training data")
        
        # Columns to exclude (IDs, weights, label, etc.)
        exclude_cols = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    exclude_cols.append(col)
                    break
        
        # Feature columns = all columns minus those to exclude
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_cols].values
        y = df[label_col].values
        
        # Convert labels to integers
        if y.dtype == object:
            y = np.where(y == 's', 1, 0)
        else:
            y = y.astype(int)
        
        # Handle missing values (-999.0)
        X = self._handle_missing(X)
        
        return X, y, self.feature_cols
    
    def load_test_data(self):
        """Load HIGGS test data."""
        df = pd.read_csv(os.path.join(self.data_dir, 'higgs_test.csv'))
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # If training has already been loaded, reuse the same feature columns
        if self.feature_cols is not None:
            # Use only the feature columns that exist in the test file
            available_cols = [col for col in self.feature_cols if col in df.columns]
            X = df[available_cols].values
        else:
            # Exclude possible ID / weight / label columns
            exclude_patterns = ['eventid', 'weight', 'label', 'id']
            feature_cols = []
            for col in df.columns:
                col_lower = col.lower()
                exclude = False
                for pattern in exclude_patterns:
                    if pattern in col_lower:
                        exclude = True
                        break
                if not exclude:
                    feature_cols.append(col)
            
            X = df[feature_cols].values
            self.feature_cols = feature_cols
        
        X = self._handle_missing(X)
        return X, self.feature_cols
    
    def _handle_missing(self, X):
        """
        Handle missing values in HIGGS.

        Here -999.0 indicates missing; we replace it with the median
        of the non-missing values in that column.
        """
        X = X.astype(float)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            mask = col == -999.0
            if mask.any():
                valid_values = col[~mask]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    X[mask, col_idx] = median_val
        return X


def get_data_loader(dataset_name):
    """Return the appropriate DataLoader instance for a given dataset name."""
    loaders = {
        'covtype': CovTypeDataLoader,
        'heloc': HELOCDataLoader,
        'higgs': HIGGSDataLoader
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    return loaders[dataset_name]()


# Unified (single model) helpers#

def _align_to_union_features(
    X: np.ndarray,
    feature_cols: List[str],
    union_features: List[str],
) -> np.ndarray:
    """
    Align a dataset to the union feature space by adding missing columns (filled with 0)
    and ordering columns consistently.
    """
    df = pd.DataFrame(X, columns=feature_cols)
    missing_cols = [c for c in union_features if c not in df.columns]
    for col in missing_cols:
        df[col] = 0
    # Reorder to the union feature order
    df = df[union_features]
    return df.values


def load_unified_train_data():
    """
    Load and merge all training splits into a single feature/label space.

    Returns
    -------
    X_all : np.ndarray
        Features concatenated across datasets, aligned to the union of columns,
        with an extra 'dataset_id' feature appended as the last column.
    y_all : np.ndarray
        Labels mapped into a single unified label space using LABEL_OFFSET_MAP.
    union_features_with_id : List[str]
        Ordered list of feature names used for training, including 'dataset_id' as the last entry.
    """
    train_parts: Dict[str, Dict] = {}
    union_features = set()

    # 1) Load each dataset and collect the union of feature names.
    for name in DATASET_ID_MAP.keys():
        loader = get_data_loader(name)
        X, y, feature_cols = loader.load_train_data()
        train_parts[name] = {
            "X": X,
            "y": y,
            "feature_cols": feature_cols,
        }
        union_features.update(feature_cols)

    union_features = sorted(list(union_features))

    # 2) Align, append dataset_id, and map labels into a unified space.
    X_all = []
    y_all = []
    for name, part in train_parts.items():
        X_aligned = _align_to_union_features(part["X"], part["feature_cols"], union_features)
        dataset_id = np.full((X_aligned.shape[0], 1), DATASET_ID_MAP[name])
        X_all.append(np.hstack([X_aligned, dataset_id]))
        y_all.append(part["y"] + LABEL_OFFSET_MAP[name])

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    union_features_with_id = union_features + ["dataset_id"]

    return X_all, y_all, union_features_with_id


def load_unified_test_data(union_features: List[str]):
    """
    Load and align all test splits to the unified feature space (plus dataset_id).

    Returns
    -------
    test_parts : Dict[str, Dict]
        Mapping dataset name -> {
            "X": np.ndarray aligned to union_features + dataset_id,
            "ids": np.ndarray of row IDs for submission,
        }
    """
    test_parts: Dict[str, Dict] = {}
    base_features = [f for f in union_features if f != "dataset_id"]

    for name in DATASET_ID_MAP.keys():
        loader = get_data_loader(name)
        # Load train first to ensure feature ordering, then test
        _, _, train_feature_cols = loader.load_train_data()
        X_test, _ = loader.load_test_data()
        X_aligned = _align_to_union_features(X_test, train_feature_cols, base_features)
        dataset_id = np.full((X_aligned.shape[0], 1), DATASET_ID_MAP[name])
        X_full = np.hstack([X_aligned, dataset_id])

        ids = np.arange(loader.id_start, loader.id_start + X_full.shape[0])
        test_parts[name] = {"X": X_full, "ids": ids}

    return test_parts


def map_global_to_dataset_labels(
    dataset_name: str,
    global_preds: np.ndarray,
) -> np.ndarray:
    """
    Map predictions from the unified label space back to per-dataset labels.

    If a prediction falls outside the expected range for the dataset, it is
    clamped to the first class of that dataset.
    """
    offset = LABEL_OFFSET_MAP[dataset_name]
    num_classes = DATASET_CLASS_COUNTS[dataset_name]
    low, high = offset, offset + num_classes - 1

    mapped = []
    for p in global_preds:
        if p < low or p > high:
            mapped.append(0)
        else:
            mapped.append(p - offset)
    return np.array(mapped, dtype=int)
