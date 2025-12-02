"""
Data loading and preprocessing utilities
for the three datasets: CoverType, HELOC, and HIGGS.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Data path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')


class DataLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_train_data(self):
        """Load training data (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def load_test_data(self):
        """Load test data (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def preprocess(self, X_train, X_test):
        """Standardize features using z-score normalization."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


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
        y = df['Cover_Type'].values
        
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
    
    def get_sample_submission(self):
        """Load the sample submission file for CoverType."""
        return pd.read_csv(os.path.join(self.data_dir, 'covtype_test_submission.csv'))


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
    
    def get_sample_submission(self):
        """Load the sample submission file for HELOC."""
        return pd.read_csv(os.path.join(self.data_dir, 'heloc_test_submission.csv'))


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
    
    def get_sample_submission(self):
        """Load the sample submission file for HIGGS."""
        return pd.read_csv(os.path.join(self.data_dir, 'higgs_test_submission.csv'))


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


def load_all_datasets():
    """
    Convenience function: load and standardize all three datasets.

    Returns a dict with entries 'covtype', 'heloc', and 'higgs',
    each containing train/test arrays and metadata.
    """
    datasets = {}
    for name in ['covtype', 'heloc', 'higgs']:
        loader = get_data_loader(name)
        X_train, y_train, feature_cols = loader.load_train_data()
        X_test, _ = loader.load_test_data()
        
        # Standardize features
        X_train_scaled, X_test_scaled = loader.preprocess(X_train, X_test)
        
        datasets[name] = {
            'X_train': X_train,
            'X_train_scaled': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'feature_cols': feature_cols,
            'loader': loader
        }
    return datasets


if __name__ == '__main__':
    # Quick smoke-test for data loading
    for name in ['covtype', 'heloc', 'higgs']:
        print(f"Dataset: {name}")
        
        loader = get_data_loader(name)
        X_train, y_train, feature_cols = loader.load_train_data()
        X_test, _ = loader.load_test_data()
        
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Unique labels: {np.unique(y_train)}")
        # For covtype labels start at 1, so subtract 1 for bincount
        if name != 'covtype':
            counts = np.bincount(y_train.astype(int))
        else:
            counts = np.bincount(y_train.astype(int) - 1)
        print(f"Label distribution (bincount): {counts}")
