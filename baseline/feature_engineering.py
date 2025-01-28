import pandas as pd
from typing import List, Tuple, Optional
from openfe import OpenFE, transform
import logging

def feature_engineering(train_df: pd.DataFrame, 
                        test_df: pd.DataFrame, 
                        columns_to_ignore: Optional[List[str]] = None,
                        target_column: str = 'target',
                        n_jobs: int = 56,
                        debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform automated feature engineering using OpenFE.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        columns_to_ignore: Columns to exclude from feature engineering
        target_column: Name of the target column (required by OpenFE)
        n_jobs: Number of parallel jobs to run
        debug: Enable debug output
        
    Returns:
        Tuple of engineered training and test DataFrames
    """
    # Validate inputs
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
        
    if columns_to_ignore and target_column in columns_to_ignore:
        raise ValueError("Target column cannot be in columns_to_ignore")

    # Create copies to avoid modifying original DataFrames
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Remove ignored columns
    if columns_to_ignore:
        train_df = train_df.drop(columns=[c for c in columns_to_ignore if c in train_df.columns])
        test_df = test_df.drop(columns=[c for c in columns_to_ignore if c in test_df.columns])

    if debug:
        initial_memory = train_df.memory_usage().sum() + test_df.memory_usage().sum()
        print(f"Initial shape: Train {train_df.shape}, Test {test_df.shape}")
        print(f"Initial memory usage: {initial_memory / 1024**2:.2f} MB")

    try:
        # Initialize OpenFE with proper target specification
        ofe = OpenFE()
        
        # Generate features using OpenFE
        features = ofe.fit(
            data=train_df.drop(columns=[target_column]),
            label=train_df[target_column],
            n_jobs=n_jobs,
            verbose=debug
        )
        
        # Apply transformations
        train_df, test_df = transform(
            train_df.drop(columns=[target_column]),
            test_df,
            features,
            n_jobs=n_jobs
        )
        
        # Add back target column
        train_df[target_column] = train_df[target_column]
        
    except Exception as e:
        logging.error("Feature engineering failed: %s", str(e))
        raise

    if debug:
        final_memory = train_df.memory_usage().sum() + test_df.memory_usage().sum()
        print(f"Final shape: Train {train_df.shape}, Test {test_df.shape}")
        print(f"Memory change: {final_memory / 1024**2:.2f} MB ({(final_memory - initial_memory) / 1024**2:+.2f} MB)")

    return train_df, test_df