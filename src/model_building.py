import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from category_encoders import TargetEncoder
from logger import setup_logger


# Load the featured dataset
def load_data(file_path: str, logger) -> pd.DataFrame:
    """Loads the featured dataset from CSV."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

load_data(file_path='data/featured_data.csv', logger=setup_logger())

import pandas as pd
df = pd.read_csv(r'.\data\processed\featured_data.csv')