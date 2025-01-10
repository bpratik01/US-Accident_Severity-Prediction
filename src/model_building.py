import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from category_encoders import TargetEncoder
from logger import setup_logger

# logger = setup_logger("model_building")


df = pd.read_csv('data/featured_data.csv')

df.head()