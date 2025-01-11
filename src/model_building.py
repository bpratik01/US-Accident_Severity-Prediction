import pandas as pd
import numpy as np
import os
import joblib
import yaml
from typing import Tuple, Dict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import QuantileTransformer
from logger import setup_logger
from datetime import datetime
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.utils.class_weight import compute_class_weight  # To compute class weights

def create_model_directory(base_dir: str, model_name: str) -> str:
    """Creates and returns a timestamped model directory."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped directory for this model run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir

def load_params(params_path: str) -> Dict:
    """Loads parameters from YAML file."""
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(file_path: str, logger) -> pd.DataFrame:
    """Loads feature engineered dataset."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def transform_features(df: pd.DataFrame, quantile_cols: list, logger) -> Tuple[pd.DataFrame, QuantileTransformer]:
    """Applies quantile transformation to numerical features."""
    logger.info("Applying quantile transformation")
    
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    df[quantile_cols] = transformer.fit_transform(df[quantile_cols])
    
    logger.info("Completed quantile transformation")
    return df, transformer

def prepare_data(df: pd.DataFrame, target_col: str, logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target for modeling."""
    logger.info("Preparing features and target")
    
    X = df.drop(columns=[target_col])
    y = df[target_col] - 1  # Adjusting target values
    
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float, random_state: int, logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits data into train, validation, and test sets."""
    logger.info("Performing train-validate-test split")
    
    # First split: train+val, test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Second split: train, val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_categorical(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, cat_cols: list, logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """Applies target encoding to categorical features."""
    logger.info("Applying target encoding to categorical features")
    
    encoder = TargetEncoder(cols=cat_cols)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    
    logger.info("Completed target encoding")
    return X_train_encoded, X_val_encoded, X_test_encoded, encoder

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE to balance the dataset."""
    logger.info("Applying SMOTE to balance the dataset")
    
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE applied: X_train shape {X_train_balanced.shape}, y_train shape {y_train_balanced.shape}")
    return X_train_balanced, y_train_balanced

def save_artifacts(model: XGBClassifier, transformer: QuantileTransformer, 
                  encoder: TargetEncoder, model_dir: str, params: Dict, logger) -> None:
    """Saves model and preprocessing artifacts."""
    logger.info(f"Saving model artifacts to {model_dir}")
    
    # Save model and preprocessors
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(transformer, os.path.join(model_dir, "quantile_transformer.joblib"))
    joblib.dump(encoder, os.path.join(model_dir, "target_encoder.joblib"))
    
    # Save parameters used for this model
    with open(os.path.join(model_dir, "model_params.yaml"), 'w') as f:
        yaml.dump(params, f)
    
    logger.info("Saved all artifacts successfully")

def build_model(params_path: str) -> None:
    """Orchestrates the model building pipeline."""
    # Load parameters
    params = load_params(params_path)
    
    # Create model directory
    model_dir = create_model_directory(params['paths']['model_dir'], params['paths']['model_name'])
    
    # Setup logging
    os.makedirs(params['paths']['log_dir'], exist_ok=True)
    log_file = os.path.join(params['paths']['log_dir'], f"model_building_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger('ModelBuilding', log_file)
    
    # Execute pipeline
    df = load_data(params['paths']['input_path'], logger)
    df, transformer = transform_features(df, params['feature_params']['quantile_cols'], logger)
    X, y = prepare_data(df, 'Severity', logger)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, 
        params['split_params']['test_size'],
        params['split_params']['val_size'],
        params['split_params']['random_state'],
        logger
    )
    
    X_train_encoded, X_val_encoded, X_test_encoded, encoder = encode_categorical(
        X_train, X_val, X_test, y_train, 
        params['feature_params']['categorical_cols'], 
        logger
    )
    
    # Apply SMOTE to balance training data
    X_train_balanced, y_train_balanced = apply_smote(X_train_encoded, y_train, logger)
    
    # Compute class weights based on the training set
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
    
    logger.info(f"Computed class weights: {class_weight_dict}")
    
    # Train model with class weights
    logger.info("Training model with specified parameters")
    model = XGBClassifier(**params['xgb_params'])
    
    # Pass the class weights to the model
    model.fit(X_train_balanced, y_train_balanced, sample_weight=y_train_balanced.map(class_weight_dict))
    
    # Save artifacts
    save_artifacts(model, transformer, encoder, model_dir, params, logger)
    logger.info(f"Model pipeline completed. Artifacts saved in {model_dir}")

if __name__ == "__main__":
    params_path = 'params.yaml'
    build_model(params_path)
