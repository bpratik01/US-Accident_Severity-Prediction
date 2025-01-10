import pandas as pd
import numpy as np
import re
import os
from logger import setup_logger
from typing import List

def load_data(file_path: str, logger) -> pd.DataFrame:
    """Loads the cleaned dataset from CSV."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def handle_missing_values(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Fills missing values in categorical and numerical columns."""
    categorical_columns = ['City', 'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset']
    numerical_columns = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                        'Visibility(mi)', 'Wind_Speed(mph)']

    logger.info("Handling missing values")

    # Mode imputation for categorical
    for col in categorical_columns:
        before_count = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].mode()[0])
        logger.info(f"Filled {before_count} missing values in {col} using mode")

    # Median imputation for numerical
    for col in numerical_columns:
        before_count = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].median())
        logger.info(f"Filled {before_count} missing values in {col} using median")

    return df

def process_wind_direction(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Standardizes wind direction categories."""
    logger.info("Processing wind direction categories")

    mappings = {
        'Calm': 'CALM',
        'Variable': 'VAR',
        ('West', 'WSW', 'WNW'): 'W',
        ('South', 'SSW', 'SSE'): 'S',
        ('North', 'NNW', 'NNE'): 'N',
        ('East', 'ESE', 'ENE'): 'E'
    }

    for key, value in mappings.items():
        if isinstance(key, tuple):
            condition = df['Wind_Direction'].isin(key)
        else:
            condition = df['Wind_Direction'] == key
        df.loc[condition, 'Wind_Direction'] = value

    logger.info("Completed wind direction standardization")
    return df

def encode_weather_conditions(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Creates binary columns for weather conditions."""
    logger.info("Encoding weather conditions")

    weather_mappings = {
        'Clear': r'\bClear\b',
        'Cloud': r'\bCloud|Overcast\b',
        'Rain': r'\bRain|Shower|Storm\b',
        'Heavy_Rain': r'\bHeavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms\b',
        'Snow': r'\bSnow|Sleet|Ice|Hail\b',
        'Heavy_Snow': r'\bHeavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls\b',
        'Fog': r'\bFog|Mist|Haze\b',
        'Windy': r'\bWindy|Dust|Tornado\b',
        'Thunderstorm': r'\bT-Storm|Thunder\b'
    }

    for col, pattern in weather_mappings.items():
        df[col] = df['Weather_Condition'].str.contains(pattern, case=False, na=False).astype(int)
        logger.info(f"Created {col} column with {df[col].sum()} positive cases")

    df = df.drop(columns=['Weather_Condition'])
    return df

def encode_binary_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Converts binary and categorical features to numeric."""
    logger.info("Encoding binary and categorical features")

    # Convert binary columns - map True/False to 1/0 explicitly
    binary_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
                   'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
    
    for col in binary_cols:
        df[col] = df[col].map({True: 1, False: 0})
        logger.info(f"Converted {col} to binary (0/1): Counts - 1s: {df[col].sum()}, 0s: {len(df[col]) - df[col].sum()}")

    # Convert Sunrise_Sunset
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})

    # Create dummies for remaining categorical columns
    df = pd.get_dummies(df, columns=['Source', 'Wind_Direction'], drop_first=True)
    
    # Map dummy columns to 1/0 explicitly (if needed)
    dummy_cols = [col for col in df.columns if col.startswith('Source_') or col.startswith('Wind_Direction_')]
    for col in dummy_cols:
        df[col] = df[col].map({True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)
        logger.info(f"Mapped dummy column {col} to 0/1 explicitly")
    
    logger.info(f"Final dataframe shape: {df.shape}")
    return df



def process_features(input_path: str, output_path: str, log_file: str) -> pd.DataFrame:
    """Orchestrates the complete feature engineering process."""
    # Create log directory if needed
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger('FeatureEngineering', log_file)

    logger.info("Starting feature engineering process")

    df = load_data(input_path, logger)
    df = handle_missing_values(df, logger)
    df = process_wind_direction(df, logger)
    df = encode_weather_conditions(df, logger)
    df = encode_binary_features(df, logger)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Feature engineering completed successfully")

    return df

if __name__ == "__main__":
    input_path = 'data/interim/cleaned_data.csv'
    output_path = 'data/processed/featured_data.csv'
    log_file = 'logs/feature_engineering.log'

    processed_df = process_features(input_path, output_path, log_file)
