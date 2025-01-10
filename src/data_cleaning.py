# data_ingestion.py
import pandas as pd
from typing import List
from logger import setup_logger
from datetime import datetime
import os

def read_data(file_path: str, logger) -> pd.DataFrame:
    """Reads and returns data from a CSV file."""
    logger.info(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully read {len(df)} rows")
    return df

def convert_datetime_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Converts Start_Time and End_Time to datetime format."""
    logger.info("Converting Start_Time and End_Time to datetime")
    
    for col in ['Start_Time', 'End_Time']:
        df[col] = pd.to_datetime(df[col], format='mixed')
        logger.info(f"Converted {col} to datetime. Null values: {df[col].isnull().sum()}")
    
    return df

def calculate_total_time(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Adds Total_Time column with duration in seconds between start and end times."""
    logger.info("Calculating Total_Time column")
    df['Total_Time'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()
    
    negative_times = (df['Total_Time'] < 0).sum()
    if negative_times > 0:
        logger.warning(f"Found {negative_times} records with negative time duration")
    
    logger.info(f"Average time duration: {df['Total_Time'].mean():.2f} seconds")
    return df

def drop_columns(df: pd.DataFrame, columns_to_drop: List[str], logger) -> pd.DataFrame:
    """Removes specified columns from the dataframe."""
    logger.info("Dropping unnecessary columns")
    
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    logger.info(f"Remaining columns: {', '.join(df.columns)}")
    return df

def save_data(df: pd.DataFrame, output_path: str, logger) -> None:
    """Saves processed data to CSV file."""
    logger.info(f"Saving cleaned data to {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(df)} rows")

def ingest_data(file_path: str, log_file: str, output_path: str) -> pd.DataFrame:
    """Handles the complete data processing pipeline and returns cleaned dataframe."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = setup_logger('DataIngestion', log_file)
    logger.info(f"Starting data ingestion process at {datetime.now()}")
    
    columns_to_drop = [
        'ID', 'Description', 'Street', 'Zipcode', 'Country',
        'Timezone', 'Airport_Code', 'Weather_Timestamp',
        'Turning_Loop', 'Nautical_Twilight', 'Civil_Twilight',
        'Astronomical_Twilight', 'End_Lat', 'End_Lng', 'Wind_Chill(F)', 
        'Precipitation(in)', 'End_Time', 'Start_Time'
    ]
    
    df = read_data(file_path, logger)
    df = convert_datetime_columns(df, logger)
    df = calculate_total_time(df, logger)
    df = drop_columns(df, columns_to_drop, logger)
    
    save_data(df, output_path, logger)
    logger.info(f"Completed data ingestion process at {datetime.now()}")
    return df

if __name__ == "__main__":
    file_path = './data/raw/sampled_data.csv'
    log_file = './logs/data_ingestion.log'
    output_path = './data/interim/cleaned_data.csv'
    processed_df = ingest_data(file_path, log_file, output_path)