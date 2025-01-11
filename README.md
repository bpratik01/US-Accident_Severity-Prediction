

# US Accident Severity Prediction using XGBoost  

This project focuses on predicting accident severity in the United States using advanced machine learning techniques, specifically XGBoost. By leveraging historical accident data and applying data preprocessing, feature engineering, and model tuning, we aim to create a reliable prediction system that can aid decision-making for traffic management and public safety.

---

## Project Overview  

The main goal of this project is to predict the **severity of accidents** based on historical data. We have taken a methodical approach to build and evaluate machine learning models, incorporating steps like data cleaning, feature engineering, hyperparameter tuning, and performance optimization.  

Some key highlights:  
- **Model Used**: XGBoost, chosen for its robustness and ability to handle imbalanced datasets.  
- **Techniques**: We utilized **SMOTE** and **class weights** to address imbalanced classes and ensure the model performs well across all severity levels.  
- **Optimization**: Hyperparameter tuning was performed to maximize model accuracy and F1 scores.  

---

## Project Structure  

The project is organized into the following directories:  

```plaintext
├── data
│   ├── raw            <- Original, unprocessed data.
│   ├── interim        <- Data after initial cleaning and transformations.
│   ├── processed      <- Final datasets used for modeling.
│   ├── external       <- Any third-party data sources used.
│
├── models
│   └── xgb_model_<timestamp> <- Serialized XGBoost models and evaluation results.
│
├── notebooks          <- Jupyter notebooks for exploration and experiments.
│
├── src
│   ├── data_cleaning.py       <- Scripts for cleaning and preprocessing raw data.
│   ├── feature_engineering.py <- Feature engineering and transformation logic.
│   ├── model_building.py      <- Code for training and tuning XGBoost models.
│   ├── model_eval.py          <- Evaluation scripts for the trained models.
│   ├── logger.py              <- Logging utility for tracking progress and debugging.
│
├── requirements.txt   <- Python libraries and dependencies used in this project.
├── README.md          <- You are here!
├── .env               <- Configuration file for managing environment variables.
├── .gitignore         <- Files and folders excluded from version control.
├── LICENSE            <- Licensing information for the project.
```

---

## What’s Been Done So Far  

1. **Data Cleaning**: 
   - Removed redundant or missing entries.  
   - Standardized columns for consistency.  

2. **Feature Engineering**:  
   - Created meaningful features such as weather conditions, traffic volume, and time of the day.  
   - Applied scaling and encoding where necessary.  

3. **Model Training and Tuning**:  
   - Implemented **XGBoost** as the primary model.  
   - Addressed **class imbalance** using SMOTE and class weights.  
   - Tuned hyperparameters to boost performance (accuracy: ~87.65%, weighted F1-score: ~87.77%).  

4. **Evaluation**:  
   - Assessed precision, recall, and F1-score across all severity levels.  
   - Validated the robustness of the model with extensive testing.  

---


## How to Use  

1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd US-Accident_Severity-Prediction
   ```

2. Set up the environment:  
   - Install dependencies from `requirements.txt`:  
     ```bash
     pip install -r requirements.txt
     ```  
   - Configure the `.env` file with necessary variables.  

3. Initialize DVC:  
   - If you haven't already initialized DVC, do so with the following command:  
     ```bash
     dvc init
     ```

4. Reproduce the results using DVC:  
   - Run the DVC pipeline to reproduce the entire process (data preprocessing, model training, and evaluation):  
     ```bash
     dvc repro
     ```

This command will automatically execute the necessary steps and rebuild the model if the data or code has changed.

---

Let me know if there's anything else you want to adjust!

## Results  

Our model demonstrates significant improvement in predicting accident severity, with optimized performance across imbalanced classes. The following are key metrics:  

| Class | Precision | Recall | F1-Score |  
|-------|-----------|--------|----------|  
| Low Severity (0) | 45.58%  | 65.83%   | 53.86%   |  
| Moderate Severity (1) | 93.08% | 91.85% | 92.46%   |  
| High Severity (2) | 71.38%  | 74.81%   | 73.05%   |  
| Severe (3) | 52.84%  | 49.99%   | 51.38%   |  

---

## Future Improvements  
- Enhance feature engineering by incorporating geographic and demographic data.   

---

Feel free to tweak further!
