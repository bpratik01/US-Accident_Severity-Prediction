├── data
│   ├── raw            <- Original, unprocessed data.
│   ├── interim        <- Data after initial cleaning and transformations.
│   ├── processed      <- Final datasets used for modeling.
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
