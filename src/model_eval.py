import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import yaml

def load_latest_model(models_dir: str) -> tuple:
    """Load the most recent model and its artifacts."""
    # Get all model directories
    model_dirs = glob(os.path.join(models_dir, "*"))
    if not model_dirs:
        raise ValueError("No model directories found")
    
    # Get the most recent model directory
    latest_model_dir = max(model_dirs, key=os.path.getctime)
    
    # Load model and artifacts
    model = joblib.load(os.path.join(latest_model_dir, "model.joblib"))
    transformer = joblib.load(os.path.join(latest_model_dir, "quantile_transformer.joblib"))
    encoder = joblib.load(os.path.join(latest_model_dir, "target_encoder.joblib"))
    
    with open(os.path.join(latest_model_dir, "model_params.yaml"), 'r') as f:
        params = yaml.safe_load(f)
    
    return model, transformer, encoder, params

def prepare_evaluation_data(data_path: str, transformer, encoder, params):
    """Prepare data for evaluation using saved transformers."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Transform features
    df[params['feature_params']['quantile_cols']] = transformer.transform(df[params['feature_params']['quantile_cols']])
    
    # Prepare X and y
    X = df.drop(columns=['Severity'])
    y = df['Severity'] - 1
    
    # Encode categorical features
    X_encoded = encoder.transform(X)
    
    return X_encoded, y

def create_evaluation_artifacts(model, X, y):
    """Create and save evaluation artifacts."""
    # Ensure evaluation directory exists
    os.makedirs('evaluation', exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average='weighted'),
        "classification_report": classification_report(y, y_pred, output_dict=True)
    }
    
    # Save metrics
    with open('evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png')
    plt.close()
    
    # Create and save feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('evaluation/feature_importance.png')
    plt.close()

def main():
    """Main function to run model evaluation."""
    # Load latest model and artifacts
    model, transformer, encoder, params = load_latest_model('models')
    
    # Prepare evaluation data
    X, y = prepare_evaluation_data(
        'data/processed/featured_data.csv',
        transformer,
        encoder,
        params
    )
    
    # Create evaluation artifacts
    create_evaluation_artifacts(model, X, y)

if __name__ == "__main__":
    main()