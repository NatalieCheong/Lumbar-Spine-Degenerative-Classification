import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

def analyze_regression_model(model_path: str, val_loader: DataLoader, device: torch.device):
    """Analyze regression model performance in detail"""
    # Load the best model
    model = LumbarRegressor().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize lists to store predictions and true values
    predictions = []
    true_values = []
    conditions = []
    levels = []
    study_ids = []
    weights = []

    # Get predictions
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            condition_tensor = batch['condition'].to(device)
            level_tensor = batch['level'].to(device)
            targets = batch['severity']

            outputs = model(images, condition_tensor, level_tensor)

            predictions.extend(outputs.cpu().numpy())
            true_values.extend(targets.numpy())
            weights.extend(batch['weight'].numpy())

            # Get condition and level names
            condition_indices = torch.argmax(condition_tensor, dim=1).cpu().numpy()
            level_indices = torch.argmax(level_tensor, dim=1).cpu().numpy()

            for c_idx, l_idx in zip(condition_indices, level_indices):
                conditions.append(list(val_loader.dataset.condition_map.keys())[c_idx])
                levels.append(list(val_loader.dataset.level_map.keys())[l_idx])

            study_ids.extend(batch['study_id'])

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)

    print("\nOverall Metrics:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Create DataFrame for detailed analysis
    results_df = pd.DataFrame({
        'Study_ID': study_ids,
        'True_Value': true_values.flatten(),
        'Predicted': predictions.flatten(),
        'Condition': conditions,
        'Level': levels,
        'Weight': weights,
        'Absolute_Error': np.abs(predictions.flatten() - true_values.flatten())
    })

    # Analyze by condition
    print("\nMAE by Condition:")
    condition_mae = results_df.groupby('Condition')['Absolute_Error'].mean()
    print(condition_mae)

    # Analyze by level
    print("\nMAE by Level:")
    level_mae = results_df.groupby('Level')['Absolute_Error'].mean()
    print(level_mae)

    # Plot prediction vs true values
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([0, 2], [0, 2], 'r--')  # Perfect prediction line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    plt.savefig('regression_scatter.png')
    plt.close()

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df['Absolute_Error'], bins=50)
    plt.title('Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.savefig('error_distribution.png')
    plt.close()

    # Plot MAE by condition
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    condition_mae.plot(kind='bar')
    plt.title('MAE by Condition')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot MAE by level
    plt.subplot(1, 2, 2)
    level_mae.plot(kind='bar')
    plt.title('MAE by Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mae_analysis.png')
    plt.close()

    # Error analysis by severity range
    def get_severity_range(value):
        if value <= 0.5:
            return 'Normal/Mild'
        elif value <= 1.5:
            return 'Moderate'
        else:
            return 'Severe'

    results_df['Severity_Range'] = results_df['True_Value'].apply(get_severity_range)

    print("\nMAE by Severity Range:")
    severity_mae = results_df.groupby('Severity_Range')['Absolute_Error'].mean()
    print(severity_mae)

    # Save detailed results
    results_df.to_csv('regression_results.csv', index=False)

    # Compare with classification model performance
    print("\nSeverity Range Distribution:")
    print(results_df['Severity_Range'].value_counts(normalize=True) * 100)

    return results_df

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation data
    val_samples = load_processed_data('/kaggle/input/spine-processed-data/val_processed.npy')
    val_dataset = LumbarSpineRegDataset(val_samples, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # Analyze model
    results_df = analyze_regression_model('best_regression_model.pth', val_loader, device)
    print("\nAnalysis completed and saved to files.")

if __name__ == "__main__":
    main()
