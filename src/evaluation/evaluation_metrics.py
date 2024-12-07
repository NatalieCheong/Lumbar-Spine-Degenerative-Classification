import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from typing import Dict, List, Tuple

# Import our model classes and dataset classes
from classification_model import LumbarClassifier, LumbarSpineDataset
from regression_model import LumbarRegressor, LumbarSpineRegDataset

def load_processed_data(filename: str) -> List[Dict]:
    """Load processed samples from numpy file"""
    processed_data = np.load(filename, allow_pickle=True).item()

    # Convert back to list of dictionaries format
    samples = []
    for i in range(len(processed_data['images'])):
        sample = {
            'image': processed_data['images'][i],
            'condition': processed_data['conditions'][i],
            'level': processed_data['levels'][i],
            'severity': processed_data['severities'][i],
            'study_id': processed_data['study_ids'][i]
        }
        samples.append(sample)

    return samples

def compute_competition_metric(predictions: np.ndarray,
                             true_labels: np.ndarray,
                             study_ids: np.ndarray,
                             conditions: List[str],
                             levels: List[str]) -> Tuple[float, Dict]:
    """
    Compute the competition metric: average of sample weighted log losses and any_severe_spinal
    """
    # Create DataFrame with all information
    results_df = pd.DataFrame({
        'study_id': study_ids,
        'condition': conditions,
        'level': levels
    })

    # Convert true labels to one-hot encoding
    true_labels_onehot = np.zeros((len(true_labels), 3))
    for i, label in enumerate(true_labels):
        true_labels_onehot[i, int(label)] = 1

    # Add true labels and predictions to DataFrame
    results_df['true_normal_mild'] = true_labels_onehot[:, 0]
    results_df['true_moderate'] = true_labels_onehot[:, 1]
    results_df['true_severe'] = true_labels_onehot[:, 2]
    results_df['pred_normal_mild'] = predictions[:, 0]
    results_df['pred_moderate'] = predictions[:, 1]
    results_df['pred_severe'] = predictions[:, 2]

    # Initialize metrics
    metrics = {}
    total_loss = 0
    total_weight = 0

    # Define condition weights
    condition_weights = {
        'Spinal Canal Stenosis': {
            'L1_L2': 0.08, 'L2_L3': 0.14, 'L3_L4': 0.22, 'L4_L5': 0.35, 'L5_S1': 0.21
        },
        'Left Neural Foraminal Narrowing': {
            'L1_L2': 0.07, 'L2_L3': 0.13, 'L3_L4': 0.21, 'L4_L5': 0.33, 'L5_S1': 0.26
        },
        'Right Neural Foraminal Narrowing': {
            'L1_L2': 0.07, 'L2_L3': 0.13, 'L3_L4': 0.21, 'L4_L5': 0.33, 'L5_S1': 0.26
        },
        'Left Subarticular Stenosis': {
            'L1_L2': 0.07, 'L2_L3': 0.13, 'L3_L4': 0.21, 'L4_L5': 0.33, 'L5_S1': 0.26
        },
        'Right Subarticular Stenosis': {
            'L1_L2': 0.07, 'L2_L3': 0.13, 'L3_L4': 0.21, 'L4_L5': 0.33, 'L5_S1': 0.26
        }
    }

    # Compute weighted log loss for each condition and level
    for condition in condition_weights.keys():
        for level in condition_weights[condition].keys():
            mask = (results_df['condition'] == condition) & (results_df['level'] == level)
            if mask.any():
                y_true = results_df.loc[mask, ['true_normal_mild', 'true_moderate', 'true_severe']].values
                y_pred = results_df.loc[mask, ['pred_normal_mild', 'pred_moderate', 'pred_severe']].values

                # Add small epsilon to avoid log(0)
                y_pred = np.clip(y_pred, 1e-7, 1-1e-7)

                try:
                    ll = log_loss(y_true, y_pred)
                    weight = condition_weights[condition][level]
                    total_loss += ll * weight
                    total_weight += weight
                    metrics[f'{condition}_{level}_log_loss'] = ll
                except ValueError as e:
                    print(f"Warning: Error computing log loss for {condition} {level}: {e}")
                    continue

    # Compute weighted average log loss
    if total_weight > 0:
        weighted_log_loss = total_loss / total_weight
    else:
        weighted_log_loss = 0
    metrics['weighted_log_loss'] = weighted_log_loss

    # Compute any_severe_spinal metric
    study_severe_true = []
    study_severe_pred = []

    for study_id in results_df['study_id'].unique():
        study_mask = results_df['study_id'] == study_id
        study_data = results_df[study_mask]

        # True severe cases
        has_severe = (study_data['true_severe'] == 1).any()
        study_severe_true.append(has_severe)

        # Predicted severe cases
        pred_severe = study_data['pred_severe'].max()
        study_severe_pred.append(pred_severe)

    # Add small epsilon to avoid log(0)
    study_severe_pred = np.clip(study_severe_pred, 1e-7, 1-1e-7)

    try:
        any_severe_loss = log_loss(study_severe_true, study_severe_pred)
    except ValueError as e:
        print(f"Warning: Error computing any_severe_loss: {e}")
        any_severe_loss = 0

    metrics['any_severe_loss'] = any_severe_loss

    # Compute final score
    final_score = 0.7 * weighted_log_loss + 0.3 * any_severe_loss
    metrics['final_score'] = final_score

    return final_score, metrics

def evaluate_model(model, val_loader, device):
    """Evaluate model using competition metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_study_ids = []
    all_conditions = []
    all_levels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            levels = batch['level'].to(device)

            # Get model predictions
            outputs = model(images, conditions, levels)
            probs = F.softmax(outputs, dim=1)

            # Store results
            all_predictions.append(probs.cpu().numpy())
            all_labels.append(batch['severity'].numpy())
            all_study_ids.extend(batch['study_id'])

            # Get condition and level names
            for c_idx, l_idx in zip(torch.argmax(conditions, dim=1).cpu().numpy(),
                                  torch.argmax(levels, dim=1).cpu().numpy()):
                all_conditions.append(list(val_loader.dataset.condition_map.keys())[c_idx])
                all_levels.append(list(val_loader.dataset.level_map.keys())[l_idx])

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    # Compute metrics
    score, metrics = compute_competition_metric(
        predictions,
        labels,
        np.array(all_study_ids),
        all_conditions,
        all_levels
    )

    return score, metrics

def evaluate_regression_model(model, val_loader, device):
    """Evaluate regression model with proper probability conversion"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_study_ids = []
    all_conditions = []
    all_levels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            levels = batch['level'].to(device)

            # Get regression predictions
            reg_output = model(images, conditions, levels)
            reg_output = reg_output.squeeze()  # Remove all extra dimensions

            # Convert regression values to probabilities
            probs = torch.zeros((reg_output.shape[0], 3), device=device)

            # Normal/Mild: values <= 0.5
            normal_mask = reg_output <= 0.5
            probs[normal_mask, 0] = 1.0

            # Moderate: values between 0.5 and 1.5
            moderate_mask = (reg_output > 0.5) & (reg_output <= 1.5)
            probs[moderate_mask, 1] = 1.0

            # Severe: values > 1.5
            severe_mask = reg_output > 1.5
            probs[severe_mask, 2] = 1.0

            # Add smoothing to avoid zero probabilities
            probs = probs + 1e-7
            probs = probs / probs.sum(dim=1, keepdim=True)

            # Store results
            all_predictions.append(probs.cpu().numpy())
            all_labels.append(batch['severity'].numpy())
            all_study_ids.extend(batch['study_id'])

            # Get condition and level names
            for c_idx, l_idx in zip(torch.argmax(conditions, dim=1).cpu().numpy(),
                                  torch.argmax(levels, dim=1).cpu().numpy()):
                all_conditions.append(list(val_loader.dataset.condition_map.keys())[c_idx])
                all_levels.append(list(val_loader.dataset.level_map.keys())[l_idx])

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    # Compute metrics
    score, metrics = compute_competition_metric(
        predictions,
        labels,
        np.array(all_study_ids),
        all_conditions,
        all_levels
    )

    return score, metrics

def main():
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation data
    print("Loading validation data...")
    val_samples = load_processed_data('/kaggle/input/spine-processed-data/val_processed.npy')

    # Create dataloaders
    val_dataset_class = LumbarSpineDataset(val_samples, augment=False)
    val_loader_class = DataLoader(
        val_dataset_class,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    val_dataset_reg = LumbarSpineRegDataset(val_samples, augment=False)
    val_loader_reg = DataLoader(
        val_dataset_reg,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    try:
        # Evaluate classification model
        print("\nEvaluating Classification Model...")
        class_model = LumbarClassifier().to(device)
        class_model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
        class_model.eval()

        class_score, class_metrics = evaluate_model(class_model, val_loader_class, device)

        print("\nClassification Model Results:")
        print(f"Final Score: {class_score:.4f}")
        print("\nDetailed Metrics:")
        for key, value in class_metrics.items():
            if not key.endswith('_log_loss'):
                print(f"{key}: {value:.4f}")

        print("\nPer-condition Log Losses:")
        for key, value in class_metrics.items():
            if key.endswith('_log_loss') and key != 'weighted_log_loss':
                print(f"{key}: {value:.4f}")

        # Evaluate regression model
        print("\nEvaluating Regression Model...")
        reg_model = LumbarRegressor().to(device)
        reg_model.load_state_dict(torch.load('best_regression_model.pth')['model_state_dict'])
        reg_model.eval()

        reg_score, reg_metrics = evaluate_regression_model(reg_model, val_loader_reg, device)

        print("\nRegression Model Results:")
        print(f"Final Score: {reg_score:.4f}")
        print("\nDetailed Metrics:")
        for key, value in reg_metrics.items():
            if not key.endswith('_log_loss'):
                print(f"{key}: {value:.4f}")

        print("\nPer-condition Log Losses:")
        for key, value in reg_metrics.items():
            if key.endswith('_log_loss') and key != 'weighted_log_loss':
                print(f"{key}: {value:.4f}")

        # Compare models
        print("\nModel Comparison:")
        print(f"Classification Model Score: {class_score:.4f}")
        print(f"Regression Model Score: {reg_score:.4f}")

        # Save results
        results = {
            'classification': {
                'score': class_score,
                'metrics': class_metrics
            },
            'regression': {
                'score': reg_score,
                'metrics': reg_metrics
            }
        }

        np.save('evaluation_results.npy', results)
        print("\nResults saved to evaluation_results.npy")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
