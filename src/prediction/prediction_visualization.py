import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
import torch

class PredictionVisualizer:
    """Visualization tools for the prediction pipeline"""
    def __init__(self):
        self.severity_classes = ['Normal/Mild', 'Moderate', 'Severe']
        self.colors = {
            'Normal/Mild': '#2ecc71',  # Green
            'Moderate': '#f1c40f',     # Yellow
            'Severe': '#e74c3c'        # Red
        }

    def plot_prediction_distribution(self, predictions: List[Dict[str, torch.Tensor]],
                                   save_path: str = None):
        """Plot distribution of predictions across severity classes"""
        plt.figure(figsize=(10, 6))

        # Convert predictions to numpy arrays
        all_probs = torch.cat([p['probabilities'] for p in predictions]).cpu().numpy()

        # Create boxplot
        bp = plt.boxplot([all_probs[:, i] for i in range(3)],
                        labels=self.severity_classes,
                        patch_artist=True)

        # Color boxes
        for patch, color in zip(bp['boxes'], self.colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        plt.title('Distribution of Prediction Probabilities by Class')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(f"{save_path}/prediction_distribution.png")
        plt.show()

    def plot_confidence_heatmap(self, predictions: List[Dict[str, torch.Tensor]],
                              conditions: List[str],
                              levels: List[str],
                              save_path: str = None):
        """Create heatmap of prediction confidence by condition and level"""
        confidence_matrix = np.zeros((len(set(conditions)), len(set(levels))))
        count_matrix = np.zeros_like(confidence_matrix)

        unique_conditions = list(set(conditions))
        unique_levels = list(set(levels))

        # Aggregate confidences
        for pred, cond, lvl in zip(predictions, conditions, levels):
            i = unique_conditions.index(cond)
            j = unique_levels.index(lvl)
            confidence_matrix[i, j] += pred['confidence'].item()
            count_matrix[i, j] += 1

        # Calculate average confidence
        avg_confidence = np.divide(confidence_matrix, count_matrix,
                                 where=count_matrix != 0)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_confidence,
                   xticklabels=unique_levels,
                   yticklabels=unique_conditions,
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd')

        plt.title('Average Prediction Confidence by Condition and Level')
        plt.xlabel('Spinal Level')
        plt.ylabel('Condition')

        if save_path:
            plt.savefig(f"{save_path}/confidence_heatmap.png")
        plt.show()

    def plot_severity_distribution(self, predictions: List[Dict[str, torch.Tensor]],
                                 conditions: List[str],
                                 save_path: str = None):
        """Plot severity distribution by condition"""
        severity_counts = {cond: [0, 0, 0] for cond in set(conditions)}

        # Count predictions for each severity level
        for pred, cond in zip(predictions, conditions):
            severity = pred['severity_prediction'].item()
            severity_counts[cond][severity] += 1

        # Create stacked bar chart
        df = pd.DataFrame(severity_counts, index=self.severity_classes).T

        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', stacked=True, color=[self.colors[c] for c in self.severity_classes])

        plt.title('Predicted Severity Distribution by Condition')
        plt.xlabel('Condition')
        plt.ylabel('Count')
        plt.legend(title='Severity', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/severity_distribution.png")
        plt.show()

    def plot_confidence_histogram(self, predictions: List[Dict[str, torch.Tensor]],
                                save_path: str = None):
        """Plot histogram of prediction confidences"""
        confidences = [pred['confidence'].item() for pred in predictions]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')

        plt.title('Distribution of Prediction Confidences')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(f"{save_path}/confidence_histogram.png")
        plt.show()

    def plot_prediction_changes(self, predictions: List[Dict[str, torch.Tensor]],
                              save_path: str = None):
        """Compare original vs adjusted predictions"""
        orig_probs = torch.cat([p['original_probabilities'] for p in predictions]).cpu().numpy()
        adj_probs = torch.cat([p['probabilities'] for p in predictions]).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, severity in enumerate(self.severity_classes):
            axes[i].scatter(orig_probs[:, i], adj_probs[:, i],
                          alpha=0.5, color=list(self.colors.values())[i])
            axes[i].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[i].set_title(f'{severity} Probabilities')
            axes[i].set_xlabel('Original')
            axes[i].set_ylabel('Adjusted')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/prediction_changes.png")
        plt.show()

def visualize_predictions(pipeline, dataloader, save_path=None):
    """Generate comprehensive visualization of predictions"""
    # Get predictions
    predictions = pipeline.batch_predict(dataloader)

    # Get conditions and levels from dataloader
    conditions = []
    levels = []
    for batch in dataloader:
        for c, l in zip(batch['condition'], batch['level']):
            c_idx = torch.argmax(c).item()
            l_idx = torch.argmax(l).item()
            conditions.append(list(pipeline.condition_weights.keys())[c_idx])
            levels.append(list(pipeline.level_weights.keys())[l_idx])

    # Create visualizer
    visualizer = PredictionVisualizer()

    # Generate all plots
    print("Generating visualization plots...")

    print("\n1. Prediction Distribution")
    visualizer.plot_prediction_distribution(predictions, save_path)

    print("\n2. Confidence Heatmap")
    visualizer.plot_confidence_heatmap(predictions, conditions, levels, save_path)

    print("\n3. Severity Distribution")
    visualizer.plot_severity_distribution(predictions, conditions, save_path)

    print("\n4. Confidence Histogram")
    visualizer.plot_confidence_histogram(predictions, save_path)

    print("\n5. Prediction Adjustments")
    visualizer.plot_prediction_changes(predictions, save_path)

    print("\nVisualization completed!")

def main():
    """Example usage of visualization tools"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and create pipeline
    model = LumbarClassifier().to(device)
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    pipeline = OptimizedPredictionPipeline(model, device)

    # Load validation data
    val_samples = load_processed_data('/kaggle/input/spine-processed-data/val_processed.npy')
    val_dataset = LumbarSpineDataset(val_samples, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Generate visualizations
    visualize_predictions(pipeline, val_loader)#, save_path='/kaggle/working/visualizations')

if __name__ == "__main__":
    main()
