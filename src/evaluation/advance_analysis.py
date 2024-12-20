import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix

class AdvancedAnalysis:
    def __init__(self, pipeline, preprocessed_path: str):
        self.pipeline = pipeline
        self.preprocessed_data = np.load(preprocessed_path, allow_pickle=True).item()
        self.severity_classes = ['Normal/Mild', 'Moderate', 'Severe']

    def analyze_prediction_patterns(self, num_samples: int = 1000):
        """Perform statistical analysis of prediction patterns with error handling"""
        total_samples = len(self.preprocessed_data['images'])
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

        results = {
            'condition_level_accuracy': defaultdict(lambda: defaultdict(list)),
            'confidence_scores': [],
            'true_vs_pred': [],
            'level_difficulty': defaultdict(list),
            'condition_difficulty': defaultdict(list)
        }

        print("Analyzing prediction patterns...")
        processed_samples = 0

        for idx in sample_indices:
            try:
                # Get sample data
                image = self.preprocessed_data['images'][idx]
                condition = self.preprocessed_data['conditions'][idx]
                level = self.preprocessed_data['levels'][idx]
                true_severity = self.preprocessed_data['severities'][idx]

                # Get prediction
                prediction = self.pipeline.predict(image, condition, level)
                pred_class = prediction['severity_prediction'].item()
                confidence = prediction['confidence'].item()

                # Store results
                results['confidence_scores'].append(confidence)
                results['true_vs_pred'].append((true_severity, self.severity_classes[pred_class]))
                results['condition_level_accuracy'][condition][level].append(
                    true_severity == self.severity_classes[pred_class]
                )
                results['level_difficulty'][level].append(confidence)
                results['condition_difficulty'][condition].append(confidence)

                processed_samples += 1
                if processed_samples % 100 == 0:
                    print(f"Processed {processed_samples}/{num_samples} samples...")

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue

        print(f"\nCompleted analysis of {processed_samples} samples")
        self._visualize_statistics(results)
        return results

    def _visualize_statistics(self, results: Dict):
        """Visualize statistical analysis results with fixed accuracy calculations"""
        # 1. Overall Confidence Distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.histplot(results['confidence_scores'], bins=30)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')

        # 2. Level-wise Accuracy
        level_acc = {}
        for level in set([l for d in results['condition_level_accuracy'].values() for l in d.keys()]):
            level_values = []
            for cond_data in results['condition_level_accuracy'].values():
                if level in cond_data:
                    level_values.extend(cond_data[level])
            if level_values:
                level_acc[level] = np.mean(level_values)

        plt.subplot(1, 3, 2)
        if level_acc:
            plt.bar(level_acc.keys(), level_acc.values())
            plt.title('Accuracy by Spinal Level')
            plt.xticks(rotation=45)

        # 3. Condition-wise Accuracy
        condition_acc = {}
        for condition, level_data in results['condition_level_accuracy'].items():
            condition_values = []
            for level_list in level_data.values():
                condition_values.extend(level_list)
            if condition_values:
                condition_acc[condition] = np.mean(condition_values)

        plt.subplot(1, 3, 3)
        if condition_acc:
            plt.bar(range(len(condition_acc)), condition_acc.values())
            plt.xticks(range(len(condition_acc)),
                      [c.split()[0] for c in condition_acc.keys()],
                      rotation=45)
            plt.title('Accuracy by Condition')

        plt.tight_layout()
        plt.show()

        # 4. Confusion Matrix
        true_labels = [t for t, _ in results['true_vs_pred']]
        pred_labels = [p for _, p in results['true_vs_pred']]

        cm = confusion_matrix(true_labels, pred_labels,
                            labels=self.severity_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                   xticklabels=self.severity_classes,
                   yticklabels=self.severity_classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total Samples Analyzed: {len(results['confidence_scores'])}")
        print(f"Average Confidence: {np.mean(results['confidence_scores']):.4f}")
        print(f"Confidence Std Dev: {np.std(results['confidence_scores']):.4f}")

        print("\nAccuracy by Level:")
        for level, acc in level_acc.items():
            print(f"{level}: {acc:.4f}")

        print("\nAccuracy by Condition:")
        for cond, acc in condition_acc.items():
            print(f"{cond}: {acc:.4f}")

        # Add detailed analysis for challenging cases
        print("\nDetailed Analysis of Challenging Cases:")
        correct_predictions = sum(t == p for t, p in results['true_vs_pred'])
        total_predictions = len(results['true_vs_pred'])
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        # Analyze accuracy by severity
        severity_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
        for true, pred in results['true_vs_pred']:
            severity_acc[true]['total'] += 1
            if true == pred:
                severity_acc[true]['correct'] += 1

        print("\nAccuracy by Severity:")
        for severity, counts in severity_acc.items():
            acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            print(f"{severity}: {acc:.4f} ({counts['correct']}/{counts['total']})")

    def analyze_challenging_cases(self, num_cases: int = 5):
        """Find and analyze particularly challenging cases"""
        print("\nAnalyzing challenging cases...")
        challenging_cases = []

        # Define challenging criteria
        challenging_criteria = [
            ('L4_L5', 'Severe'),    # Known difficult level with severe cases
            ('L5_S1', 'Moderate'),  # Transition cases at difficult level
            ('L3_L4', 'Severe'),    # Higher level severe cases
            ('L4_L5', 'Moderate')   # Moderate cases at difficult level
        ]

        for criterion_level, criterion_severity in challenging_criteria:
            # Find matching cases
            for idx in range(len(self.preprocessed_data['images'])):
                if len(challenging_cases) >= num_cases:
                    break

                level = self.preprocessed_data['levels'][idx]
                severity = self.preprocessed_data['severities'][idx]

                if level == criterion_level and severity == criterion_severity:
                    challenging_cases.append(idx)

        # Analyze challenging cases
        print(f"\nAnalyzing {len(challenging_cases)} challenging cases:")
        for idx in challenging_cases:
            image = self.preprocessed_data['images'][idx]
            condition = self.preprocessed_data['conditions'][idx]
            level = self.preprocessed_data['levels'][idx]
            true_severity = self.preprocessed_data['severities'][idx]

            # Get prediction
            prediction = self.pipeline.predict(image, condition, level)
            probs = prediction['probabilities'].squeeze().cpu().numpy()
            pred_class = prediction['severity_prediction'].item()
            confidence = prediction['confidence'].item()

            # Plot results
            plt.figure(figsize=(10, 5))

            # Plot image
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Challenging Case\n{condition}\nLevel: {level}")
            plt.axis('off')

            # Plot prediction probabilities
            plt.subplot(1, 2, 2)
            colors = ['green', 'yellow', 'red']
            bars = plt.bar(self.severity_classes, probs, color=colors, alpha=0.6)
            plt.ylim(0, 1)
            plt.title(f"Predictions\nTrue: {true_severity}\n" +
                     f"Predicted: {self.severity_classes[pred_class]}\n" +
                     f"Confidence: {confidence:.2f}")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

            # Print analysis
            print(f"\nChallenging Case Analysis:")
            print(f"Condition: {condition}")
            print(f"Level: {level}")
            print(f"True Severity: {true_severity}")
            print(f"Predicted: {self.severity_classes[pred_class]}")
            print(f"Confidence: {confidence:.4f}")
            print("\nProbability Distribution:")
            for cls, prob in zip(self.severity_classes, probs):
                print(f"{cls}: {prob:.4f}")
            print("-" * 50)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and create pipeline
    model = LumbarClassifier().to(device)
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()

    pipeline = ModifiedOptimizedPredictionPipeline(model, device)

    # Create analyzer
    analyzer = AdvancedAnalysis(
        pipeline=pipeline,
        preprocessed_path='/kaggle/working/val_processed.npy'
    )

    # Run statistical analysis
    print("\nRunning Statistical Analysis...")
    stats_results = analyzer.analyze_prediction_patterns(num_samples=500)

    # Analyze challenging cases
    print("\nAnalyzing Challenging Cases...")
    analyzer.analyze_challenging_cases(num_cases=5)

if __name__ == "__main__":
    main()
