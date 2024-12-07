class ModifiedOptimizedPredictionPipeline:
    def __init__(self, classification_model, device):
        self.classification_model = classification_model
        self.device = device
        self.classification_model.eval()

        # Define maps
        self.condition_map = {
            'Spinal Canal Stenosis': 0,
            'Left Neural Foraminal Narrowing': 1,
            'Right Neural Foraminal Narrowing': 2,
            'Left Subarticular Stenosis': 3,
            'Right Subarticular Stenosis': 4
        }

        self.level_map = {
            'L1_L2': 0, 'L2_L3': 1, 'L3_L4': 2, 'L4_L5': 3, 'L5_S1': 4
        }

    def prepare_input_tensors(self, image, condition, level):
        """Prepare input tensors with correct dimensions"""
        # Image tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if len(image.shape) == 3:  # (H, W, C)
            image = image.permute(2, 0, 1)  # (C, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        # Condition tensor
        condition_tensor = torch.zeros(1, len(self.condition_map))
        condition_tensor[0, self.condition_map[condition]] = 1

        # Level tensor
        level_tensor = torch.zeros(1, len(self.level_map))
        level_tensor[0, self.level_map[level]] = 1

        # Move to device
        image = image.to(self.device)
        condition_tensor = condition_tensor.to(self.device)
        level_tensor = level_tensor.to(self.device)

        return image, condition_tensor, level_tensor

    def predict(self, image, condition, level):
        """Make prediction with proper tensor handling"""
        with torch.no_grad():
            # Prepare inputs
            image_tensor, condition_tensor, level_tensor = self.prepare_input_tensors(
                image, condition, level
            )

            # Get predictions
            outputs = self.classification_model(image_tensor, condition_tensor, level_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get prediction class and confidence
            pred_class = torch.argmax(probabilities, dim=1)[0]
            confidence = torch.max(probabilities, dim=1)[0][0]

            return {
                'probabilities': probabilities,
                'severity_prediction': pred_class,
                'confidence': confidence,
            }

def predict_preprocessed_samples(pipeline, preprocessed_path: str, num_samples: int = 5):
    """Make predictions on randomly selected samples"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    preprocessed_data = np.load(preprocessed_path, allow_pickle=True).item()

    # Get random indices
    total_samples = len(preprocessed_data['images'])
    sample_indices = np.random.choice(total_samples, num_samples, replace=False)

    severity_classes = ['Normal/Mild', 'Moderate', 'Severe']
    colors = ['green', 'yellow', 'red']

    for idx in sample_indices:
        try:
            # Get sample data
            image = preprocessed_data['images'][idx]
            condition = preprocessed_data['conditions'][idx]
            level = preprocessed_data['levels'][idx]
            true_severity = preprocessed_data['severities'][idx]
            study_id = preprocessed_data['study_ids'][idx]

            print(f"\nProcessing Sample {idx} (Study ID: {study_id})")

            # Get prediction
            prediction = pipeline.predict(image, condition, level)

            # Plot results
            plt.figure(figsize=(10, 5))

            # Plot image
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Study ID: {study_id}\n{condition}\nLevel: {level}")
            plt.axis('off')

            # Plot prediction results
            plt.subplot(1, 2, 2)
            probs = prediction['probabilities'].squeeze().cpu().numpy()
            pred_class = prediction['severity_prediction'].item()
            confidence = prediction['confidence'].item()

            # Create bar plot
            bars = plt.bar(severity_classes, probs, color=colors, alpha=0.6)
            plt.ylim(0, 1)
            plt.title(f"Predictions\nTrue: {true_severity}\n" +
                     f"Predicted: {severity_classes[pred_class]}\n" +
                     f"Confidence: {confidence:.2f}")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

            # Print detailed prediction information
            print("\nDetailed Prediction Information:")
            print(f"Condition: {condition}")
            print(f"Level: {level}")
            print(f"True Severity: {true_severity}")
            print(f"Predicted Severity: {severity_classes[pred_class]}")
            print(f"Confidence: {confidence:.4f}")
            print("\nClass Probabilities:")
            for cls, prob in zip(severity_classes, probs):
                print(f"{cls}: {prob:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = LumbarClassifier().to(device)
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()

    # Create prediction pipeline
    pipeline = ModifiedOptimizedPredictionPipeline(model, device)

    # Make predictions on preprocessed samples
    predict_preprocessed_samples(
        pipeline,
        preprocessed_path='/kaggle/working/val_processed.npy',
        num_samples=10
    )

if __name__ == "__main__":
    main()
