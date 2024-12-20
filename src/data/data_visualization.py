import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy as np
import os
from pathlib import Path
import warnings
from typing import List, Dict, Tuple

def main():
    """Main function for data exploration and visualization"""
    # Set up paths
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
    train_path = f"{base_path}/train_images"

    # Load metadata
    train_df = pd.read_csv(f"{base_path}/train.csv")
    coordinates_df = pd.read_csv(f"{base_path}/train_label_coordinates.csv")
    series_desc_df = pd.read_csv(f"{base_path}/train_series_descriptions.csv")

    # Print basic dataset information
    print_dataset_stats(train_df)

    # Visualize distribution of conditions
    plot_condition_distributions(train_df)

    # Example: Load and display images for one patient
    example_patient_id = train_df['study_id'].iloc[0]
    patient_images = load_patient_images(example_patient_id, train_path, series_desc_df)
    visualize_patient_images(patient_images)

    # Show pathology locations for the example patient
    show_pathology_locations(example_patient_id, patient_images, coordinates_df, train_df)

def print_dataset_stats(train_df: pd.DataFrame) -> None:
    """Print basic statistics about the dataset"""
    print(f"Total number of cases: {len(train_df)}")
    print("\nColumns in dataset:")
    for col in train_df.columns:
        print(f"- {col}")

    # Count conditions by type
    condition_types = ['foraminal', 'subarticular', 'canal']
    print("\nCondition counts:")
    for condition in condition_types:
        cols = [col for col in train_df.columns if condition in col]
        total_cases = train_df[cols].notna().sum().sum()
        print(f"{condition}: {total_cases} annotations")

def plot_condition_distributions(train_df: pd.DataFrame) -> None:
    """Plot distribution of different conditions"""
    condition_types = ['foraminal', 'subarticular', 'canal']

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for idx, condition in enumerate(condition_types):
        # Get columns for this condition
        condition_cols = [col for col in train_df.columns if condition in col]
        condition_data = train_df[condition_cols]

        # Count value distributions
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            value_counts = condition_data.apply(pd.value_counts).fillna(0).T

        # Plot
        value_counts.plot(kind='bar', stacked=True, ax=axes[idx])
        axes[idx].set_title(f'{condition} Distribution')
        axes[idx].set_xlabel('Vertebral Level')
        axes[idx].set_ylabel('Count')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def load_patient_images(study_id: int, base_path: str, series_desc_df: pd.DataFrame) -> Dict:
    """Load all images for a single patient"""
    study_path = Path(base_path) / str(study_id)

    image_data = {}
    for series_id in os.listdir(study_path):
        if series_id.startswith('.'):
            continue

        # Get series description
        series_desc = series_desc_df[
            (series_desc_df['study_id'] == study_id) &
            (series_desc_df['series_id'] == int(series_id))
        ]['series_description'].iloc[0]

        # Load all DICOM images in this series
        series_path = study_path / series_id
        image_data[series_id] = {
            'description': series_desc,
            'images': []
        }

        for dcm_file in sorted(os.listdir(series_path)):
            if dcm_file.endswith('.dcm'):
                dcm_path = series_path / dcm_file
                dcm = pydicom.dcmread(str(dcm_path))
                image_data[series_id]['images'].append({
                    'instance_number': dcm_file.replace('.dcm', ''),
                    'dicom': dcm
                })

    return image_data

def visualize_patient_images(image_data: Dict) -> None:
    """Display all images for a patient organized by series"""
    for series_id, series_data in image_data.items():
        images = [img['dicom'].pixel_array for img in series_data['images']]

        # Calculate grid dimensions
        n_images = len(images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        # Create subplot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        fig.suptitle(f"Series: {series_data['description']}")

        if n_rows == 1:
            axes = [axes]

        # Plot each image
        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].imshow(img, cmap='gray')
            axes[row][col].axis('off')
            axes[row][col].set_title(f"Image {idx+1}")

        # Turn off empty subplots
        for idx in range(len(images), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')

        plt.tight_layout()
        plt.show()

def show_pathology_locations(
    study_id: int,
    image_data: Dict,
    coordinates_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> None:
    """Display images with annotated pathology locations"""
    # Get coordinates for this study
    study_coords = coordinates_df[coordinates_df['study_id'] == study_id]

    # Get conditions for this study
    study_conditions = train_df[train_df['study_id'] == study_id]

    for _, coord in study_coords.iterrows():
        series_id = str(coord['series_id'])
        instance_num = str(coord['instance_number'])

        # Find the corresponding image
        for img in image_data[series_id]['images']:
            if img['instance_number'] == instance_num:
                # Create a copy of the image for annotation
                pixel_array = img['dicom'].pixel_array
                normalized_img = cv2.normalize(
                    pixel_array,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U
                )

                # Draw circle at pathology location
                annotated_img = cv2.circle(
                    normalized_img.copy(),
                    (int(coord['x']), int(coord['y'])),
                    radius=10,
                    color=(255, 0, 0),
                    thickness=2
                )

                # Get severity for this condition/level
                condition_col = f"{coord['condition'].lower().replace(' ', '_')}_{coord['level'].lower().replace('/', '_')}"
                severity = study_conditions[condition_col].iloc[0] if condition_col in study_conditions else "Unknown"

                # Display image
                plt.figure(figsize=(8, 8))
                plt.imshow(annotated_img, cmap='gray')
                plt.title(f"Level: {coord['level']}\nCondition: {coord['condition']}\nSeverity: {severity}")
                plt.axis('off')
                plt.show()

if __name__ == "__main__":
    main()
