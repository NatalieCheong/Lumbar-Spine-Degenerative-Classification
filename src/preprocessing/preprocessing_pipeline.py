import pandas as pd
import numpy as np
import cv2
import pydicom
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

def save_processed_data(samples: List[Dict], filename: str):
    """Save processed samples to a numpy file"""
    # Convert samples to numpy arrays for efficient storage
    processed_data = {
        'images': [],
        'conditions': [],
        'levels': [],
        'severities': [],
        'study_ids': []
    }

    # Only save samples with valid severity
    valid_samples = [
        sample for sample in samples
        if isinstance(sample.get('severity'), str) and not pd.isna(sample.get('severity'))
    ]

    for sample in valid_samples:
        processed_data['images'].append(sample['image'])
        processed_data['conditions'].append(sample['condition'])
        processed_data['levels'].append(sample['level'])
        processed_data['severities'].append(sample['severity'])
        processed_data['study_ids'].append(sample['study_id'])

    # Convert lists to numpy arrays
    for key in processed_data:
        processed_data[key] = np.array(processed_data[key])

    # Save to file
    np.save(filename, processed_data)
    print(f"Saved {len(valid_samples)} valid samples to {filename}")

def create_stratified_folds(train_df: pd.DataFrame, n_splits: int = 5) -> List[Dict]:
    """Create stratified folds ensuring similar distribution of severe cases"""
    # Create severity indicator for stratification
    severity_cols = [col for col in train_df.columns
                    if any(c in col for c in ['canal', 'foraminal', 'subarticular'])]

    # Create binary indicator for having any severe case
    train_df['has_severe'] = (train_df[severity_cols] == 'Severe').any(axis=1)

    # Initialize fold splitter
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create folds
    folds = []
    for train_idx, val_idx in skf.split(
        train_df,
        train_df['has_severe'],
        groups=train_df['study_id']
    ):
        fold = {
            'train': train_df.iloc[train_idx]['study_id'].tolist(),
            'val': train_df.iloc[val_idx]['study_id'].tolist()
        }
        folds.append(fold)

    return folds

def process_fold_data(
    study_ids: List[int],
    base_path: str,
    coords_df: pd.DataFrame,
    series_df: pd.DataFrame,
    train_df: pd.DataFrame,
    augment: bool = False
) -> List[Dict]:
    """Process data for a set of studies"""
    processed_data = []

    for study_id in study_ids:
        # Get series information for this study
        study_series = series_df[series_df['study_id'] == study_id]

        # Process each condition type with its corresponding series
        processed_data.extend(
            process_study_images(
                study_id=study_id,
                base_path=base_path,
                study_series=study_series,
                coords_df=coords_df,
                train_df=train_df,
                augment=augment
            )
        )

    return processed_data

def process_study_images(
    study_id: int,
    base_path: str,
    study_series: pd.DataFrame,
    coords_df: pd.DataFrame,
    train_df: pd.DataFrame,
    augment: bool = False
) -> List[Dict]:
    """Process images for a single study"""
    processed_samples = []
    study_path = Path(base_path) / 'train_images' / str(study_id)

    # Get study data
    study_coords = coords_df[coords_df['study_id'] == study_id]
    study_labels = train_df[train_df['study_id'] == study_id]

    # Process each type of condition with its corresponding series type
    condition_series_map = {
        'Spinal Canal Stenosis': 'Sagittal T2',
        'Neural Foraminal Narrowing': 'Sagittal T1',
        'Subarticular Stenosis': 'Axial T2'
    }

    for condition, series_type in condition_series_map.items():
        # Get relevant series
        series = study_series[
            study_series['series_description'].str.contains(series_type, na=False)
        ]

        for _, series_row in series.iterrows():
            series_id = series_row['series_id']
            series_path = study_path / str(series_id)

            # Get coordinates for this series
            series_coords = study_coords[
                (study_coords['series_id'] == series_id) &
                (study_coords['condition'].str.contains(condition, na=False))
            ]

            # Process each image in the series
            for _, coord_row in series_coords.iterrows():
                # Get severity from labels
                label_col = f"{coord_row['condition'].lower().replace(' ', '_')}_{coord_row['level'].lower().replace('/', '_')}"
                if label_col in study_labels.columns:
                    severity = study_labels[label_col].iloc[0]

                    # Process image
                    processed = process_single_image(
                        series_path=series_path,
                        coord_row=coord_row,
                        severity=severity,
                        augment=augment
                    )
                    if processed:
                        processed_samples.extend(processed)

    return processed_samples

def process_single_image(
    series_path: Path,
    coord_row: pd.Series,
    severity: str,
    augment: bool = False
) -> List[Dict]:
    """Process a single image and its annotation"""
    processed_samples = []

    # Load DICOM image
    dcm_path = series_path / f"{int(coord_row['instance_number'])}.dcm"
    if not dcm_path.exists():
        return None

    try:
        dcm = pydicom.dcmread(str(dcm_path))
        image = dcm.pixel_array

        # Preprocess image
        processed_image = preprocess_image(
            image,
            int(coord_row['x']),
            int(coord_row['y'])
        )

        # Create base sample
        sample = {
            'image': processed_image,
            'study_id': coord_row['study_id'],
            'condition': coord_row['condition'],
            'level': coord_row['level'].replace('/', '_'),
            'severity': severity,
            'coordinates': (coord_row['x'], coord_row['y'])
        }

        processed_samples.append(sample)

        # Apply augmentations if required
        if augment and severity == 'Severe':
            augmented_samples = apply_augmentations(sample)
            processed_samples.extend(augmented_samples)

        return processed_samples
    except:
        return None

def preprocess_image(
    image: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 224
) -> np.ndarray:
    """Preprocess a single image"""
    # Normalize pixel values
    normalized = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    # Extract patch around the annotation point
    half_size = patch_size // 2

    # Pad image if necessary
    padded = np.pad(
        normalized,
        ((half_size, half_size), (half_size, half_size)),
        mode='constant',
        constant_values=0
    )

    # Extract patch
    x_start = x + half_size - half_size
    x_end = x + half_size + half_size
    y_start = y + half_size - half_size
    y_end = y + half_size + half_size

    patch = padded[y_start:y_end, x_start:x_end]

    # Ensure patch size
    if patch.shape != (patch_size, patch_size):
        patch = cv2.resize(patch, (patch_size, patch_size))

    # Add channel dimension
    patch = np.expand_dims(patch, axis=-1)

    return patch

def apply_augmentations(sample: Dict) -> List[Dict]:
    """Apply augmentations to balance severe cases"""
    augmented_samples = []

    # Add rotated versions
    for angle in [90, 180, 270]:
        aug_sample = sample.copy()
        aug_sample['image'] = np.rot90(sample['image'], k=angle//90)
        augmented_samples.append(aug_sample)

    # Add flipped versions
    aug_sample = sample.copy()
    aug_sample['image'] = np.flip(sample['image'], axis=1)
    augmented_samples.append(aug_sample)

    return augmented_samples

def main():
    """Main function for preprocessing pipeline"""
    # Set paths
    base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'

    # Load data
    train_df = pd.read_csv(f"{base_path}/train.csv")
    coords_df = pd.read_csv(f"{base_path}/train_label_coordinates.csv")
    series_df = pd.read_csv(f"{base_path}/train_series_descriptions.csv")

    # Create train-validation split
    print("Creating stratified fold splits...")
    folds = create_stratified_folds(train_df)

    # Process example fold
    fold_num = 0
    train_studies = folds[fold_num]['train']
    val_studies = folds[fold_num]['val']

    print(f"\nProcessing fold {fold_num}...")
    # Process training data
    train_samples = process_fold_data(
        study_ids=train_studies,
        base_path=base_path,
        coords_df=coords_df,
        series_df=series_df,
        train_df=train_df,
        augment=True
    )

    # Process validation data
    val_samples = process_fold_data(
        study_ids=val_studies,
        base_path=base_path,
        coords_df=coords_df,
        series_df=series_df,
        train_df=train_df,
        augment=False
    )

    print(f"Processed {len(train_samples)} training samples and {len(val_samples)} validation samples")

    # Save processed data
    print("\nSaving processed data...")
    save_processed_data(train_samples, '/kaggle/working/train_processed.npy')
    save_processed_data(val_samples, '/kaggle/working/val_processed.npy')

if __name__ == "__main__":
    main()
