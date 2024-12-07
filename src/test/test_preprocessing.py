import unittest
import numpy as np
import torch
from src.preprocessing.preprocessing_pipeline import preprocess_image, create_stratified_folds

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.sample_image = np.random.rand(512, 512)
        self.sample_coords = (256, 256)

    def test_preprocess_image(self):
        """Test image preprocessing function"""
        x, y = self.sample_coords
        processed = preprocess_image(self.sample_image, x, y)

        # Check output shape
        self.assertEqual(processed.shape, (224, 224, 1))

        # Check normalization
        self.assertTrue(processed.min() >= 0)
        self.assertTrue(processed.max() <= 255)

    def test_create_stratified_folds(self):
        """Test fold creation"""
        # Create sample DataFrame
        sample_df = pd.DataFrame({
            'study_id': range(100),
            'spinal_canal_stenosis_l4_l5': ['Normal/Mild'] * 70 + ['Moderate'] * 20 + ['Severe'] * 10
        })

        folds = create_stratified_folds(sample_df, n_splits=5)

        # Check number of folds
        self.assertEqual(len(folds), 5)

        # Check fold sizes
        for fold in folds:
            self.assertGreater(len(fold['train']), 0)
            self.assertGreater(len(fold['val']), 0)
