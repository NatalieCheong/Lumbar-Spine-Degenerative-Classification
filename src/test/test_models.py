import unittest
import torch
from src.models.classification_model import LumbarClassifier
from src.models.regression_model import LumbarRegressor

class TestModels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.image_size = 224

        # Create sample inputs
        self.sample_image = torch.randn(self.batch_size, 1, self.image_size, self.image_size)
        self.sample_condition = torch.zeros(self.batch_size, 5)  # 5 conditions
        self.sample_level = torch.zeros(self.batch_size, 5)      # 5 levels

    def test_classification_model(self):
        """Test classification model"""
        model = LumbarClassifier().to(self.device)
        model.eval()

        # Move inputs to device
        images = self.sample_image.to(self.device)
        conditions = self.sample_condition.to(self.device)
        levels = self.sample_level.to(self.device)

        with torch.no_grad():
            outputs = model(images, conditions, levels)

        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, 3))  # 3 classes

        # Check output range
        probs = torch.softmax(outputs, dim=1)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_regression_model(self):
        """Test regression model"""
        model = LumbarRegressor().to(self.device)
        model.eval()

        # Move inputs to device
        images = self.sample_image.to(self.device)
        conditions = self.sample_condition.to(self.device)
        levels = self.sample_level.to(self.device)

        with torch.no_grad():
            outputs = model(images, conditions, levels)

        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, 1))

        # Check output range (should be between 0 and 2)
        self.assertTrue(torch.all(outputs >= 0))
        self.assertTrue(torch.all(outputs <= 2))
