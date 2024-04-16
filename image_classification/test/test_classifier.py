import unittest

from image_classification.classifier import ImageClassifier

class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = ImageClassifier.init_from_config_path('specific/config/classifier_config.yaml')

    def test_initialize_from_file(self):
        self.assertIsNotNone(self.classifier)

    def test_classify_from_file(self):
        with open("specific/data/fig.jpg", "rb") as f:
            classification = self.classifier.classify_from_file(f)

