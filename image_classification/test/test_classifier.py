import unittest
from cv2 import imread, IMREAD_COLOR
from image_classification.classifier import ImageClassifier


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = ImageClassifier.init_from_config_path('specific/config/classifier_config.yaml')
        self.fig_path = 'image_classification/test/healthy.jpg'
        self.expected_classification = {'healthy': 1.0, 'neutral': 0.0, 'unhealthy': 0.0}


    def test_initialize_from_config(self):
        self.assertIsNotNone(self.classifier)


    def test_classify_from_array(self):
        img = imread(self.fig_path, IMREAD_COLOR)
        classification = self.classifier.classify_from_array(img)
        self.assertDictEqual(classification, self.expected_classification)


    def test_classify_from_file(self):
        with open(self.fig_path, "rb") as f:
            classification = self.classifier.classify_from_file(f)
            self.assertDictEqual(classification, self.expected_classification)


    def test_classify_from_path(self):
        classification = self.classifier.classify_from_path(self.fig_path)
        self.assertDictEqual(classification, self.expected_classification)

if __name__ == '__main__':
    unittest.main()