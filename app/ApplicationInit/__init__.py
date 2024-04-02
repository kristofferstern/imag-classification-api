from fastapi import FastAPI, File, UploadFile
from image_classification.classifier import ImageClassifier

class ClassificationApp:

    def __init__(self, config_path: str = "config.yaml"):
        self.app = FastAPI()
        self.classifier = ImageClassifier.init_from_config_path(config_path)
