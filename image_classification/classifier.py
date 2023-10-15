import argparse
import sys
import typing
import numpy as np
import yaml
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from image_classification.preprocessing_utilities import (
    read_image_from_path,
    resize_image,
    read_image_from_file,
)
from image_classification.utils import download_model


class ImageClassifier:
    def __init__(
        self, model_path, resize_size, targets, pre_processing_function=preprocess_input
    ):
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path)
        self.resize_size = resize_size
        self.targets = targets

    @classmethod
    def init_from_config_path(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        classifier = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"],
        )
        return classifier

    @classmethod
    def init_from_config_url(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["model_url"], config["model_path"], config["model_sha256"]
        )

        return cls.init_from_config_path(config_path)

    def classify_from_array(self, arr: np.array) -> typing.Dict[str, float]:
        arr = resize_image(arr, h=self.resize_size[0], w=self.resize_size[1])
        arr = self.pre_processing_function(arr)
        pred = self.model.predict(arr[np.newaxis, ...]).ravel().tolist()
        pred = [round(x, 3) for x in pred]
        return {k: v for k, v in zip(self.targets, pred)}

    def classify_from_path(self, path: str) -> typing.Dict[str, float]:
        arr = read_image_from_path(path)
        return self.classify_from_array(arr)

    def classify_from_file(self, file_object) -> typing.Dict[str, float]:
        arr = read_image_from_file(file_object)
        return self.classify_from_array(arr)


if __name__ == "__main__":
    """
    python classifier.py --classifier_config "specific/config/classifier_config.yaml"

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_config_path",
        help="classifier_config_path",
        default="specific/config/classifier_config.yaml",
    )
    args = parser.parse_args()

    classifier_config_path = args.classifier_config_path

    classifier = ImageClassifier.init_from_config_path(classifier_config_path)

    # classification = classifier.classify_from_path(
    #     "specific/data/fig.jpg"
    # )
    #
    # classification = classifier.classify_from_path(
    #     "specific/data/fig.jpg"
    # )
    #
    # with open("specific/data/fig.jpg", "rb") as f:
    #     classification = classifier.classify_from_file(f)
    #
    # print(classification)