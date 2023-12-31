from collections import namedtuple
from pathlib import Path
from random import shuffle
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from image_classification.preprocessing_utilities import (
    read_image_from_path,
    resize_image,
)
import collections
collections.Iterable = collections.abc.Iterable
SampleFromPath = namedtuple("Sample", ["path", "target_vector"])
import imgaug.augmenters as iaa


def chunks(seq: list, size: int) -> list:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_seq() -> iaa.Sequential:
    random_add_augmentation = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            random_add_augmentation(iaa.Affine(scale={"x": (0.8, 1.2)})),
            random_add_augmentation(iaa.Fliplr(p=0.5)),
            random_add_augmentation(iaa.Affine(scale={"y": (0.8, 1.2)})),
            random_add_augmentation(iaa.Affine(translate_percent={"x": (-0.2, 0.2)})),
            random_add_augmentation(iaa.Affine(translate_percent={"y": (-0.2, 0.2)})),
            random_add_augmentation(iaa.Affine(rotate=(-20, 20))),
            random_add_augmentation(iaa.Affine(shear=(-20, 20))),
            random_add_augmentation(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            random_add_augmentation(iaa.GaussianBlur(sigma=(0, 3.0))),
        ],
        random_order=True,
    )
    return seq


def batch_generator(
    list_samples,
    batch_size: int = 32,
    pre_processing_function=None,
    resize_size: tuple = (128, 128),
    augment: bool = False,
):
    seq = get_seq()
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        for batch_samples in chunks(list_samples, size=batch_size):
            images = [read_image_from_path(sample.path) for sample in batch_samples]

            if augment:
                images = seq.augment_images(images=images)

            images = [resize_image(x, h = resize_size[0], w = resize_size[1]) for x in images]

            images = [pre_processing_function(a) for a in images]
            targets = [sample.target_vector for sample in batch_samples]
            X = np.array(images)
            Y = np.array(targets)

            yield X, Y


def dataframe_to_list_samples(
        df,
        binary_targets: str,
        base_path: str,
        image_name_col: str):
    
    paths = df[image_name_col].apply(lambda x: str(Path(base_path) / x)).tolist()
    targets = df[binary_targets].values.tolist()

    samples = [
        SampleFromPath(path=path, target_vector=target_vector)
        for path, target_vector in zip(paths, targets)
    ]

    return samples
