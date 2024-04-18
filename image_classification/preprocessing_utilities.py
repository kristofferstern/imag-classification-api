from cv2 import imread, imdecode, resize, copyMakeBorder, IMREAD_COLOR, BORDER_CONSTANT
import numpy as np


def read_image_from_path(path: str) -> np.array:
    img = imread(path, IMREAD_COLOR)
    return img


def read_image_from_file(file_object) -> np.array:
    arr = np.fromstring(file_object.read(), np.uint8)
    img_np = imdecode(arr, IMREAD_COLOR)

    return img_np


def resize_image(
        image: np.array,
        h: int = 128,
        w: int = 128) -> np.array:
    desired_size_h = h
    desired_size_w = w

    orig_size = image.shape[:2]

    ratio = min(desired_size_w / orig_size[1], desired_size_h / orig_size[0])

    new_size = tuple([int(x * ratio) for x in orig_size])

    im = resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size_w - new_size[1]
    delta_h = desired_size_h - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_image = copyMakeBorder(
        im, top, bottom, left, right, BORDER_CONSTANT, value=color
    )

    return new_image