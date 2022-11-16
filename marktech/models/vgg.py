# Global import
import gc
import os
import re
from pathlib import Path
from typing import Tuple
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input

# Local import


def create_vgg_model(
        input_dim: Tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """Create VGG16 network and keep output of the second fully connected layer.

    """
    # Load vgg model trained on imagenet dataset
    full_vgg = VGG16(include_top=True, weights='imagenet', input_shape=input_dim)

    # Keep last FC layer (4096 entries vector)
    X_input = full_vgg.input
    X_output = full_vgg.layers[-2].output

    # Create new model and make its layers non-trainable
    vgg = tf.keras.Model(inputs=X_input, outputs=X_output, name='model_seq_lld')
    for layer in vgg.layers:
        layer.trainable = False

    return vgg


def transform_image(
        pth_img: Path, model: tf.keras.Model, w: int = 224, h: int = 224, n_frame: int = 10,
        is_dir: bool = True, regex: str = None
) -> np.ndarray:
    """ Transform image data with a given Keras Model.

    """
    # Load image
    if not is_dir:
        img_loader = ImageLoader(w, h)
    else:
        img_loader = ImageSeqLoader(n_frame, w, h, regex=regex)

    # transform with model
    ax_input = img_loader(pth_img)
    tr_transform = model(ax_input)

    return tr_transform.numpy()


class ImageLoader:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, pth_img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.read_file(pth_img.as_posix())
        img = tf.io.decode_jpeg(img, channels=3)

        # preprocess image for vgg16
        return preprocess_input(tf.image.resize(img, (self.h, self.w))).numpy().astype(np.uint8)[np.newaxis, :]


class ImageSeqLoader:
    regex = r"([0-9]{1,3}_example\.png)"

    def __init__(self, n_frame, w, h, regex: re.Pattern):
        self.n_frame = n_frame
        self.w, self.h = w, h
        self.regex = regex

    def __call__(self, pth_img) -> np.ndarray:
        l_files = [f.split('_') for f in os.listdir(pth_img) if re.findall(self.regex, str(f))]
        l_imgs = []
        for i, f in sorted(l_files, key=lambda t: int(t[0])):
            if int(i) >= self.n_frame:
                break

            l_imgs.append(self.encode_img(pth_img / f"{i}_{f}"))

        # Pad if necessary so that all sequence have same length
        if int(i) < self.n_frame - 1:
            out = np.concatenate([
                np.stack(l_imgs),
                np.zeros((self.n_frame - int(i) - 1, self.h, self.w, 3), dtype=np.uint8)
            ])
        else:
            out = np.stack(l_imgs)

        # Let gc know l_imgs can be deleted
        del l_imgs
        gc.collect()

        return out

    def encode_img(self, pth_img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.read_file(pth_img.as_posix())
        img = tf.io.decode_jpeg(img, channels=3)

        # preprocess image for vgg16
        return preprocess_input(tf.image.resize(img, (self.h, self.w))).numpy().astype(np.uint8)
