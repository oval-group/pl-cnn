import numpy as np
from PIL import Image


def center_data(X_train, X_val, X_test,
                mode, offset=None):
    """ center images per channel or per pixel
    """

    if offset is None:
        if mode == "per channel":
            n_channels = np.shape(X_train)[1]
            offset = np.mean(X_train, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
        elif mode == "per pixel":
            offset = np.mean(X_train, 0)
        else:
            raise ValueError("Specify mode of centering "
                             "(should be 'per channel' or 'per pixel')")

    X_train -= offset
    X_val -= offset
    X_test -= offset


def normalize_data(X_train, X_val, X_test,
                   mode="per channel", scale=None):
    """ normalize images per channel, per pixel or with a fixed value
    """

    if scale is None:
        if mode == "per channel":
            n_channels = np.shape(X_train)[1]
            scale = np.std(X_train, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
        elif mode == "per pixel":
            scale = np.std(X_train, 0)
        elif mode == "fixed value":
            scale = 255.
        else:
            raise ValueError("Specify mode of scaling (should be "
                             "'per channel', 'per pixel' or 'fixed value')")

    X_train /= scale
    X_val /= scale
    X_test /= scale


def filename_to_array_imagenet(filename, mean, crop):

    # decode image (generic enough to deal with PNG)
    img = Image.open(filename)

    # deal with CMYK mode
    if img.mode == 'CMYK':
        img = img.convert('RGB')

    # center crop of 224x224 at training time (no processing at test time)
    if crop:
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img = img.crop((half_the_width - 112,
                        half_the_height - 112,
                        half_the_width + 112,
                        half_the_height + 112))

    # store as numpy array
    array = np.asarray(img, dtype=np.float32)

    # deal with grayscale images
    if array.ndim < 3:
        array = np.stack([array] * 3, axis=2)

    # move RGB axis and switch to BGR
    array = np.moveaxis(array, 2, 0)[::-1, :, :]

    # subtract mean channel
    array = array - mean

    return array
