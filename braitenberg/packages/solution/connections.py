from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Function to get the left motor matrix
    Here everything on the left half of the image (width/2 = shape[1] is set to 1
    and everything on the right half of the image (width/2 = shape[1]) is set to -1
    """
    res = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]
    res[:,:int(im_width/2)] = 1
    res[:,int(im_width/2):] = -1
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Function to get the right motor matrix
    Here everything on the left half of the image (width/2 = shape[1] is set to -1
    and everything on the right half of the image (width/2 = shape[1]) is set to 1
    """
    res = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]
    res[:,:int(im_width/2)] = -1
    res[:,int(im_width/2):] = 1
    return res
