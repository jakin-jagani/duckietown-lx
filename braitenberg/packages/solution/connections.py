from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]
    # res[int(im_height/2):, int(im_width/3):int(im_width/2)] = 1
    # res[int(im_height/2):, int(im_width/2):int(2*im_width/3)] = -1
    res[:,:int(im_width/2)] = 1
    res[:,int(im_width/2):] = -1
    # ---
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]
    # res[int(im_height/2):, int(im_width/3):int(im_width/2)] = -1
    # res[int(im_height/2):, int(im_width/2):int(2*im_width/3)]= 1
    res[:,:int(im_width/2)] = -1
    res[:,int(im_width/2):] = 1
    # ---
    return res
