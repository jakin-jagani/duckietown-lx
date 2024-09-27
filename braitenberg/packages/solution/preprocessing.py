import cv2
import numpy as np

# Obtain the lower and upper hsv from braitenberg02 exercise
lower_hsv = np.array([0, 0, 70])
upper_hsv = np.array([60, 255, 255])


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Returns a 2D array"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask
