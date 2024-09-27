from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    steer_matrix_left = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]



    # The output of this is used to determine the angular rate of steering
    # where -ve values means right turn and positive values means left turn
    # Hence set the left half as negative and right half as positive to cause vehicle to move away from left lane markings
    # 0.3 is selected as it is around the value selected in the Braintenberg control exercise
    # Setting higher values close to 1 resulted in instability of the duckiebot.

    steer_matrix_left[:,:int(im_width/2)] = -0.3
    steer_matrix_left[:,int(im_width/2):] = 0.3
 
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right = np.zeros(shape=shape, dtype="float32")
    im_height = shape[0]
    im_width = shape[1]



    # The output of this is used to determine the angular rate of steering
    # where -ve values means right turn and positive values means left turn
    # Hence set the left half as negative and right half as positive to cause vehicle to move away from right lane markings

    # Set the right matrix weight as 0.8 times that of the left matrix.
    # This value was found based on experiment of values ranging from 1 to 0.4 in steps of 0.1
    steer_matrix_right[:,:int(im_width/2)] = -0.24
    steer_matrix_right[:,int(im_width/2):] = 0.24

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # Perform all the below steps from visual_servoing_activity.ipynb

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set the standard deviation that removes noise while not eliminating too much valid content.
    sigma = 4

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # Add Sobel Filter to the blurred image using Gaussian filter to detect lane markings
    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    # Filter out the gradient which is lower than a specific threshold obtained from the histogram of the sample images
    # Edges whos gradient magnitude is below this threshold will be filtered out.
    threshold = 50

    # Create an image mask based on the Gradient Magnitude
    mask_mag = (Gmag > threshold)

    # Defie upper and lower hsv limit to identify yellow and white colors within the image
    # which will be used for lane marking identification
    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([179, 80, 255])
    yellow_lower_hsv = np.array([0, 50, 130])
    yellow_upper_hsv = np.array([60, 255, 255])

    # Create white and yellow masks based on the upper and lower limits defined above
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # Let's create masks for the left- and right-halves of the image
    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    # In the left-half image, we are interested in the right-half of the dashed yellow line, which corresponds to negative x- and y-derivatives
    # In the right-half image, we are interested in the left-half of the solid white line, which correspons to a positive x-derivative and a negative y-derivative
    # Generate a mask that identifies pixels based on the sign of their x-derivative
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
