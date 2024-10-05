from typing import Tuple

IMAGE_SIZE = 416
def DT_TOKEN() -> str:
    # Set your duckietown token
    dt_token = "dt1-3nT7FDbT7NLPrXykNJW6pwZZwVDgWk3qzvwSgb4kkAgPZQ8-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfchj3tBXHViZNvCCcqB7e5mBceHRsjpnL3"
    return dt_token


def MODEL_NAME() -> str:
    # Set your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5n"
    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:
    # TODO: change this number to drop more frames
    # (must be a positive integer)
    return 4


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    # only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return pred_class == 0


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    # Right now, this returns True for every object's confidence
    # TODO: Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return True


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """
    # TODO: Like in the other cases, return False if the bbox should not be considered.
    x_left_norm = bbox[0]/IMAGE_SIZE
    y_left_norm = bbox[1]/IMAGE_SIZE
    x_right_norm = bbox[2]/IMAGE_SIZE
    y_right_norm = bbox[3]/IMAGE_SIZE

    bbox_norm = (x_left_norm, y_left_norm, x_right_norm, y_right_norm)

    # Uncomment the below for debugging
    # print(f"bbox = {bbox}")
    # print(f"bbox_norm = {bbox_norm}")

    if (x_left_norm >= 0.33 and 
        y_left_norm >= 0.33 and
        x_right_norm <= 0.85 and
        y_right_norm <= 1
    ):
        print(f"bbox condition = True")
        return True
    else:
        print(f"bbox condition = False")
        return False
