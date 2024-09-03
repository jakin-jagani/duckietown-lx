from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """

    # Revolution per tick in radians
    alpha = 2 * np.pi / resolution
    delta_ticks = ticks - prev_ticks
    dphi = alpha * delta_ticks
    # ---
    return dphi, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:                  estimated x coordinate
        y_curr:                  estimated y coordinate
        theta_curr:              estimated heading
    """

    # Determine the distnace traveled by each wheel
    d_l = R * delta_phi_left
    d_r = R * delta_phi_right

    # Distance traveled by the center of the robot
    d_a = (d_l + d_r)/2

    # Determine the change in robot orientation angle
    delta_theta = (d_r - d_l)/baseline
    theta_curr = theta_prev + delta_theta

    x_curr = x_prev + d_a * np.cos(theta_curr)
    y_curr = y_prev + d_a * np.sin(theta_curr)
    

    # ---
    return x_curr, y_curr, theta_curr
