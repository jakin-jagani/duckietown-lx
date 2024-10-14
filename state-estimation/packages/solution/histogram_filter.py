# start by importing some things we will need
import numpy as np
from math import floor, sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief

# Let's define function to calculate delta phi (change in oritentation of the robot)
def delta_phi_calc(delta_ticks: int, resolution: int):
    """
    Args:
        delta_ticks: Delta tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
    """

    # Revolution per tick in radians
    alpha = 2 * np.pi / resolution
    dphi = alpha * delta_ticks
    # ---
    return dphi

# Let's define a function to determine the pose delta of the robot
def delta_pose_estimation(
    R: float,
    baseline: float,
    delta_phi_left: float,
    delta_phi_right: float,
):

    """
    Calculate the Duckiebot pose delta with respect to previous position using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        delta_distance:     estimated change in robot distance
        delta_theta:          estimated change in robot orientation angle
    """

    # Determine the distnace traveled by each wheel
    d_l = R * delta_phi_left
    d_r = R * delta_phi_right

    # Distance traveled by the center of the robot
    delta_distance = (d_l + d_r)/2

    # Determine the change in robot orientation angle
    delta_theta = (d_r - d_l)/baseline    

    # ---
    return delta_distance, delta_theta

# Now let's define the predict function

def histogram_predict(belief, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief

    # TODO calculate v and w from ticks using kinematics.
    #  You will need  some parameters in the `robot_spec` defined above
    # Determine the delta phi for left and right wheels based on encoder ticks
    encoder_resolution = robot_spec["encoder_resolution"]
    delta_left_encoder_phi = delta_phi_calc(delta_ticks=left_encoder_ticks, resolution=encoder_resolution)
    delta_right_encoder_phi = delta_phi_calc(delta_ticks=right_encoder_ticks, resolution=encoder_resolution)

    # You may find the following code useful to find the current best heading estimate:
    maxids = np.unravel_index(belief_in.argmax(), belief_in.shape)
    phi_max = grid_spec['phi_min'] + (maxids[1] + 0.5) * grid_spec['delta_phi']

    # Determine delta distance and delta robot orientation
    wheel_radius = robot_spec["wheel_radius"]
    wheel_baseline = robot_spec["wheel_baseline"]
    delta_d, delta_phi = delta_pose_estimation(R=wheel_radius, baseline=wheel_baseline, delta_phi_left=delta_left_encoder_phi, delta_phi_right=delta_right_encoder_phi)

    # propagate each centroid
    d_t = grid_spec["d"] + delta_d
    phi_t = grid_spec["phi"] + delta_phi

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            # If belief[i,j] there was no mass to move in the first place
            if belief[i, j] > 0:
                # Now check that the centroid of the cell wasn't propagated out of the allowable range
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue

                # Find the new grid cell for the propagated cell
                i_new = int((d_t[i, j] - grid_spec['d_min']) / (grid_spec['d_max'] - grid_spec['d_min']) * belief.shape[0])
                j_new = int((phi_t[i, j] - grid_spec['phi_min']) / (grid_spec['phi_max'] - grid_spec['phi_min']) * belief.shape[1])

                # Make sure i_new and j_new are within the bounds of the belief grid
                i_new = np.clip(i_new, 0, belief.shape[0] - 1)
                j_new = np.clip(j_new, 0, belief.shape[1] - 1)

                p_belief[i_new, j_new] += belief[i, j]

    # Finally we are going to add some "noise" according to the process model noise
    # This is implemented as a Gaussian blur
    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)
    return belief


# We will start by doing a little bit of processing on the segments to remove anything that is
# behing the robot (why would it be behind?) or a color not equal to yellow or white


def prepare_segments(segments, grid_spec):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        point_range = getSegmentDistance(segment)
        if grid_spec["range_est"] > point_range > 0:
            filtered_segments.append(segment)
    return filtered_segments


def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])

    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec["linewidth_white"]
        else:  # left edge of white lane
            d_i -= road_spec["linewidth_white"]
            d_i = road_spec["lanewidth"] * 2 + road_spec["linewidth_yellow"] - d_i
            phi_i = -phi_i
        d_i -= road_spec["lanewidth"]/2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec["linewidth_yellow"]
            d_i = road_spec["lanewidth"]/2 - d_i
            phi_i = -phi_i
        else:  # right edge of yellow lane
            d_i += road_spec["linewidth_yellow"]
            d_i -= road_spec["lanewidth"]/2

    return d_i, phi_i


def generate_measurement_likelihood(segments, road_spec, grid_spec):
    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        # So now we have d_i and phi_i which correspond to the estimate of the distance
        # from the center and the angle relative to the center generated by the single
        # segment under consideration
        # Find the cell index that corresponds to the measurement d_i, phi_i
        d_shape_size = grid_spec['d'].shape[0]
        phi_shape_size = grid_spec['phi'].shape[0]
        # Find the cell index that corresponds to the measurement d_i and phi_i
        i = int((d_i - grid_spec['d_min']) / (grid_spec['d_max'] - grid_spec['d_min']) * d_shape_size)
        j = int((phi_i - grid_spec['phi_min']) / (grid_spec['phi_max'] - grid_spec['phi_min']) * phi_shape_size-1)

        # Make sure i and j are within the bounds of the d and phi grid
        i = np.clip(i, 0, d_shape_size - 1)
        j = np.clip(j, 0, phi_shape_size - 1)

        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments, grid_spec)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        # Combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid probability distribution
        posterio_belief = belief * measurement_likelihood
        if np.sum(posterio_belief) != 0:
            belief = posterio_belief/np.sum(posterio_belief)
    return measurement_likelihood, belief

def getSegmentDistance(segment):
    x_c = (segment.points[0].x + segment.points[1].x) / 2
    y_c = (segment.points[0].y + segment.points[1].y) / 2
    return sqrt(x_c**2 + y_c**2)