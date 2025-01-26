import itertools
from typing import List, Tuple
import math

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
)

__all__ = ["CollisionChecker"]


class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        print(f"Robot Body = {self.params.body}")
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)


def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    """
    Function which returns of the robot has collided with any object within the environment
    Args:
        environment: List of PlacedPrimitive obstacles in the environment
        robot_body: List of PlacedPrimitive parts of the robot
        robot_pose: Robot pose in global frame of reference

    Returns: True if collided else False
    """

    # Roto-translate the robot_body by robot_pose to get robot_body in global frame
    roto_translated_robot: List[PlacedPrimitive] = get_robot_body_in_global_frame(robot_body, robot_pose)

    # Check if any part of the robot body collides with the environment
    for robot, envObject in itertools.product(roto_translated_robot, environment):
        if check_collision_shape(robot, envObject):
            return True

    return False

def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    """
    Function to check if the 2 PlacedPrimitive shapes are colliding
    :return bool: True if colliding else False
    """

    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        a_center = (a.pose.x, a.pose.y)
        b_center = (b.pose.x, b.pose.y)
        a_radius = a.primitive.radius
        b_radius = b.primitive.radius
        return math.dist(a_center, b_center) <= (a_radius + b_radius) 
        
    elif isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        a_rect_corners = get_rectangle_global_coordinates(a)
        # Divide circle circumference into multiple points to see if any of its
        # circumference points lie inside rectangle
        theta_step_deg = 20
        collision = False
        for theta in range(0, 360, theta_step_deg):
            theta_rad = math.radians(theta)
            circle_x = b.pose.x + b.primitive.radius * math.cos(theta_rad)
            circle_y = b.pose.y + b.primitive.radius * math.sin(theta_rad)
            if is_point_in_bounding_box((circle_x, circle_y), a_rect_corners):
                collision = True
                break
        return collision
    elif isinstance(a.primitive, Circle) and isinstance(b.primitive, Rectangle):
        b_rect_corners = get_rectangle_global_coordinates(b)
        # Divide circle circumference into multiple points to see if any of its
        # circumference points lie inside rectangle
        theta_step_deg = 20
        collision = False
        for theta in range(0, 360, theta_step_deg):
            theta_rad = math.radians(theta)
            circle_x = a.pose.x + a.primitive.radius * math.cos(theta_rad)
            circle_y = a.pose.y + a.primitive.radius * math.sin(theta_rad)
            if is_point_in_bounding_box((circle_x, circle_y), b_rect_corners):
                collision = True
                break
        return collision
    elif isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        a_rect_corners = get_rectangle_global_coordinates(a)
        b_rect_corners = get_rectangle_global_coordinates(b)
        collision = False
        for a_rect_corner in a_rect_corners:
            if is_point_in_bounding_box(a_rect_corner, b_rect_corners):
                collision = True
                break
        return collision

def get_robot_body_in_global_frame(robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose) -> List[PlacedPrimitive]:
    """
    Convert PlacedPrimitives in robot_body from local frame to global frame pose provided by robot_pose
    Args:
        robot_body: List[PlacedPrimitive] List of robot body parts defined in PlacedPrimitive type
        robot_pose: [FriendlyPose] Robot Pose in global frame of reference

    Returns:
        List of robot body parts in global frame of reference
    """

    # Get robot pose
    robot_x = robot_pose.x
    robot_y = robot_pose.y
    robot_theta_deg = robot_pose.theta_deg

    # Initialize roto_translated robot body parts in global frame
    robot_body_global_frame = []
    for body_part in robot_body:
        relative_x = robot_x - body_part.pose.x
        relative_y = robot_y - body_part.pose.y
        global_x = body_part.pose.x + relative_x
        global_y = body_part.pose.y + relative_y
        global_theta = body_part.pose.theta_deg + robot_theta_deg
        body_part_global_pose = FriendlyPose(global_x, global_y, global_theta)
        body_part_global_frame = PlacedPrimitive(pose= body_part_global_pose,primitive=body_part.primitive)
        robot_body_global_frame.append(body_part_global_frame)

    return robot_body_global_frame

def get_rectangle_global_coordinates(rect: PlacedPrimitive) -> Tuple:
    """
    Function to get the coordinates of a given Rectangle PlacedPrimitive in
    global frame of reference using the rectangle pose, length and breadth values
    Args:
        rect: [PlacedPrimitve] Rectangle shape

    Returns:
        Tuple of (x,y) coorinates of each of the rectangle corners
    """

    # Get the distance from center (pose) to the sides of the rectangle
    x = rect.primitive.xmax # length/2
    y = rect.primitive.ymax # breadth/2

    # Define the center of the rectangle
    x_center = rect.pose.x
    y_center = rect.pose.y
    theta_deg = rect.pose.theta_deg
    theta_rad = math.radians(theta_deg)

    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # Determine the coordinates of the 4 corners of the rectangle
    # Coordinates are defined starting from top right, top left, bottom left, bottom right
    x1, y1 = (x_center + x * cos_theta - y * sin_theta, y_center + x * sin_theta + y * cos_theta)
    x2, y2 = (x_center - x * cos_theta - y * sin_theta, y_center - x * sin_theta + y * cos_theta)
    x3, y3 = (x_center - x * cos_theta + y * sin_theta, y_center - x * sin_theta - y * cos_theta)
    x4, y4 = (x_center + x * cos_theta + y * sin_theta, y_center + x * sin_theta - y * cos_theta)

    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

def is_point_in_bounding_box(p: Tuple, box: Tuple) -> bool:
    """
    Function to check if a point is within a bounding box with 4 corners
    :return bool: True if point lies within the quadrilateral else False
    """
    x, y = p
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box

    # Compute cross products for each edge
    cp1 = cross_product(x1, y1, x2, y2, x, y)
    cp2 = cross_product(x2, y2, x3, y3, x, y)
    cp3 = cross_product(x3, y3, x4, y4, x, y)
    cp4 = cross_product(x4, y4, x1, y1, x, y)

    # Check if all cross products have the same sign
    if (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0):
        return True  # Point is inside the bounding box
    else:
        return False  # Point is outside the bounding box


def cross_product(x1, y1, x2, y2, x, y):
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
