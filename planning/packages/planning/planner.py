from typing import List, Tuple
import math
import numpy as np
import networkx as nx

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    PlacedPrimitive,
    PlanningQuery,
    PlanningResult,
    PlanningSetup,
    PlanStep,
    Circle,
    Rectangle,
    SimulationResult,
    simulate,
)

__all__ = ["Planner"]

# Constants
GRID_SIZE = 0.5  # distance between 2 nodes in x and y direction in the 3d graph
THETA_STEP_DEG = 15  # [deg] minimum theta step for each node in the 3d graph
SAFE_DISTANCE = 0.5 # Safety tolerance around the robot pose to detect collision

def get_relative_angle_between_nodes(node1: FriendlyPose, node2: FriendlyPose):

    # Determine relative angle between the node1 and node2
    delta_x = node2.x - node1.x
    delta_y = node2.y - node1.y
    relative_angle_rad = math.atan2(delta_y, delta_x)
    relative_angle_deg = math.degrees(relative_angle_rad)

    return relative_angle_deg


def get_angle_to_rotate(start_angle, end_angle):
    """
    Function to determine the shortest amount of rotation to perform to reach from start_angle to end_angle
    :param start_angle: [deg]
    :param end_angle: [deg]
    :return: [deg] rotation angle degrees
    """

    # Check if the difference is greater than 180 degrees
    diff_between_angles = end_angle - start_angle
    if diff_between_angles > 180:
        diff_between_angles -= 360
    elif diff_between_angles < -180:
        diff_between_angles += 360

    return diff_between_angles

def is_point_in_bounding_box(p, box):
    """
    Function to check if the given point is in the bounding box
    :param p: [Tuple] Point with x,y coordinates
    :param box: [List] A list of tuple coordinates as the 4 corners of the box
    :retrun: [bool] True if the point is in the bounding box else False
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

def get_rectangle_global_coordinates(rect: PlacedPrimitive):
    """
    Function to get the global coordinates of a rectangle which is provides with its center (x,y), theta orientation
    and xmin, xmax, ymin and ymax values
    :return: [List] List of tuple of (x,y) coordinates of the rectangle in global frame of reference
    """

    theta = math.radians(rect.pose.theta_deg)
    x_c = rect.pose.x
    y_c = rect.pose.y

    x = rect.primitive.xmax + SAFE_DISTANCE
    y = rect.primitive.ymax + SAFE_DISTANCE

    # top right coordinates
    x1 = x_c + x * math.cos(theta) - y * math.sin(theta)
    y1 = y_c + x * math.sin(theta) + y * math.cos(theta)

    # bottom right coordinates
    x2 = x_c + x * math.cos(theta) + y * math.sin(theta)
    y2 = y_c + x * math.sin(theta) - y * math.cos(theta)

    # bottom left coordinates
    x3 = x_c - x * math.cos(theta) + y * math.sin(theta)
    y3 = y_c - x * math.sin(theta) - y * math.cos(theta)

    # top left coordinates
    x4 = x_c - x * math.cos(theta) - y * math.sin(theta)
    y4 = y_c - x * math.sin(theta) + y * math.cos(theta)

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def check_and_get_node(given_node, nodes_list):
    """
    Function to check if the given_node exists in the nodes_list and if not then get the node closest to the given_node
    within the nodes_list
    """
    # Check if given node exist in the nodes_list
    # if not then find nearest node to the given node in the nodes_list
    nearest_given_node = ()
    if given_node not in nodes_list:
        print(f"Finding the nearest node to the given_node = {given_node}")
        nearest_given_node = find_nearest_node(given_node)
        print(f"Found nearest node = {nearest_given_node}")
        if nearest_given_node not in nodes_list:
            raise Exception(
                f"The nearest given_node {nearest_given_node} is not in the nodes_list. "
                f"Make sure the given_node is one of the nodes in the grid.")

    return nearest_given_node

def find_nearest_node(current_node: Tuple) -> Tuple:
    """
    Function to find the nearest node to the given node for a given global grid size and THETA_STEP_DEG
    """

    x, y, theta = current_node

    nearest_x = round(x/GRID_SIZE)*GRID_SIZE
    nearest_y = round(y/GRID_SIZE)*GRID_SIZE
    nearest_theta = (round(theta/THETA_STEP_DEG)*THETA_STEP_DEG)%360

    nearest_node = (nearest_x, nearest_y, nearest_theta)

    return nearest_node

class Planner:
    params: PlanningSetup

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: PlanningSetup):
        context.info("initialized")
        self.params = data

        # This is the interval of allowed linear velocity
        # Note that min_velocity_x_m_s and max_velocity_x_m_s might be different.
        # Note that min_velocity_x_m_s may be 0 in advanced exercises (cannot go backward)
        max_velocity_x_m_s: float = self.params.max_linear_velocity_m_s
        min_velocity_x_m_s: float = self.params.min_linear_velocity_m_s

        # This is the max curvature. In earlier exercises, this is +inf: you can turn in place.
        # In advanced exercises, this is less than infinity: you cannot turn in place.
        max_curvature: float = self.params.max_curvature

        # these have the same meaning as the collision exercises
        body: List[PlacedPrimitive] = self.params.body
        environment: List[PlacedPrimitive] = self.params.environment

        # these are the final tolerances - the precision at which you need to arrive at the goal
        tolerance_theta_deg: float = self.params.tolerance_theta_deg
        tolerance_xy_m: float = self.params.tolerance_xy_m

        # For convenience, this is the rectangle that contains all the available environment,
        # so you don't need to compute it
        bounds: Rectangle = self.params.bounds

        # Determine the bounds of the graph in which the robot moves
        self.xmin = bounds.xmin
        self.xmax = bounds.xmax
        self.ymin = bounds.ymin
        self.ymax = bounds.ymax

        print(f"Bounds = {self.xmax, self.xmin, self.ymax, self.ymin}")

    def on_received_query(self, context: Context, data: PlanningQuery):
        # A planning query is a pair of initial and goal poses
        start_pose: FriendlyPose = data.start
        end_pose: FriendlyPose = data.target

        # You start at the start pose. You must reach the goal with a tolerance given by
        # tolerance_xy_m and tolerance_theta_deg.

        # You need to declare if it is feasible or not
        feasible = True

        if not feasible:
            # If it's not feasible, just return this.
            result: PlanningResult = PlanningResult(False, None)
            context.write("response", result)
            return

        # Determine if the environment contains obstacles
        print(f"Environment = {self.params.environment}")
        print(f"Number of obstacles by bounding box = {len(self.params.environment)}")
        # If no obstacles are present then connect the start and end pose directly
        if len(self.params.environment) <= 4:
            # Get plan steps between the start and the end pose when there are no obstacles in the environment
            plan, total_duration = self.get_plan_wo_obstacles(start_pose, end_pose)
        else:
            # Get plan steps when there are obstacles in the environment
            start_node = (start_pose.x, start_pose.y, start_pose.theta_deg)
            end_node = (end_pose.x, end_pose.y, end_pose.theta_deg)
            graph_w_edges = self.generate_3d_graph(start_node, end_node)
            shortest_path = self.get_shortest_path_for_graph(graph_w_edges, start_node, end_node)
            plan = []
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                plan.append(graph_w_edges[u][v]['plan'])

        result: PlanningResult = PlanningResult(feasible, plan)
        context.write("response", result)

    def get_straight_line_motion_plan_step(self, distance_m) -> (PlanStep, float):
        """
        Function to get plan step for straight line motion
        :param distance_m: [m] Distance in meters to travel
        :return:
        """

        duration_straight_m_s = distance_m / self.params.max_linear_velocity_m_s
        straight_line_plan_step = PlanStep(
            duration=duration_straight_m_s,
            angular_velocity_deg_s=0.0,
            velocity_x_m_s=self.params.max_linear_velocity_m_s,
        )

        return straight_line_plan_step, duration_straight_m_s

    def get_angular_motion_plan_step(self, angle_deg) -> (PlanStep, float):
        """
        Function to get a plan step for angular motion
        :param angle_deg: [deg] Angle in degrees to traverse
        :return:
        """

        duration_turn_deg_s = abs(angle_deg / self.params.max_angular_velocity_deg_s)
        angular_velocity_sign = 1 if angle_deg >=0 else -1
        angular_motion_plan_step = PlanStep(
            duration=duration_turn_deg_s,
            angular_velocity_deg_s=self.params.max_angular_velocity_deg_s * angular_velocity_sign,
            velocity_x_m_s=0.0,
        )

        return angular_motion_plan_step, duration_turn_deg_s

    def get_plan_wo_obstacles(self, start_pose: FriendlyPose, end_pose: FriendlyPose) -> (List[PlanStep], float):
        """
        Function to get the plan which is list of PlanSteps for the robot to get from given start pose to the end pose
        when there are no obstacles in the environment
        :param start_pose: [FriendlyPose] Robot pose at the start location
        :param end_pose: [FriendlyPose] Robot pose at the end location
        :return: List of PlanStep
        """

        # Initialize plan_steps with an empty list
        plan_steps = []

        # Initialize Total Plan duration
        total_plan_duration_sec = 0

        # Determine the distance between the start and end pose
        start_coord = (start_pose.x, start_pose.y)
        end_coord = (end_pose.x, end_pose.y)
        distance_btw_poses = math.dist(start_coord, end_coord)

        # Determine relative angle between the start and end pose
        relative_angle_deg = get_relative_angle_between_nodes(start_pose, end_pose)

        # Step 1:
        # Rotate from start pose to the relative angle between the 2 poses
        start_pose_to_relative_angle_rotation_deg = get_angle_to_rotate(start_pose.theta_deg, relative_angle_deg)
        plan1, duration1 = self.get_angular_motion_plan_step(start_pose_to_relative_angle_rotation_deg)
        plan_steps.append(plan1)
        total_plan_duration_sec += duration1

        # Step 2:
        # Translate from start pose to end pose
        if distance_btw_poses > 0:
            plan2, duration2 = self.get_straight_line_motion_plan_step(distance_btw_poses)
            plan_steps.append(plan2)
            total_plan_duration_sec += duration2

        # Step 3:
        # Rotate from relative angle between 2 poses to the end pose
        relative_angle_to_end_pose_rotation_deg = get_angle_to_rotate(relative_angle_deg, end_pose.theta_deg)
        plan3, duration3 = self.get_angular_motion_plan_step(relative_angle_to_end_pose_rotation_deg)
        plan_steps.append(plan3)
        total_plan_duration_sec += duration3


        return plan_steps, total_plan_duration_sec

    def generate_3d_graph(self, given_start_node: Tuple, given_end_node: Tuple):

        # Get x and y value array for the given bounds and grid size
        x_values = np.arange(self.xmin, self.xmax+GRID_SIZE, GRID_SIZE)
        y_values = np.arange(self.ymin, self.ymax+GRID_SIZE, GRID_SIZE)

        # Generate nodes list which do not lie within any obstacle
        nodes_list = []

        for x in x_values:
            for y in y_values:
                # Check if the x and y of the node is within any of the obstacles
                is_node_safe = True
                node_xy = (x,y)
                # Get obstacles without the bounding box
                obstacles_wo_bounding_box =self.params.environment[4:]
                for obstacle in obstacles_wo_bounding_box:
                    if not self.check_if_node_is_safe(node_xy, obstacle):
                        print(f"Point {node_xy} is within {obstacle}")
                        is_node_safe = False
                        break
                if not is_node_safe:
                    continue
                # Add theta steps to node_xy to generate nodes with 3rd dimension of theta_deg
                for theta in range(0, 360, THETA_STEP_DEG):
                    node = (x, y, theta)
                    nodes_list.append(node)

        # Check and get start and end node
        start_node = check_and_get_node(given_start_node, nodes_list)
        end_node = check_and_get_node(given_end_node, nodes_list)

        # Create a Multi Directional Graph
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(nodes_list)

        # Add edges to the graph
        graph_w_edges = self.add_edges_to_graph(graph)

        return graph_w_edges

    def check_if_node_is_safe(self, node: Tuple[float, float], obstacle: PlacedPrimitive) -> bool:
        """
        Checks if a node is a safe distance from an obstacle.
        """
        if isinstance(obstacle.primitive, Circle):
            # Return True if the distance between the circle center and the node is greater
            # than the radius of the circle + some safety buffer
            return math.dist(node, (obstacle.pose.x, obstacle.pose.y)) > (obstacle.primitive.radius + SAFE_DISTANCE)
        
        elif isinstance(obstacle.primitive, Rectangle):
            rect_corners = get_rectangle_global_coordinates(obstacle)
            # Return True if the point is not in bounding box
            return not is_point_in_bounding_box(node, rect_corners)

        return True  # Safe if obstacle type is unknown

    def add_edges_to_graph(self, graph):

        # Add edges between each node and its neighbor
        nodes_list = list(graph.nodes)
        nodes_x_y_list = list({(x, y) for x, y, _ in graph.nodes})

        for node_xy in nodes_x_y_list:
            # Extract x,y from node
            x, y = node_xy

            neighbor_x_values = [x + GRID_SIZE, x, x - GRID_SIZE]
            neighbor_y_values = [y + GRID_SIZE, y, y - GRID_SIZE]

            for neighbor_x in neighbor_x_values:
                # skip if neighbor_x is outside x bounds
                if not self.xmin <= neighbor_x <= self.xmax:
                    continue
                for neighbor_y in neighbor_y_values:
                    # skip if neighbor_y is outside y bounds
                    if not self.ymin <= neighbor_y <= self.ymax:
                        continue
                    # skip if the neighbor node overlaps with current node
                    neighbor_node_xy = (neighbor_x, neighbor_y)
                    if node_xy == neighbor_node_xy:
                        continue
                    # skip if the neighbor_node_xy does not belong within the nodes_list
                    elif neighbor_node_xy not in nodes_x_y_list:
                        continue
                    for neighbor_theta in range(0, 360, THETA_STEP_DEG):
                        neighbor_node = (neighbor_x, neighbor_y, neighbor_theta)

                        # Add edges between original node and neighbor nodes
                        for original_node_theta in range(0, 360, THETA_STEP_DEG):
                            original_node = (x, y, original_node_theta)

                            # Get the plan and duration to connect the original and neighbor node
                            plan, plan_duration_sec = self.get_plan_wo_obstacles(
                                FriendlyPose(x, y, original_node_theta),
                                FriendlyPose(neighbor_x, neighbor_y, neighbor_theta)
                            )
                            graph.add_edge(original_node, neighbor_node, plan=plan, weight=plan_duration_sec)

        return graph

    def get_shortest_path_for_graph(self, graph, start_node, end_node):

        try:
            shortest_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
            print(shortest_path)
            return shortest_path
        except nx.NetworkXNoPath as e:
            print(e)