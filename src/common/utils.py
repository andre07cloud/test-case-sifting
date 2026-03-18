import json
import numpy as np
import yaml
import pandas as pd
from typing import List, Tuple
import math


class GeometryUtils:
    """Geometry and spatial utility functions for robot manipulation test cases."""

    BOX_WIDTH = 0.14   # x 0.13205
    BOX_HEIGHT = 0.1789  # _y 0.16495
    ROBOT_COORD = np.array([-0.04, 0.257])

    @staticmethod
    def filter_duplicates_3d(failed_picks_locs):
        """Function that filters out the duplicates in the failed pick locations"""
        filtered_failed_picks_locs = []
        for i, pick in enumerate(failed_picks_locs):
            if i == 0:
                filtered_failed_picks_locs.append(pick)
            else:
                if pick[0] != failed_picks_locs[i-1][0] or pick[1] != failed_picks_locs[i-1][1]:
                    filtered_failed_picks_locs.append(pick)
        return filtered_failed_picks_locs

    @staticmethod
    def filter_values_1d(values):
        """Function that filters out the duplicates in the failed pick locations"""
        filtered_values = []
        for i, value in enumerate(values):
            if i == 0:
                filtered_values.append(value)
            else:
                if value != values[i-1]:
                    filtered_values.append(value)
        return filtered_values

    @staticmethod
    def get_rotations_from_location(failed_picks_locs, test):
        """Function that finds the closest points to the given points
        and returns the corresponding rotations"""
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        points = [[x1, y1], [x2, y2], [x3, y3]]
        rotations = [(90 - test[0][0][1][2]), (90 - test[0][1][1][2]), (90 - test[0][2][1][2])]

        closest_rotations = []

        for point in failed_picks_locs:
            min_distance = float('inf')
            for i, other_point in enumerate(points):
                distance = np.linalg.norm(np.array(point[:2]) - np.array(other_point))
                if distance < min_distance:
                    min_distance = distance
                    closest_rotation = rotations[i]
            closest_rotations.append(closest_rotation)

        return closest_rotations

    @staticmethod
    def distance_between_points(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def closest_neighbor_distances(points):
        distances = []
        for i, point in enumerate(points):
            min_distance = float('inf')
            for j, other_point in enumerate(points):
                if i != j:
                    distance = GeometryUtils.distance_between_points(point, other_point)
                    if distance < min_distance:
                        min_distance = distance
            distances.append(min_distance)
        return distances

    @staticmethod
    def get_inter_box_closest_dist(test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        points = [[x1, y1], [x2, y2], [x3, y3]]
        distances = GeometryUtils.closest_neighbor_distances(points)
        return distances

    @staticmethod
    def closest_pair_of_points_rotations(test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        points = [[x1, y1], [x2, y2], [x3, y3]]
        rotations = [abs(90 - test[0][0][1][2]), abs(90 - test[0][1][1][2]), abs(90 - test[0][2][1][2])]
        min_distance = float('inf')
        closest_pair = None
        closest_rotations = None

        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                distance = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (points[i], points[j])
                    closest_rotations = (rotations[i], rotations[j])

        return closest_pair, closest_rotations

    @staticmethod
    def get_inter_box_dist(test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        points = [[x1, y1], [x2, y2], [x3, y3]]

        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                distances.append(distance)

        return distances

    @staticmethod
    def get_inter_box_dist_given_box(point, test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        points = [[x1, y1], [x2, y2], [x3, y3]]

        distances = []
        for i in range(len(points)):
            distance = np.linalg.norm(np.array(point) - np.array(points[i]))
            distances.append(distance)

        distances = sorted(distances)
        return distances[1]

    @staticmethod
    def get_robot_dist_failed(failed_picks_locs):
        if len(failed_picks_locs[0]) > 2:
            points = [p[:-1] for p in failed_picks_locs]
        else:
            points = [p for p in failed_picks_locs]
        robot_coord = np.array([-0.04, 0.257])

        distances = []
        for i in range(len(points)):
            distance = np.linalg.norm(np.array(points[i]) - robot_coord)
            distances.append(distance)

        return distances

    @staticmethod
    def get_robot_dist(test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        robot_coord = np.array([-0.04, 0.257])
        points = [[x1, y1], [x2, y2], [x3, y3]]

        distances = []
        for i in range(len(points)):
            distance = np.linalg.norm(np.array(points[i]) - robot_coord)
            distances.append(distance)

        return distances

    @staticmethod
    def get_x_y_list(test):
        x1, y1 = test[0][0][0][:-1]
        x2, y2 = test[0][1][0][:-1]
        x3, y3 = test[0][2][0][:-1]

        return np.array([x1, x2, x3]), np.array([y1, y2, y3])

    @staticmethod
    def get_x_y_list_fail(positions):
        x = []
        y = []

        for pos in positions:
            x.append(pos[0])
            y.append(pos[1])

        return np.array(x), np.array(y)
