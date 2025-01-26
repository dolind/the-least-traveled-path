import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import combinations
import random
import matplotlib.colors as mcolors

class PointsDatabase:
    def __init__(self, num_points):
        self.points = self.generate_points(num_points)
        self.neighbors = self.find_neighbors(self.points)
        self.neighbors_cost = {u: {v: dist for v, dist in self.neighbors[u].items()} for u in self.neighbors}
        self.traveled_edges = {u: {v: 0 for v in self.neighbors[u]} for u in self.neighbors}

    def generate_points(self, num_points):
        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        points = np.array([(i, j) for i in x for j in y])
        center_point = np.array([0.5, 0.5])
        points = np.vstack((points, center_point))
        return points

    def find_neighbors(self, points, radius=0.2):
        dist_matrix = distance_matrix(points, points)
        neighbors = {i: {} for i in range(len(points))}
        for i, j in combinations(range(len(points)), 2):
            if dist_matrix[i][j] <= radius:
                neighbors[i][j] = dist_matrix[i][j]
                neighbors[j][i] = dist_matrix[i][j]
        return neighbors

    def plot_visited_areas(self):
        fig, ax = plt.subplots()

        for point in self.points:
            ax.scatter(*point, color='blue')

        for u, neighbors_dict in self.neighbors.items():
            for v, weight in neighbors_dict.items():
                line_weight = self.traveled_edges[v][u]*0.5
                ax.plot(*zip(self.points[u], self.points[v]), color='black', linewidth=line_weight)

        plt.savefig('traveled_area.png', dpi=300)

    def plot_paths(self, paths):
        fig, ax = plt.subplots()

        for point in self.points:
            ax.scatter(*point, color='blue')

        for u, neighbors_dict in self.neighbors.items():
            for v, weight in neighbors_dict.items():
                ax.plot(*zip(self.points[u], self.points[v]), color='black', linewidth=0.5)

        for path in paths:
            color = random_color()
            for u, v in zip(path[:-1], path[1:]):
                ax.plot(*zip(self.points[u], self.points[v]), color=color)

        plt.savefig('result.png', dpi=300)

def path_length(path, original_neighbors):
    total_length = 0
    for u, v in zip(path[:-1], path[1:]):
        total_length += original_neighbors[u][v]
    return total_length

def find_least_visited_paths(database, start, num_paths=10, max_length=1.1):
    paths = []

    for _ in range(num_paths):
        planned_destination = find_destination(database.points, start, max_length)
        search_destination = planned_destination

        path1, _ = dijkstra_shortest_path(database.neighbors_cost, start, search_destination)

        # Check if the path1 length is greater than half the maximum length
        path1_length = path_length(path1, database.neighbors)
        if path1_length > max_length / 2:
            # Find the last valid point on the path
            temp_length = 0
            new_destination = search_destination
            for u, v in zip(path1[:-1], path1[1:]):
                temp_length += database.neighbors[u][v]
                if temp_length > max_length / 2:
                    new_destination = u
                    break
            # Calculate the new path
            search_destination = new_destination
            path1, _ = dijkstra_shortest_path(database.neighbors_cost, start, new_destination)

        # Update the neighbors_cost and traveled_edges
        for u, v in zip(path1[:-1], path1[1:]):
            database.neighbors_cost[u][v] += database.neighbors[u][v]
            database.neighbors_cost[v][u] += database.neighbors[v][u]
            database.traveled_edges[v][u] += 1
            database.traveled_edges[u][v] += 1

        # Now calculate the return
        path2, _ = dijkstra_shortest_path(database.neighbors_cost, search_destination, start)

        # Combine path1 and path2, excluding the first node of path2 to avoid duplicate nodes
        path1.extend(path2[1:])

        # Update the neighbors_cost and traveled_edges
        for u, v in zip(path2[:-1], path2[1:]):
            database.neighbors_cost[u][v] += database.neighbors[u][v]
            database.neighbors_cost[v][u] += database.neighbors[v][u]
            database.traveled_edges[v][u] += 1
            database.traveled_edges[u][v] += 1

        paths.append(path1)

    return paths

def dijkstra_shortest_path(neighbors, a, b):
    min_heap = [(0, a, [])]
    visited = set()

    while min_heap:
        (cost, current_node, path) = heapq.heappop(min_heap)

        if current_node not in visited:
            visited.add(current_node)
            path = path + [current_node]

            if current_node == b:
                return path, cost

            for neighbor, edge_cost in neighbors[current_node].items():
                if neighbor not in visited:
                    heapq.heappush(min_heap, (cost + edge_cost, neighbor, path))

    return None, float('inf')

def find_destination(points, center_point, max_length):
    centroid_factor = 0.8
    # Find a point within the desired distance from the center_point
    dist_matrix = distance_matrix([points[center_point]], points)
    valid_points = [i for i, d in enumerate(dist_matrix[0]) if 0 < d <= centroid_factor * max_length / 2]

    if valid_points:
        destination = random.choice(valid_points)
    else:
        destination = None

    return destination

def random_color():
    return mcolors.CSS4_COLORS[random.choice(list(mcolors.CSS4_COLORS.keys()))]

num_points = 10
start = 55

database = PointsDatabase(num_points)
paths = find_least_visited_paths(database, start, 40, 1.1)
database.plot_paths(paths)
database.plot_visited_areas()
