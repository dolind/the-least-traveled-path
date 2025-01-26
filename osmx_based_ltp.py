import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pyproj
from matplotlib.colors import LinearSegmentedColormap
from utm import from_latlon

MAX_DISTANCE = 2000


class OSMMap:
    def __init__(self, latitude, longitude, dist=2000, network_type='walk', file_path='map.graphml', location=''):
        self.latitude = latitude
        self.longitude = longitude
        self.dist = dist
        self.network_type = network_type
        self.file_path = location + file_path
        self.location = location

        if not os.path.exists(self.file_path):
            self.download_map()
            self.initTravelTime()
            self.save_map()
        else:
            self.load_map()

    def download_map(self):
        self.G = ox.graph_from_point((self.latitude, self.longitude), dist=self.dist, network_type=self.network_type)

    def save_map(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.G, f)

    def load_map(self):
        with open(self.file_path, 'rb') as f:
            self.G = pickle.load(f)

    def display_map(self):
        fig, ax = ox.plot_graph(self.G)
        plt.savefig('map.png', dpi=300)

    def get_points_from_graph(self):
        points = {n: (data['y'], data['x']) for n, data in self.G.nodes(data=True)}

        # Define the WGS84 ellipsoid model
        wgs84 = pyproj.CRS('EPSG:4326')

        # Calculate the UTM zone based on the input latitude and longitude
        utm_result = from_latlon(self.latitude, self.longitude)
        utm_zone = utm_result[2]
        utm_crs_code = f"EPSG:326{utm_zone}" if latitude >= 0 else f"EPSG:327{utm_zone}"

        # Define the projection for meters (UTM)
        utm = pyproj.CRS(utm_crs_code)

        # Define the transformation function
        transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)

        # Transform the reference point to meters
        self.x_ref, self.y_ref = transformer.transform(self.latitude, self.longitude)

        def transform_point(lat, lon):
            x, y = transformer.transform(lon, lat)
            return x - self.x_ref, y - self.y_ref

        # Transform the points
        transformed_points = {n: transform_point(lat, lon) for n, (lat, lon) in points.items()}

        return transformed_points

    def get_nearest_node(self, point):
        nearest_node, dist = ox.nearest_nodes(self.G, point[1], point[0], return_dist=True)
        return nearest_node

    def get_least_travelled_path(self, start_point, end_point):
        path1 = self.calculate_shortest_path(start_point, end_point)

        cumulative_length = 0
        last_valid_node = path1[0]
        last_valid_node_idx = 0
        max_path_length = MAX_DISTANCE
        # Check if the path1 length is greater than half the maximum length
        for i in range(len(path1) - 1):

            data = min(self.G.get_edge_data(path1[i], path1[i + 1]).values(), key=lambda d: d["length"])

            cumulative_length += data['length']

            if cumulative_length > max_path_length:
                break

            last_valid_node = path1[i + 1]
            last_valid_node_idx = i + 1

            last_valid_node = path1[i + 1]

        last_valid_node_coords = (self.G.nodes[last_valid_node]['y'], self.G.nodes[last_valid_node]['x'])
        path1 = path1[:last_valid_node_idx + 1]

        self.update_travel_times(path1)

        # Now calculate the return
        path2 = self.calculate_shortest_path(last_valid_node_coords, start_point)
        self.update_travel_times(path2)
        # Combine path1 and path2, excluding the first node of path2 to avoid duplicate nodes
        path1.extend(path2[1:])

        return path1

    def calculate_shortest_path(self, start_point, end_point):
        start_node = self.get_nearest_node(start_point)
        end_node = self.get_nearest_node(end_point)

        print(f"Start node: {start_node}, coordinates: {self.get_points_from_graph()[start_node]}")
        print(f"End node: {end_node}, coordinates: {self.get_points_from_graph()[end_node]}")

        if start_node != end_node:
            path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
            print(f"Shortest path: {path}")
            return path
        else:
            print("Start and end nodes are the same.")
            return [start_node]

    def display_route(self, route):
        fig, ax = ox.plot_graph_routes(self.G, route, route_colors=['red', 'blue', 'green'], route_linewidth=6,
                                       node_size=0.5, show=False, save=True, dpi=300, filepath=self.location + '_selected_paths.png',)

    def initTravelTime(self):
        for u, v, edge in self.G.edges(data=True):
            edge['travel_time'] = edge['length'] / (100 * 1000) / 3600  # Convert speed limit to m/s
            edge['visits'] = 0

    def update_travel_times(self, last_route):
        route_edges = {(last_route[i], last_route[i + 1]) for i in range(len(last_route) - 1)}

        for u, v, edge in osm_map.G.edges(data=True):
            if (u, v) in route_edges or (v, u) in route_edges:
                # If the edge is in the route, start with a speed of 100 and reduce it by half
                edge['travel_time'] = edge['travel_time'] * 2
                edge['visits'] += 1

    def plot_visits(self):
        # Normalize the 'visits' attribute for better visualization
        max_visits = max(edge['visits'] for _, _, edge in self.G.edges(data=True))
        normalized_visits = [(edge['visits'] / max_visits) for _, _, edge in self.G.edges(data=True)]

        # Create a color map based on the normalized 'visits' attribute
        colors = [
            (0.0, "black"),  # 0 is black
            (0.01, "blue"),  # Slightly above 0 transitions to blue
            (0.1, "purple"),  # Slightly above 0 transitions to blue
            (0.25, "green"),  # Middle values transition to white
            (0.5, "white"),  # Middle values transition to white
            (0.75, "orange"),  # Middle values transition to white
            (1.0, "red")  # Max value (300) transitions to red
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        edge_colors = [custom_cmap(visits) for visits in normalized_visits]

        # Plot the graph with edge colors based on the 'visits' attribute
        fig, ax = ox.plot_graph(
            self.G,
            node_size=0.01,
            node_zorder=3,
            edge_color=edge_colors,
            edge_linewidth=1,
            save=True,
            filepath=self.location + '_visits.png',
            dpi=300,
            show=False
        )



def random_point_on_circle(center, radius):
    from geopy import Point
    import random
    import math
    # Convert the center point to a geopy Point object
    center_point = Point(center)

    # Generate a random angle in radians
    angle = random.uniform(0, 2 * math.pi)

    # Calculate the new latitude and longitude
    new_latitude = center_point.latitude + (radius * math.sin(angle)) / 111000
    new_longitude = center_point.longitude + (radius * math.cos(angle)) / (111000 * math.cos(center_point.latitude))

    # Return the new point as a tuple (latitude, longitude)
    return (new_latitude, new_longitude)


import argparse

parser = argparse.ArgumentParser(description="Process coordinates and paths.")
parser.add_argument('--latitude', type=float, required=False, help="Latitude value", default=51.5155211)
parser.add_argument('--longitude', type=float, required=False, help="Longitude value", default=-0.1321471)
parser.add_argument('--paths_visit', type=int, required=False, help="Number of paths to visit", default=100)
parser.add_argument('--paths_output', type=int, required=False, help="Number of output paths", default=3)
parser.add_argument('--location', required=False, help="Location", default='london')

args = parser.parse_args()

latitude = args.latitude
longitude = args.longitude
paths_visit = args.paths_visit
paths_output = args.paths_output
location = args.location


# Initialize the OSMMap class with the given coordinates
osm_map = OSMMap(latitude, longitude, location=location)

# Define your start and end points (in meters, relative to the reference point)
start_point = (latitude, longitude)

# Calculate the shortest path between the start and end points
paths = []
for i in range(0, paths_visit):
    end_point = random_point_on_circle(start_point, MAX_DISTANCE / 2)
    shortest_path = osm_map.get_least_travelled_path(start_point, end_point)
for i in range(0, paths_output):
    end_point = random_point_on_circle(start_point, MAX_DISTANCE / 2)
    shortest_path = osm_map.get_least_travelled_path(start_point, end_point)
    paths.append(shortest_path)

osm_map.save_map()

# Display the route using matplotlib
osm_map.display_route(paths)
osm_map.plot_visits()
