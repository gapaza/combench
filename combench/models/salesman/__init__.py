problem1 = {
    'name': 'traveling-salesman-problem1',
    'cities': [(61, 16), (4, 44), (37, 26), (47, 72), (100, 85), (40, 93), (2, 73), (19, 85), (79, 50)],
    'costs': [(71, 55), (68, 34), (46, 1), (15, 41), (58, 37), (91, 12), (28, 10), (18, 98), (46, 88)],
}








import random
import matplotlib.pyplot as plt

def generate():
    # Generate 9 coordinate pairs
    coordinates = [
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1

    ]

    print(coordinates)

    # Plot the coordinates to visualize
    x_vals, y_vals = zip(*coordinates)
    plt.scatter(x_vals, y_vals)

    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, f'({x},{y})', fontsize=9, ha='right')

    plt.title('Difficult TSP Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_route(cities, route):
    # Extract coordinates from the cities list using the route vector
    route_coords = [cities[i] for i in route]

    # Unzip the coordinates into x and y lists
    x_coords, y_coords = zip(*route_coords)

    plt.figure(figsize=(10, 8))

    # Plot the route
    plt.plot(x_coords, y_coords, 'o-', label='Route')

    # Highlight the start node
    start_node = route_coords[0]
    plt.plot(start_node[0], start_node[1], 'go', markersize=10, label='Start Node')

    # Annotate the points with their indices
    for i, (x, y) in enumerate(route_coords):
        plt.text(x, y, f'{i}', fontsize=9, ha='right')

    # Add some additional features to the plot
    plt.title('Route Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()









