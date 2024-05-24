import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    # Provided cities and route vector
    cities = [(48, 0), (27, 4), (81, 7), (20, 24), (20, 24), (29, 26), (46, 49), (51, 9), (10, 43)]
    route_vector = [7, 1, 0, 2, 3, 4, 8, 5, 3]

    # Call the function to plot the route
    plot_route(cities, route_vector)