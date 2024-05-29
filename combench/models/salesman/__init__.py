problem1 = {
    'name': 'traveling-salesman-problem1',
    'cities': [(61, 16), (4, 44), (37, 26), (47, 72), (100, 85), (40, 93), (2, 73), (19, 85), (79, 50)],
    'costs': [(71, 55), (68, 34), (46, 1), (15, 41), (58, 37), (91, 12), (28, 10), (18, 98), (46, 88)],
}

# problem 2 only has 5 cities
problem2 = {
    'name': 'traveling-salesman-problem2',
    'cities': [(59, 1), (1, 9), (11, 20), (14, 65), (29, 10)],
    'costs': [(71, 55), (68, 34), (46, 1), (15, 41), (58, 37)],
}




import random
import matplotlib.pyplot as plt
import json
import config
import os

mt_tsp_path = os.path.join(config.database_dir, 'multitask-tsp.json')

def generate_problem_set(num_problems, num_cities):
    problems = []
    for i in range(num_problems):
        cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]
        problem = {
            'name': f'multitask-learning-tsp-{num_cities}-{i}',
            'cities': cities,
            'costs': [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]
        }
        problems.append(problem)
    # with open(mt_tsp_path, 'w') as f:
    #     json.dump(problems, f, indent=4)
    return problems

def load_problem_set():
    with open(mt_tsp_path, 'r') as f:
        problems = json.load(f)
    return problems


def generate():
    # Generate 9 coordinate pairs
    coordinates = [
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        # (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        # (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        # (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
        # (random.randint(0, 100), random.randint(0, 100)),  # Cluster 1
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
        plt.text(x, y, f'{route[i]}', fontsize=15, ha='right')

    # Add some additional features to the plot
    plt.title('Route Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    # generate()
   #  plot_route(problem2['cities'],
   #             [
   #          2,
   #          4,
   #          0,
   #          1,
   #          2
   #      ]
   # )
    generate_problem_set(10000, 9)






