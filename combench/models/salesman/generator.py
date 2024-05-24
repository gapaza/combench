import random
import matplotlib.pyplot as plt

# Seed random number generator for reproducibility
# random.seed(852)



if __name__ == '__main__':
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