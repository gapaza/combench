import random


def generate_synergy_matrix(size):
    # Initialize a matrix with zeros
    matrix = [[0] * size for _ in range(size)]

    # Populate the matrix with random values ensuring symmetry
    for i in range(size):
        for j in range(i + 1, size):
            if random.random() < 0.5:
                value = random.randint(1, 10)
            else:
                value = 0
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix


if __name__ == '__main__':
    # Example usage
    size = 15
    synergy_matrix = generate_synergy_matrix(size)

    for row in synergy_matrix:
        print(row)