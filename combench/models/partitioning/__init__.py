problem1 = {
    'name': 'set-partitioning-problem1',
    'max_weight': 20,
    'synergy_matrix': [
        [0, 5, 8, 0, 0, 6, 7, 0, 4, 10],
        [5, 0, 0, 9, 10, 0, 1, 0, 0, 0],
        [8, 0, 0, 0, 0, 10, 0, 0, 0, 7],
        [0, 9, 0, 0, 2, 0, 0, 10, 9, 8],
        [0, 10, 0, 2, 0, 9, 5, 0, 0, 0],
        [6, 0, 10, 0, 9, 0, 0, 7, 0, 0],
        [7, 1, 0, 0, 5, 0, 0, 0, 9, 0],
        [0, 0, 0, 10, 0, 7, 0, 0, 0, 10],
        [4, 0, 0, 9, 0, 0, 9, 0, 0, 0],
        [10, 0, 7, 8, 0, 0, 0, 10, 0, 0],
    ],
    'cost_matrix': [
        [0, 8, 10, 9, 1, 7, 6, 4, 5, 3],
        [8, 0, 6, 6, 7, 8, 2, 9, 1, 2],
        [10, 6, 0, 4, 7, 1, 9, 5, 3, 4],
        [9, 6, 4, 0, 6, 4, 3, 2, 5, 4],
        [1, 7, 7, 6, 0, 3, 5, 10, 5, 8],
        [7, 8, 1, 4, 3, 0, 2, 1, 9, 9],
        [6, 2, 9, 3, 5, 2, 0, 4, 3, 10],
        [4, 9, 5, 2, 10, 1, 4, 0, 3, 2],
        [5, 1, 3, 5, 5, 9, 3, 3, 0, 7],
        [3, 2, 4, 4, 8, 9, 10, 2, 7, 0],
    ],
    'weights': [3, 5, 7, 4, 6, 5, 4, 3, 6, 7]
}






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











