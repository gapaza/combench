problem1 = {
    'name': 'knapsack-problem1',
    'synergy_matrix': [
        [0, 6, 5, 10, 0, 0, 4, 8, 0, 1],
        [6, 0, 0, 0, 0, 0, 0, 2, 1, 0],
        [5, 0, 0, 0, 8, 0, 0, 0, 0, 8],
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 8, 1],
        [0, 0, 0, 0, 0, 0, 0, 4, 7, 0],
        [4, 0, 0, 0, 0, 0, 0, 4, 10, 0],
        [8, 2, 0, 0, 0, 4, 4, 0, 4, 0],
        [0, 1, 0, 0, 8, 7, 10, 4, 0, 6],
        [1, 0, 8, 0, 1, 0, 0, 0, 6, 0],
    ],
    'values': [7.689238208038703, 1.6044405239774464, 7.084634957525383, 8.75175020944213, 6.052052167652323, 1.6026237903157354, 9.986493082041163, 4.4289263419101506, 6.778143927217658, 5.213429956872445],
    'weights': [6.989541123469603, 7.914993646402801, 5.414968139673826, 9.06884995205389, 2.407164495944209, 4.1316893934825325, 4.738714607892528, 1.9084021312081358, 3.21802253667589, 3.918582202724898],
    'max_weight': 35.0
}



import random
def generate_problem_formulation(size):
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
    print('Synergy matrix:')
    for row in matrix:
        print(row)

    # Random values for each item
    values = [random.uniform(1, 10) for _ in range(size)]
    print('Values: {}'.format(values))

    # Random weights for each item
    weights = [random.uniform(1, 10) for _ in range(size)]
    print('Weights: {}'.format(weights))



if __name__ == '__main__':
    size = 10
    matrix = generate_problem_formulation(size)



