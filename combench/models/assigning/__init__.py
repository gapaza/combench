import numpy as np
problem1 = {
    'name': 'assigning-problem',
    'num_agents': 5,
    'num_tasks': 5,
    'costs': np.array([
        [4, 2, 5, 6, 3],
        [7, 5, 8, 6, 2],
        [3, 9, 7, 4, 8],
        [5, 4, 6, 2, 7],
        [6, 7, 5, 3, 4]
    ]),
    'profits': np.array([
        [8, 6, 7, 9, 5],
        [6, 7, 8, 5, 4],
        [5, 9, 6, 7, 8],
        [7, 6, 8, 5, 9],
        [9, 8, 7, 6, 5]
    ]),
    'budgets': np.array([15, 15, 15, 15, 15])
}