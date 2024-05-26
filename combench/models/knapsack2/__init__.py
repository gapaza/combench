problem1 = {
    'name': 'knapsack2-problem1',
    'values_1': [15.170309095776112, 10.165088002737575, 5.765839236816749, 4.956181858915009, 2.979580450600329, 3.3774389418926227, 7.191001030689883, 15.051316901338053, 19.95900603554269, 2.1655420614212613, 14.19155208351614, 4.529600085281511, 4.98302133373432, 6.778081175338306, 11.004957307901162, 12.951723283335111, 19.751711153371364, 18.096884851381482, 13.70959955116396, 14.035758339399273, 6.8725524196944185, 5.803600358308156, 7.211094815000966, 3.189569618448676, 11.827694607557262, 6.474865116741959, 4.434802953317461, 8.640306999939185, 4.333704340464258, 18.203549636252077],
    'values_2': [5.495198665787081, 1.6581703903487974, 16.213295809651672, 6.108829036033276, 8.68981508328309, 10.178516369545246, 1.486719446298852, 18.19062391649412, 18.765541588341502, 10.312188531285775, 13.58437631739364, 1.7011384377205245, 10.15832914765453, 9.124007742612818, 19.208577993903244, 14.835030755833673, 9.69262466792444, 8.339400862324307, 16.741280032510154, 13.250726701234633, 9.551715450961344, 11.503295522062263, 11.913772434529173, 19.241345546850013, 9.862933614937665, 15.257921922191473, 12.83858108403204, 13.283891126222844, 2.1191051182552956, 17.087074854882747],
    'weights': [10, 2, 2, 8, 9, 3, 1, 3, 9, 4, 6, 9, 1, 10, 9, 5, 8, 6, 2, 5, 5, 7, 9, 3, 5, 2, 3, 6, 7, 10],
    'max_weight': 50
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
    # generate 30 random values
    values = [random.randint(1, 10) for _ in range(30)]
    print(values)
    exit(0)


    size = 10
    matrix = generate_problem_formulation(size)



