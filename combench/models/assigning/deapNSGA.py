import random
from deap import base, creator, tools, algorithms
import numpy as np
import time


from combench.models.assigning import problem1 as pf
from combench.ga.NSGA2 import NSGA2
from combench.models.assigning.GeneralizedAssigning import GeneralAssigning
problem = GeneralAssigning(pf)


design_set = set()
def evaluate(individual):
    individual_bitstr = ''.join([str(i) for i in individual])
    if individual_bitstr not in design_set:
        design_set.add(individual_bitstr)
    profit, cost, overrun = problem.evaluate(individual, normalize=True)
    # print('Evaluating:', individual, len(individual))
    # time.sleep(1)
    # obj1 = sum(individual) / 25
    # obj2 = -sum(individual) / 25
    return profit, cost


# Create the DEAP base structures
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMin)



# Initialize the toolbox
toolbox = base.Toolbox()

# Attribute generator: define 'attr_bool' to be an attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers: define 'individual' to be an individual generator
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 25)

# Define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Register the genetic operators
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.04)
toolbox.register("select", tools.selNSGA2)


def main():
    # Create the population
    population = toolbox.population(n=200)

    # Define the number of generations
    NGEN = 5000
    # Define the crossover and mutation probabilities
    CXPB, MUTPB = 0.7, 0.2

    # Statistics for logging
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run the algorithm
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=None)

    return population


from pymoo.indicators.hv import HV



if __name__ == "__main__":
    final_population = main()
    # Extract the Pareto front
    pareto_front = tools.sortNondominated(final_population, len(final_population), first_front_only=True)[0]

    # Print the Pareto solutions
    fit_values = []
    for ind in pareto_front:
        print(ind, ind.fitness.values)
        fit_values.append(ind.fitness.values)

    ref_point = np.array([0, 1])
    hv_client = HV(ref_point)
    F = np.array(fit_values)
    hv = hv_client.do(F)
    print('Hypervolume:', hv, 'nfe', len(design_set))












