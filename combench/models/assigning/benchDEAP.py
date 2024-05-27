import random
from deap import base, creator, tools, algorithms
import numpy as np
import time
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config


from combench.models.assigning import problem1 as pf
from combench.models.assigning.GeneralizedAssigning import GeneralAssigning
from pymoo.indicators.hv import HV


# -------------------------------------------
# Problem Definition
# -------------------------------------------

design_evaluator = GeneralAssigning(pf)
design_set = set()

def evaluate(individual):
    individual_bitstr = ''.join([str(i) for i in individual])
    if individual_bitstr not in design_set:
        design_set.add(individual_bitstr)
    profit, cost, overrun = design_evaluator.evaluate(individual, normalize=True)
    return profit, cost

# -------------------------------------------
# DEAP Toolbox
# -------------------------------------------

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 25)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.04)
toolbox.register("select", tools.selNSGA2)

def run_deap(max_nfe=1000):
    print('Starting the optimization')
    alg_progress = [
        # [0, 0]  # [nfe, hv]
    ]


    pop_size = 100
    population = toolbox.population(n=pop_size)
    # population = []
    # for _ in range(pop_size):
    #     individual = design_evaluator.random_design()
    #     ind_obj = creator.Individual(individual)
    #     population.append(ind_obj)
    # print('Population', population)

    # Define the crossover and mutation probabilities
    NGEN = 1000
    CXPB, MUTPB = 0.7, 0.2

    # Statistics for logging
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'hv'] + stats.fields

    # Reference point for the hypervolume calculation
    ref_point = np.array([0, 1])
    hv_client = HV(ref_point)

    # Option 1: Run the algorithm
    # algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=None)

    # Option 2: Run the algorithm manually to record HV
    fit_vals = []
    for ind in population:
        fit_val = toolbox.evaluate(ind)
        fit_vals.append(deepcopy(fit_val))
        ind.fitness.values = fit_val
    F = np.array(fit_vals)
    hv = hv_client.do(F)
    nfe = len(design_set)
    alg_progress.append([nfe, hv])

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        # print('Offspring', len(offspring))
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(population + offspring, len(population))
        # print('Population', len(population), population)

        # Gather all the fitnesses in one list and print the stats
        record = stats.compile(population)
        hv = hv_client.do(np.array([ind.fitness.values for ind in population]))
        nfe = len(design_set)
        alg_progress.append([nfe, hv])
        logbook.record(gen=gen, nevals=nfe, hv=hv, **record)
        print(logbook.stream)

        if nfe >= max_nfe:
            print('Max NFE reached')
            break

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    fit_values = []
    for ind in pareto_front:
        fit_values.append(ind.fitness.values)
    F = np.array(fit_values)
    hv = hv_client.do(F)
    nfe = len(design_set)
    alg_progress.append([nfe, hv])

    return alg_progress




if __name__ == "__main__":
    alg_nfe = 5000

    all_runs = []
    max_nfes = []
    max_hvs = []
    min_nfes = []
    for x in range(10):
        design_set = set()
        alg_progress = run_deap(max_nfe=alg_nfe)
        all_runs.append(alg_progress)
        max_nfes.append(alg_progress[-1][0])
        max_hvs.append(alg_progress[-1][1])
        min_nfes.append(alg_progress[0][0])
    max_nfe = min(max_nfes)
    min_nfe = max(min_nfes)


    dfs = []
    for idx, alg_progress in enumerate(all_runs):
        nfe = [x[0] for x in alg_progress]
        hv = [x[1] for x in alg_progress]
        nfe_space = np.linspace(min_nfe, max_nfe, max_nfe - min_nfe)
        hv_interp = np.interp(nfe_space, nfe, hv)
        run_df = pd.DataFrame({'nfe': nfe_space, 'hv': hv_interp, 'label': 'GA'})
        dfs.append(run_df)

    # Combine dataframes
    df = pd.concat(dfs, ignore_index=True)
    df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)
    data_frames = df

    # Plot
    plt.clf()
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 5))
    # sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean')
    sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean', linewidth=2.5)
    plt.title('DEAP Results', fontsize=20)
    plt.xlabel('NFE', fontsize=16)
    plt.ylabel('Hypervolume', fontsize=16)
    plt.xticks(fontsize=14)  # Larger x-axis tick labels
    plt.yticks(fontsize=14)  # Larger y-axis tick labels
    plt.legend(fontsize=14, loc='lower right')
    save_path = os.path.join(config.results_dir, 'assigning-bench-deap')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'deap_results.png')
    plt.savefig(save_path)
    plt.show()

    print('Average final HV:', np.mean(max_hvs))
    # Average final HV: 0.651394185724277













