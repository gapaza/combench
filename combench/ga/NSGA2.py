from tqdm import tqdm
from copy import deepcopy
import os
import config
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import association_rules, apriori

class BenchNSGA2:

    def __init__(self, problem, max_nfe, run_name='runner'):
        self.problem = problem
        self.max_nfe = max_nfe
        self.run_name = run_name
        self.nsga2_name = 'nsga2'

        # Save
        self.run_name = run_name
        self.save_dir = os.path.join(config.results_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.results_file = os.path.join(self.save_dir, 'results.json')

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        return results

    def plot_results(self, comparison=None):
        results = self.load_results()
        dfs = []

        max_hvs = []
        max_nfes = []
        min_nfes = []
        for alg_run in results:
            nfe, hv = zip(*alg_run)
            min_nfe, max_nfe = min(nfe), max(nfe)
            max_nfes.append(max_nfe)
            min_nfes.append(min_nfe)
            max_hvs.append(max(hv))
        max_nfe = min(max_nfes)
        min_nfe = max(min_nfes)


        for alg_run in results:
            nfe, hv = zip(*alg_run)
            # min_nfe, max_nfe = min(nfe), max(nfe)
            nfe_space = np.linspace(min_nfe, max_nfe, max_nfe - min_nfe)
            hv_interp = np.interp(nfe_space, nfe, hv)
            df = pd.DataFrame({'nfe': nfe_space, 'hv': hv_interp, 'label': self.nsga2_name})
            dfs.append(df)

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
        plt.title('NSGA2 Results', fontsize=20)
        plt.xlabel('NFE', fontsize=16)
        plt.ylabel('Hypervolume', fontsize=16)
        plt.xticks(fontsize=14)  # Larger x-axis tick labels
        plt.yticks(fontsize=14)  # Larger y-axis tick labels
        plt.legend(fontsize=14, loc='lower right')
        save_path = os.path.join(self.save_dir, 'nsga2_results.png')
        plt.savefig(save_path)
        plt.show()

        print('Average final HV:', np.mean(max_hvs))




    def run(self, populations):
        results = []
        for pop in populations:
            nsga2 = NSGA2(pop, self.problem, self.max_nfe, run_name=self.nsga2_name)
            result = nsga2.run()
            results.append(result)
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print('Results saved to {}'.format(self.results_file))
        return results


class NSGA2:


    def __init__(self, population, problem, max_nfe=1000, run_name='NSGA2', save_dir=None):
        self.population = population
        self.problem = problem
        self.max_nfe = max_nfe
        self.save_dir = save_dir

        # Save
        self.run_name = run_name
        if save_dir is None:
            self.save_dir = os.path.join(config.results_dir, run_name)
        else:
            self.save_dir = os.path.join(save_dir, run_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def run(self):

        self.population.init_population()
        self.population.eval_population()
        self.population.prune()
        # self.population.record()

        progress_bar = tqdm(total=self.max_nfe)
        curr_nfe = 0
        terminated = False
        gen_without_nfe_increase = 0
        while terminated is False and curr_nfe < self.max_nfe:
            curr_nfe = deepcopy(self.population.nfe)

            self.population.create_offspring()
            self.population.eval_population()

            # Prune population
            self.population.prune()

            # Log iteration
            # if curr_nfe % 10 == 0:
            #     self.population.record()
            self.population.record()
            update_delta = self.population.nfe - curr_nfe
            if update_delta == 0:
                gen_without_nfe_increase += 1
            else:
                gen_without_nfe_increase = 0
            progress_bar.update(update_delta)
            progress_bar.set_postfix({'hv': self.population.hv[-1]})

            if gen_without_nfe_increase > 30:
                terminated = True
                print('Terminated due to no increase in NFE for 10 generations')


        progress_bar.close()

        # Plot hv and population
        self.population.plot_population(self.save_dir)
        self.population.plot_hv(self.save_dir)
        results = [(nfe, hv) for nfe, hv in zip(self.population.nfes, self.population.hv)]
        # self.rule_mining()
        return results


    def rule_mining(self):
        # Association rule mining on the designs
        designs = [design for design in self.population.unique_designs if design.get_plotting_objectives()[0] > 0]
        print('DESIGNS:', len(designs), len(self.population.unique_designs))
        designs_low_stiff = []
        designs_high_stiff = []
        for design in designs:
            if abs(design.objectives[0]) < 0.4:
                designs_low_stiff.append(design.vector)
            else:
                designs_high_stiff.append(design.vector)

        set1 = np.array(designs_low_stiff, dtype=bool)
        set2 = np.array(designs_high_stiff, dtype=bool)

        print('SET1:', set1.shape)
        print('SET2:', set2.shape)

        df_set1 = pd.DataFrame(set1, columns=[f'attr{i}' for i in range(1, config.num_vars + 1)])
        df_set2 = pd.DataFrame(set2, columns=[f'attr{i}' for i in range(1, config.num_vars + 1)])

        # Add labels to distinguish sets
        df_set1['label'] = 'Low Stiffness'
        df_set2['label'] = 'High Stiffness'

        # Concatenate the two sets
        df_combined = pd.concat([df_set1, df_set2], ignore_index=True)

        # Ensure boolean values are represented as integers (0 or 1)
        df_combined.iloc[:, :-1] = df_combined.iloc[:, :-1].astype(bool)

        # Find frequent itemsets
        frequent_itemsets = apriori(df_combined.iloc[:, :-1], min_support=0.1, use_colnames=True)

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

        # Display the results
        print(rules)







