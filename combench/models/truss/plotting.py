import config
import json
import numpy as np
import random
from copy import deepcopy
import os
from combench.models.truss import rep
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from combench.models import truss


def plot_weight_graph(designs, save_file):
    valid_designs = [design for design in designs if ((design.epoch is not None) and (design.weight is not None))]
    if len(valid_designs) == 0:
        return

    all_weights = [design.weight for design in designs]
    all_weights = list(set(all_weights))
    all_weights.sort()

    # Create GridSpec layout
    rows, cols = 3, 3
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(3.5 * cols, 3.5 * rows), dpi=200)  # Adjust the figure size based on rows and cols

    for idx, weight in enumerate(all_weights):
        w_designs = [design for design in designs if design.weight == weight]
        row = idx // cols
        col = idx % cols
        plt.subplot(gs[row, col])
        x_vals, y_vals, epochs = [], [], []
        for design in w_designs:
            objs = design.get_plotting_objectives()
            x_vals.append(objs[0])
            y_vals.append(objs[1])
            epochs.append(design.epoch)
        plt.scatter(x_vals, y_vals, c=epochs, cmap='viridis')
        plt.title(f"Weight: {weight}")
        plt.xlabel("Stiffness")
        plt.ylabel("Volume Fraction")
        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.colorbar()
        if idx >= 8:
            break

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close('all')


def plot_pareto_designs(designs, pareto_plot_file, pareto_json_file):
    # Pareto designs
    x_vals_f, y_vals_f = [], []
    x_vals_i, y_vals_i = [], []
    for design in designs:
        objectives = design.get_plotting_objectives()
        if design.is_feasible is True:
            x_vals_f.append(objectives[0])
            y_vals_f.append(objectives[1])
        else:
            x_vals_i.append(objectives[0])
            y_vals_i.append(objectives[1])
    max_x, max_y = max(x_vals_f + x_vals_i), max(y_vals_f + y_vals_i)
    plt.figure(figsize=(8, 8))
    plt.xlabel('Stiffness')
    plt.ylabel('Volume Fraction')
    plt.scatter(x_vals_f, y_vals_f, c='b', label='Feasible Designs')
    plt.scatter(x_vals_i, y_vals_i, c='r', label='Infeasible Designs')
    plt.xlim([-0.1, max(1.1, max_x)])
    plt.ylim([-0.1, max(1.1, max_y)])
    plt.title('Pareto Front')
    plt.legend()
    plt.savefig(pareto_plot_file)
    plt.close('all')
    save_obj = [design.get_design_json() for design in designs if design.is_feasible]
    with open(pareto_json_file, 'w') as f:
        json.dump(save_obj, f, indent=4)


def plot_select_designs(problem, designs, save_dir):
    # Get all designs where pareto rank is 1
    pareto_designs = [design for design in designs if design.rank == 1]
    pareto_designs_strs = [design.get_design_str() for design in pareto_designs]

    # only retain unique designs
    pareto_designs_unique = []
    pareto_designs_unique_strs = []
    for idx, design in enumerate(pareto_designs):
        if design.get_design_str() not in pareto_designs_unique_strs:
            pareto_designs_unique.append(design)
            pareto_designs_unique_strs.append(design.get_design_str())
    pareto_designs = pareto_designs_unique

    pareto_designs_po = [design.get_plotting_objectives() for design in pareto_designs]
    pareto_designs_zip = list(zip(range(len(pareto_designs_po)), pareto_designs_po))
    pareto_designs_zip = sorted(pareto_designs_zip, key=lambda x: x[1][0])
    pareto_designs = [pareto_designs[idx] for idx, _ in pareto_designs_zip]

    for idx, pd in enumerate(pareto_designs):
        truss.rep.viz(problem, pd.vector, f_name=f'design_{idx}.png', base_dir=save_dir)


def plot_all_designs(designs, all_plot_file):
    x_vals_f, y_vals_f = [], []
    x_vals_i, y_vals_i = [], []
    for design in designs:
        objectives = design.get_plotting_objectives()
        if design.is_feasible is True:
            x_vals_f.append(objectives[0])
            y_vals_f.append(objectives[1])
        else:
            x_vals_i.append(objectives[0])
            y_vals_i.append(objectives[1])
    max_x, max_y = max(x_vals_f + x_vals_i), max(y_vals_f + y_vals_i)
    plt.figure(figsize=(8, 8))
    plt.xlabel('Stiffness')
    plt.ylabel('Volume Fraction')
    plt.scatter(x_vals_f, y_vals_f, c='b', label='Feasible Designs')
    plt.scatter(x_vals_i, y_vals_i, c='r', label='Infeasible Designs')
    plt.xlim([-0.1, max(1.1, max_x)])
    plt.ylim([-0.1, max(1.1, max_y)])
    plt.title('All Designs')
    plt.legend()
    plt.savefig(all_plot_file)
    plt.close('all')




