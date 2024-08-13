from combench.models.truss.studies.cantilever1 import train_problems, val_problems, oob_problems
from combench.models.truss.nsga2 import run_algorithm
from combench.models import truss
import config
import os

plots_dir = os.path.join(config.root_dir, 'combench', 'models', 'truss', 'studies', 'cantilever1', 'plots')


# This creates the baseline / evaluation data for each validation problem

def validate(problem, is_oob=False):
    x_res = problem['x_res']
    y_res = problem['y_res']

    truss.set_norms(problem)

    # 1. Create validation directory
    if is_oob:
        val_dir = os.path.join(plots_dir, f'oob_{y_res}x{x_res}')
    else:
        val_dir = os.path.join(plots_dir, f'validation_{y_res}x{x_res}')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 2. Fully connected design
    fc_design = [1 for x in range(truss.rep.get_num_bits(problem))]
    truss.rep.viz(problem, fc_design, base_dir=val_dir, f_name='fc_design.png')

    # 3. Run GA
    nfe = 10000
    pop_size = 30
    run_algorithm(problem, val_dir, nfe=nfe, pop_size=pop_size)


if __name__ == '__main__':

    for idx, val_problem in enumerate(val_problems):
        print(f'Validation Problem {idx+1}/{len(val_problems)}')
        validate(val_problem)

    for idx, oob_problem in enumerate(oob_problems):
        validate(oob_problem, is_oob=True)
















