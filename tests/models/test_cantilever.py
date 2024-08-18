import config
import unittest
import os
from copy import deepcopy
from combench.models.truss.nsga2 import TrussDesign
from combench.models.truss.TrussModel import TrussModel
from combench.models import truss

from combench.models.truss import train_problems

from combench.models.truss.problems.cantilever import get_problems


def get_ppath():
    ppath = os.path.join(config.plots_dir, 'test', 'cantilever')
    if not os.path.exists(ppath):
        os.makedirs(ppath)
    # find how many directories are in the ppath
    dir_count = len([name for name in os.listdir(ppath) if os.path.isdir(os.path.join(ppath, name))])
    dir_path = os.path.join(ppath, 'problem_space_' + str(dir_count))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


global_problem = {
    'nodes': [
        [0.0, 0.0], [0.0, 1.5], [0.0, 3.0],
        [1.2, 0.0], [1.2, 1.5], [1.2, 3.0],
        [2.4, 0.0], [2.4, 1.5], [2.4, 3.0],
        [3.5999999999999996, 0.0], [3.5999999999999996, 1.5], [3.5999999999999996, 3.0],
        [4.8, 0.0], [4.8, 1.5], [4.8, 3.0],
        [6.0, 0.0], [6.0, 1.5], [6.0, 3.0]
    ],
    'nodes_dof': [
        [0, 0], [0, 0], [0, 0],
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1]
    ],
    'member_radii': 0.2,
    'youngs_modulus': 210000000000.0,
    'load_conds': [
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, -1]]
    ]
}
load_node_idx = 17
def set_load(p, load):
    p['load_conds'][0][load_node_idx][1] = load
    return p
def set_radii(p, radii):
    p['member_radii'] = radii
    return p


class TestCantilever(unittest.TestCase):

    def test_problems(self):
        t_problems, v_problems, v_problems_out = get_problems()

        b_dir = get_ppath()
        for idx, vp_out in enumerate(v_problems_out):
            truss.set_norms(vp_out)
            design_rep = [1 for _ in range(truss.rep.get_num_bits(vp_out))]
            f_name = f'val_out_{idx}.png'
            truss.rep.viz(vp_out, design_rep, f_name=f_name, base_dir=b_dir)
        for idx, vp in enumerate(v_problems):
            truss.set_norms(vp)
            design_rep = [1 for _ in range(truss.rep.get_num_bits(vp))]
            f_name = f'val_{idx}.png'
            truss.rep.viz(vp, design_rep, f_name=f_name, base_dir=b_dir)
        for idx, tp in enumerate(t_problems):
            truss.set_norms(tp)
            design_rep = [1 for _ in range(truss.rep.get_num_bits(tp))]
            f_name = f'train_{idx}.png'
            truss.rep.viz(tp, design_rep, f_name=f_name, base_dir=b_dir)

        problem = t_problems[0]
        truss.set_norms(problem)
        model = TrussModel(problem)

        vector = [1 for _ in range(truss.rep.get_num_bits(problem))]
        results = model.evaluate(vector, normalize=False)
        print(results)

        truss.rep.viz(problem, vector, f_name='test.png', base_dir=get_ppath())



    def test_eval(self):
        print('Testing problems')

        # problem = train_problems[0][1]
        problem = deepcopy(global_problem)
        problem = set_load(problem, 1.0)
        problem = set_radii(problem, 0.1)

        truss.set_norms(problem)
        model = TrussModel(problem)

        vector = [1 for _ in range(truss.rep.get_num_bits(problem))]
        truss.rep.viz(problem, vector, f_name='test2.png', base_dir=config.plots_dir)

        max_designs = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                       1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                       1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        truss.rep.viz(problem, max_designs, f_name='max_displacementP.png', base_dir=config.plots_dir)


























if __name__ == '__main__':
    unittest.main()










