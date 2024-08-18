import config
import os
import json
import time
from tqdm import tqdm


dd = os.path.join(config.datasets_dir, 'pinn')
batch_size = 8

import tensorflow as tf
from combench.models import truss
from combench.models.truss import train_problems

from combench.models.truss.TrussModel import TrussModel

class TrussPinnDG:

    def __init__(self, dataset_dir=dd):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')



    def prepare_datasets(self):

        problem = train_problems[0][1]
        print('PROBLEM:', problem)
        truss.set_norms(problem)
        td = [1 for x in range(truss.rep.get_num_bits(problem))]
        truss.rep.viz(problem, td, f_name='test.png', base_dir=config.plots_dir)


        print(problem)
        num_bits = truss.rep.get_num_bits(problem)
        print('Num Bits:', num_bits)

        model = TrussModel(problem)

        # Generate random designs
        n_dps = 15000
        designs = set()
        while len(designs) < n_dps:
            design = model.random_design()
            designs.add(tuple(design))
        print('Num Designs:', len(designs))

        # Process datapoints
        design_vectors = []
        design_stiffnesses = []
        design_forces = []
        design_displacements = []
        design_F = []
        design_u = []
        design_K = []

        for design in tqdm(designs, desc='Evaluating Designs'):
            # print('Evaluating:', design)
            curr_time = time.time()
            design = list(design)
            result, extra_info = truss.eval_stiffness(problem, design, normalize=False, verbose=True, verbose2=True)
            if not extra_info:
                print('Error: extra info not recognized', result, extra_info)
                continue
            extra_info = extra_info[0]
            if 'Error' in extra_info.keys():
                print('Error:', extra_info['Error'])
                continue

            # print('Result:', result, 'in', time.time() - curr_time, 's')
            # print('Extra Info:', extra_info)-
            # exit(0)
            stiff = result[0]

            u = extra_info['node_dists'][0]
            F = extra_info['node_forces'][0]

            K_full = extra_info['K_full']
            u_full = extra_info['u_full']
            F_full = extra_info['F_full']

            design_vectors.append(design)
            design_stiffnesses.append(stiff)

            design_forces.append(F)
            design_displacements.append(u)

            design_F.append(F_full)
            design_u.append(u_full)
            design_K.append(K_full)

        # Create splits
        n_train = int(0.8 * len(design_vectors))


        # Split into train and val
        train_design_vectors = design_vectors[:n_train]
        train_design_stiffnesses = design_stiffnesses[:n_train]
        train_design_forces = design_forces[:n_train]
        train_design_displacements = design_displacements[:n_train]
        train_design_F = design_F[:n_train]
        train_design_u = design_u[:n_train]
        train_design_K = design_K[:n_train]

        val_design_vectors = design_vectors[n_train:]
        val_design_stiffnesses = design_stiffnesses[n_train:]
        val_design_forces = design_forces[n_train:]
        val_design_displacements = design_displacements[n_train:]
        val_design_F = design_F[n_train:]
        val_design_u = design_u[n_train:]
        val_design_K = design_K[n_train:]

        # Save datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_design_vectors, train_design_stiffnesses, train_design_forces, train_design_displacements, train_design_F, train_design_u, train_design_K))
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.save(self.train_dataset_dir)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_design_vectors, val_design_stiffnesses, val_design_forces, val_design_displacements, val_design_F, val_design_u, val_design_K))
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.save(self.val_dataset_dir)

    def load_datasets(self, batch_size=None):
        train_path = self.train_dataset_dir
        val_path = self.val_dataset_dir
        if batch_size is not None:
            train_path = train_path + '_' + str(batch_size)
            val_path = val_path + '_' + str(batch_size)
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                self.rebatch_dataset(batch_size=batch_size)
        train_dataset = tf.data.Dataset.load(train_path)
        val_dataset = tf.data.Dataset.load(val_path)
        return train_dataset, val_dataset

    def rebatch_dataset(self, batch_size=8):
        train_dataset, val_dataset = self.load_datasets()
        train_dataset = train_dataset.rebatch(batch_size)
        val_dataset = val_dataset.rebatch(batch_size)
        train_dataset.save(self.train_dataset_dir + '_' + str(batch_size))
        val_dataset.save(self.val_dataset_dir + '_' + str(batch_size))



if __name__ == '__main__':
    dg = TrussPinnDG()
    # dg.prepare_datasets()
    train_dataset, val_dataset = dg.load_datasets()

    print('Train Dataset:', train_dataset)


    total_max = 0
    design_max = None
    total_min = 0
    for idx, dp in enumerate(train_dataset):
        designs, stiffness, forces, displacements, f_full, u_full, k_full = dp




        print('F full:', f_full.shape)
        print('U full:', u_full.shape)
        print('K full:', k_full.shape)

        print(k_full)
        exit(0)

        # result = tf.einsum('bij,bj->bi', k_full, u_full)

        U_expanded = tf.expand_dims(u_full, axis=-1)
        result = tf.matmul(k_full, U_expanded)
        result = tf.squeeze(result, axis=-1)

        print('Result:', result.shape, result)

        print('Forces:', f_full)

        f_full = f_full.numpy().tolist()
        result = result.numpy().tolist()

        for f, res in zip(f_full, result):
            print('\nF:', f)
            print('Res:', res)
            print('Diff:', [abs(f[i] - res[i]) for i in range(len(f))])
            # print('Max Diff:', max([abs(f[i] - res[i]) for i in range(len(f))]))
            print('---')


        exit(0)



        u_full = u_full.numpy().tolist()
        designs_np = designs.numpy().tolist()


        for idx, row in enumerate(u_full):
            u_full_max = max(row)
            u_full_min = min(row)


            if u_full_max > total_max:
                total_max = u_full_max
                design_max = designs_np[idx]


            if u_full_min < total_min:
                total_min = u_full_min

        # print(u_full)
        # if idx > 5:
        #     exit(0)
    print('Max / Min Displacement:', total_max, total_min)
    print('Design Max:', design_max)

