from copy import deepcopy
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf
import time

import config
from combench.core.algorithm import MultiTaskAlgorithm
from combench.algorithm.nn.trussDecoder import get_models
from combench.algorithm import discounted_cumulative_sums
import random

# ------- Run name
r_num = 0
save_name = 'cantilever-NxN-pretrain-50res-flex' + str(r_num)
load_name = 'cantilever-NxN-pretrain-50res-flex' + str(r_num)
metrics_num = 0
save_freq = 50
plot_freq = 20

# ------- Sampling parameters
num_problem_samples = 8  # 1
num_weight_samples = 9  # 1
repeat_size = 1  # 3
global_mini_batch_size = num_problem_samples * repeat_size * num_weight_samples  # 4 * 4 * 4 = 64

# -------- Training Parameters
task_epochs = 800
max_nfe = 1e15
clip_ratio = 0.2
target_kl = 0.005
entropy_coef = 0.08

# -------- Problem
opt_dir = ['max', 'min']
use_constraints = False
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager
from combench.models.truss.TrussModel import TrussModel as Model
from combench.models.truss.nsga2 import TrussPopulation as Population
from combench.models.truss.nsga2 import TrussDesign as Design
from combench.models.truss.problems import train_problems, val_problems
for p in train_problems:
    truss.set_norms(p)
max_problem_node_count = max([len(p['nodes']) for p in train_problems])

# -------- Set random seed for reproducibility
seed_num = 3
random.seed(seed_num)
tf.random.set_seed(seed_num)



class TrussPPO(MultiTaskAlgorithm):

    def __init__(self, problems, populations, max_nfe, actor_path=None, critic_path=None, run_name='ppo'):
        super().__init__(problems, populations, run_name, max_nfe)
        self.val_run = True
        self.designs = []
        self.nfe = 0
        self.unique_designs = []
        self.unique_designs_bitstr = set()
        self.actor_path = actor_path
        self.critic_path = critic_path

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 40  # was 250
        self.train_critic_iterations = 40  # was 40

        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        # Get number of design variables
        self.num_vars = len(self.problems[0].random_design())
        self.actor, self.critic = get_models(self.actor_path, self.critic_path)

        # Objective Weights
        num_keys = 9
        self.objective_weights = list(np.linspace(0.05, 0.95, num_keys))

        # PPO Parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.mini_batch_size = global_mini_batch_size
        self.decision_start_token_id = 1
        self.num_actions = 2
        self.curr_epoch = 0
        self.last_updated_task = 0

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.save_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')

        # Update run info
        self.run_info['return'] = []
        self.run_info['c_loss'] = []
        self.run_info['kl'] = []
        self.run_info['entropy'] = []

        # Evaluation Manager
        self.eval_manager = EvaluationProcessManager(32)


    def save_models(self, epoch=None):
        if epoch:
            actor_save = self.actor_pretrain_save_path + '_' + str(epoch)
            critic_save = self.critic_pretrain_save_path + '_' + str(epoch)
        else:
            actor_save = self.actor_pretrain_save_path
            critic_save = self.critic_pretrain_save_path
        self.actor.save_weights(actor_save)
        self.critic.save_weights(critic_save)

    def run(self):
        print('Running TrussPPO')

        self.curr_epoch = 0
        while self.get_total_nfe() < self.max_nfe:
            curr_time = time.time()
            self.run_epoch()
            # print('Time for epoch:', time.time() - curr_time)
            curr_time = time.time()
            self.record()
            # print('Time for record:', time.time() - curr_time)
            curr_time = time.time()
            self.curr_epoch += 1
            if self.curr_epoch % plot_freq == 0:
                self.populations[self.last_updated_task].plot_hv(self.save_dir)
                self.populations[self.last_updated_task].plot_population(self.save_dir)
                self.plot_metrics(['return', 'c_loss', 'kl', 'entropy', 'avg_dist'], sn=metrics_num)
                # print('Time for plotting:', time.time() - curr_time)
            if self.curr_epoch % save_freq == 0:
                self.save_models(self.curr_epoch)
        self.save_models()



    def get_cond_vars(self):
        problem_indices = list(range(len(self.problems)))
        problem_samples_idx = random.sample(problem_indices, num_problem_samples)
        problem_samples = [self.problems[idx] for idx in problem_samples_idx]
        population_samples = [self.populations[idx] for idx in problem_samples_idx]
        weight_samples = random.sample(self.objective_weights, num_weight_samples)
        # weight_samples[0] = self.objective_weights[0]

        # Construct conditioning tensor
        cond_vars = []
        weight_samples_all = []
        problem_samples_all = []
        population_samples_all = []
        p_encoding_samples_all = []
        p_encoding_mask_samples_all = []

        for weight in weight_samples:
            for idx, p in enumerate(problem_samples):
                sample_vars = [weight]
                cond_vars.append(sample_vars)
                weight_samples_all.append(weight)
                problem_samples_all.append(p)
                population_samples_all.append(population_samples[idx])
                p_enc, p_enc_mask = p.get_padded_encoding(max_problem_node_count)
                p_encoding_samples_all.append(p_enc)
                p_encoding_mask_samples_all.append(p_enc_mask)

        problem_samples_all = [element for element in problem_samples_all for _ in range(repeat_size)]
        problem_indices_all = [idx for idx in problem_samples_idx for _ in range(repeat_size)]
        population_samples_all = [element for element in population_samples_all for _ in range(repeat_size)]
        cond_vars = [element for element in cond_vars for _ in range(repeat_size)]
        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        p_encoding_samples_all = [element for element in p_encoding_samples_all for _ in range(repeat_size)]
        p_encoding_mask_samples_all = [element for element in p_encoding_mask_samples_all for _ in range(repeat_size)]
        p_encoding_tensor = tf.convert_to_tensor(p_encoding_samples_all, dtype=tf.float32)
        p_encoding_mask_tensor = tf.convert_to_tensor(p_encoding_mask_samples_all, dtype=tf.float32)
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)

        return cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor



    def run_epoch(self):
        curr_time = time.time()

        new_designs = []

        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]
        all_dists = []
        num_feasible = 0
        num_infeasible = 0

        # Get conditioning variables
        cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor = self.get_cond_vars()
        # print('Problem Indices:', problem_indices_all)
        # print('PROBLEM ENCODING:', p_encoding_tensor.shape)


        for t in range(self.num_vars):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cond_vars_tensor, p_encoding_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.num_vars:
                done = True

                # TODO: Eval Manager evals
                e_problems, e_designs = [], []
                for idx, design in enumerate(designs):
                    pf = problem_samples_all[idx].problem_formulation
                    e_problems.append(pf)
                    e_designs.append(design)
                objectives = self.eval_manager.evaluate(e_problems, e_designs)
                evals = self.calc_reward_batch(designs, weight_samples_all, objectives, problem_samples_all, population_samples_all)
                for idx, (reward, design_obj) in enumerate(evals):
                    all_rewards[idx].append(reward)
                    all_total_rewards.append(reward)

            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    all_rewards[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new






