import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import json
import config
import matplotlib.pyplot as plt
import os
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import scipy.signal
from combench.algorithm.nn.qDecoder import get_models
from combench.core.algorithm import Algorithm

# ------- Run name
save_name = 'q-binary-search4'
max_nfe = 1000
plot_freq = 50

# ------- Sampling parameters
num_weight_samples = 9  # 4
repeat_size = 1  # 3
global_mini_batch_size = num_weight_samples * repeat_size

# -------- Training Parameters
epochs = 1000
value_learning_rate = 0.0001
use_warmup = False
update_batch_size_max = 16
update_batch_size_min = 16
update_batch_iterations = 10
update_target_network_freq = 5
replay_buffer_size = 10000
epsilon = 0.99  # was 0.99
epsilon_end = 0.01
decay_steps = 100 * config.num_vars

# -------- Problem
opt_dir = ['max', 'min']
use_constraints = False
from combench.models.assigning import problem1 as problem
from combench.models.assigning.GeneralizedAssigning import GeneralAssigning as Model
from combench.models.assigning.nsga2 import AssigningPop as Population
from combench.models.assigning.nsga2 import AssigningDesign as Design
from combench.ga.NSGA2 import BenchNSGA2

# opt_dir = ['max', 'max']
# use_constraints = False
# from combench.models.knapsack2 import problem1 as problem
# from combench.models.knapsack2.Knapsack2 import Knapsack2 as Model
# from combench.models.knapsack2.nsga2 import KPPopulation as Population
# from combench.models.knapsack2.nsga2 import KPDesign as Design
# from combench.ga.NSGA2 import BenchNSGA2

# -------- Set random seed
seed_num = 1
random.seed(seed_num)
tf.random.set_seed(seed_num)





class QLearning(Algorithm):

    def __init__(self, problem, population, max_nfe, q_network_load_path=None, run_name='q-learning'):
        super().__init__(problem, population, run_name, max_nfe)
        self.designs = []
        self.problem = problem
        self.q_network_load_path = q_network_load_path
        self.nfe = 0
        self.unique_designs = []
        self.unique_designs_bitstr = set()
        self.num_vars = len(self.problem.random_design())


        # Objective Weights
        self.objective_weights = list(np.linspace(0.00, 1.0, num_weight_samples))

        # Algorithm parameters
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.curr_epoch = 0
        self.num_actions = 2
        self.decision_start_token_id = 1
        self.steps_per_design = self.num_vars
        self.gamma = 0.99

        # Q Learning Parameters
        self.value_network = None
        self.target_value_network = None
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.step = 0
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps

        # Model steps
        self.q_network_steps = 0

        # Results
        self.plot_freq = plot_freq

        # Model / Optimizers
        self.cond_vars = 1  # Just objective weight for now
        self.value_network, self.target_value_network = get_models(self.num_vars, self.cond_vars, self.q_network_load_path)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_learning_rate)




    def run(self):
        print('Running Q Learning')

        self.curr_epoch = 0
        while self.population.nfe < self.max_nfe:
            self.gen_trajectories()
            for x in range(update_batch_iterations):
                self.update_q_network()
            if self.curr_epoch % update_target_network_freq == 0:
                self.update_target_network()
            self.record()
            self.curr_epoch += 1
            if self.curr_epoch % plot_freq == 0:
                self.population.plot_hv(self.save_dir)
                self.population.plot_population(self.save_dir)
            self.population.prune()

    def linear_decay(self):
        if self.step > self.decay_steps:
            return self.epsilon_end
        return self.epsilon - self.step * (self.epsilon - self.epsilon_end) / self.decay_steps

    def get_cross_obs(self):

        # Weight sampling
        weight_samples = []
        weight_samples = random.sample(self.objective_weights, num_weight_samples)

        # Construct conditioning tensor
        cross_obs_vars = []
        weight_samples_all = []
        task_samples_all = []
        for weight in weight_samples:
            sample_vars = [weight]
            cross_obs_vars.append(sample_vars)
            weight_samples_all.append(weight)
            task_samples_all.append(0)

        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        task_samples_all = [element for element in task_samples_all for _ in range(repeat_size)]
        cross_obs_vars = [element for element in cross_obs_vars for _ in range(repeat_size)]

        cross_obs_tensor = tf.convert_to_tensor(cross_obs_vars, dtype=tf.float32)

        return cross_obs_tensor, weight_samples_all, task_samples_all

    def gen_trajectories(self):

        # -------------------------------------
        # Sample Trajectories
        # -------------------------------------
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]

        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_actions_values = [[] for _ in range(self.mini_batch_size)]
        all_actions_values_full = [[] for _ in range(self.mini_batch_size)]

        all_rewards = [[] for _ in range(self.mini_batch_size)]

        all_values = [[] for _ in range(self.mini_batch_size)]

        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []

        # Get cross attention observation input
        cross_obs_tensor, weight_samples_all, task_samples_all = self.get_cross_obs()

        for t in range(self.steps_per_design):
            self.step += 1
            actions, actions_values, actions_values_full = self.sample_value_network(observation, cross_obs_tensor)

            for idx, act in enumerate(actions):
                all_actions[idx].append(deepcopy(act))
                all_actions_values[idx].append(deepcopy(actions_values[idx]))
                all_actions_values_full[idx].append(deepcopy(actions_values_full[idx]))
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation[idx].append(m_action + 2)

            # Assume all rewards are 0 for now
            for idx, act in enumerate(actions):
                all_rewards[idx].append(0.0)

        # Remove the last observation from each trajectory
        for idx, obs in enumerate(observation):
            observation[idx] = obs[:-1]

        # print('Design 0:', ''.join([str(x) for x in designs[0]]))
        # print('Design 0 Values:', all_actions_values[0])
        # print('Design 0 Full Values:', all_actions_values_full[0])

        # -------------------------------------
        # Evaluate Designs
        # -------------------------------------

        children = []
        all_rewards_flat = []
        all_constraints = []
        all_new_designs = []
        for idx, design in enumerate(designs):
            design_bitstr = ''.join([str(x) for x in design])
            epoch_designs.append(design_bitstr)
            reward, design_obj, is_new_design, des = self.calc_reward(
                design_bitstr,
                weight_samples_all[idx]
            )
            all_new_designs.append(is_new_design)
            children.append(des)
            all_rewards[idx][-1] = reward
            all_rewards_flat.append(reward)

        # -------------------------------------
        # Save to replay buffer
        # -------------------------------------

        memories = []
        for idx, design in enumerate(designs):
            if all_new_designs[idx] is False:
                continue

            buffer_entry = {
                'observation': observation[idx],
                'cross_obs': cross_obs_tensor[idx],
                'actions': all_actions[idx],
                'rewards': all_rewards[idx],

                'bitstr': epoch_designs[idx],
                'epoch': self.curr_epoch,
            }
            memories.append(buffer_entry)
            children[idx].memory = deepcopy(buffer_entry)
        self.replay_buffer.extend(memories)

        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = sorted(self.replay_buffer, key=lambda x: x['epoch'], reverse=True)
            self.replay_buffer = self.replay_buffer[:self.replay_buffer_size]

        # -------------------------------------
        # Update population
        # -------------------------------------

        return np.mean(all_rewards_flat), np.mean(all_constraints)




    def calc_reward(self, design_bitstr, weight):

        design_bitlst = [int(bit) for bit in design_bitstr]
        design = Design(design_bitlst, self.problem)
        curr_nfe = deepcopy(self.population.nfe)
        design = self.population.add_design(design)
        new_design = True if curr_nfe != self.population.nfe else False
        objs = design.evaluate()

        # Find weights
        w1 = weight
        w2 = 1.0 - weight

        # Calculate terms
        if opt_dir[0] == 'max':
            term1 = abs(objs[0]) * w1
        else:
            term1 = (1 - objs[0]) * w1

        if opt_dir[1] == 'max':
            term2 = abs(objs[1]) * w2
        else:
            term2 = (1 - objs[1]) * w2

        # Reward
        reward = term1 + term2

        # Implement constraints if necessary
        if use_constraints is True and design.is_feasible is False:
            reward += (design.feasibility_score * -0.1)

        return reward, objs, new_design, design

    def sample_value_network(self, observation, cross_obs_tensor):
        inf_idx = len(observation[0]) - 1
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        q_values = self._sample_value_network(observation, cross_obs_tensor)
        q_values_t = q_values[:, inf_idx, :]
        q_values = q_values_t.numpy().tolist()  # 2D list: (batch_element, action_values)

        epsilon = self.linear_decay()
        actions = []
        actions_values = []
        for sample_q_values in q_values:
            if random.random() < epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                action = np.argmax(sample_q_values)
            actions.append(action)
            actions_values.append(sample_q_values[action])

        return actions, actions_values, q_values

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def _sample_value_network(self, observation, cross_obs_tensor):
        q_values = self.value_network([observation, cross_obs_tensor])
        return q_values


    def update_q_network(self, use_pop=False):
        if len(self.replay_buffer) < update_batch_size_min:
            return

        # Population memories

        # -------------------------------------
        # Sample from replay buffer
        # -------------------------------------

        # Determine if using pop
        if use_pop is True:
            pop_memories = [design.memory for design in self.population.unique_designs]
            update_batch_size = min(update_batch_size_max, len(pop_memories))
            update_batch = random.sample(pop_memories, update_batch_size)
        else:
            update_batch_size = min(update_batch_size_max, len(self.replay_buffer))
            self.replay_buffer = sorted(self.replay_buffer, key=lambda x: x['epoch'], reverse=True)
            update_batch = random.sample(self.replay_buffer, update_batch_size)

        observation_batch = [x['observation'] for x in update_batch]
        cross_obs_batch = [x['cross_obs'] for x in update_batch]
        actions_batch = [x['actions'] for x in update_batch]
        rewards_batch = [x['rewards'] for x in update_batch]

        observation_tensor = tf.convert_to_tensor(observation_batch, dtype=tf.float32)
        cross_obs_tensor = tf.convert_to_tensor(cross_obs_batch, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions_batch, dtype=tf.int32)

        # -------------------------------------
        # Calculate Q Targets
        # -------------------------------------

        q_targets = self.sample_target_network(observation_tensor, cross_obs_tensor)
        q_targets = q_targets.numpy()
        # print('Target Network Q Values:', q_targets)

        # Calculate Target Values
        target_values = []
        for idx in range(len(update_batch)):
            rewards = np.array(rewards_batch[idx])
            targets = np.array(q_targets[idx])
            target_vals = rewards[:-1] + self.gamma * targets[1:]
            target_vals = target_vals.tolist()
            target_vals.append(rewards[-1])
            target_values.append(target_vals)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)

        # -------------------------------------
        # Train Q Network
        # -------------------------------------

        loss = self.train_q_network(observation_tensor, cross_obs_tensor, actions_tensor, target_values)
        loss = loss.numpy()

        epoch_info = {
            'loss': loss,
        }
        return epoch_info

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def sample_target_network(self, observation, cross_obs_tensor):
        q_values = self.target_value_network([observation, cross_obs_tensor], training=False)
        q_values = tf.reduce_max(q_values, axis=-1)
        return q_values

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_q_network(self, observation, cross_obs_tensor, actions, target_values):
        with tf.GradientTape() as tape:
            pred_q_values = self.value_network([observation, cross_obs_tensor])
            q_values = tf.reduce_sum(
                tf.one_hot(actions, self.num_actions) * pred_q_values, axis=-1
            )
            loss = tf.reduce_mean(tf.square(target_values - q_values))
            r_loss = tf.reduce_mean(tf.abs(target_values - q_values))

        gradients = tape.gradient(loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        return r_loss

    def update_target_network(self):
        self.target_value_network.load_target_weights(self.value_network)







if __name__ == '__main__':

    # Problem
    problem = Model(problem)

    # Population
    pop_size = 50
    ref_point = np.array([0, 1])
    pop = Population(pop_size, ref_point, problem)

    # PPO
    ppo = QLearning(problem, pop, max_nfe, run_name=save_name)
    ppo.run()






