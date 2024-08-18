from copy import deepcopy
import numpy as np
import os
import tensorflow as tf
import time

import config
from combench.core.algorithm import MultiTaskAlgorithm
from combench.nn.trussDecoderUMD import get_models, max_nodes
from combench.algorithm import discounted_cumulative_sums
import random

# ------- Run name
save_name = 'truss-search-problem-val'
load_name = 'truss-search-problem-r10'
metrics_num = 0

# ------- Sampling parameters
num_problem_samples = 1  # 1
repeat_size = 8  # 3
global_mini_batch_size = num_problem_samples * repeat_size  # 12

# -------- Training Parameters
task_epochs = 800
max_nfe = 1e15
clip_ratio = 0.2
target_kl = 0.005
entropy_coef = 0.02

# -------- Problem
opt_dir = ['min']
use_constraints = False
from combench.models.truss.TrussModel import TrussModel as Model
from combench.models.truss.nsga2 import TrussPopulation as Population
from combench.models.truss.nsga2 import TrussDesign as Design
problem = {
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
num_cities = len(problem['cities'])

# -------- Set random seed for reproducibility
seed_num = 1
random.seed(seed_num)
tf.random.set_seed(seed_num)

class TspPPO(MultiTaskAlgorithm):

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
        self.num_vars = max_nodes
        self.actor, self.critic = get_models(self.actor_path, self.critic_path)

        # PPO Parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.mini_batch_size = global_mini_batch_size
        self.decision_start_token_id = 1
        self.num_actions = num_cities
        self.curr_epoch = 0

        # Objective Weights
        num_keys = 9
        self.objective_weights = list(np.linspace(0.05, 0.95, num_keys))

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
        # self.run_info['avg_dist'] = []
        # self.run_info['min_dist'] = []
        # self.run_info['problems'] = []

    def run(self):
        print('Running TspPPO')

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
            if self.curr_epoch % 10 == 0:
                self.populations[-1].plot_hv(self.save_dir)
                self.populations[-1].plot_population(self.save_dir)
                self.plot_metrics(['return', 'c_loss', 'kl', 'entropy'], sn=metrics_num)
                # print('Time for plotting:', time.time() - curr_time)

    def get_cond_vars(self):
        problem_indices = list(range(len(self.problems)))
        problem_samples_idx = random.sample(problem_indices, num_problem_samples)
        problem_samples = [self.problems[idx] for idx in problem_samples_idx]
        population_samples = [self.populations[idx] for idx in problem_samples_idx]



        # Construct conditioning tensor
        cond_vars = []
        problem_samples_all = []
        population_samples_all = []
        weight_samples_all = []
        for idx, problem in enumerate(problem_samples):
            weight_sample = random.sample(self.objective_weights, 1)[0]
            weight_samples_all.append(weight_sample)
            encoded_problem = problem.get_padded_encoding(max_nodes, rand=False)
            cond_vars.append(encoded_problem)
            problem_samples_all.append(problem)
            population_samples_all.append(population_samples[idx])

        problem_samples_all = [element for element in problem_samples_all for _ in range(repeat_size)]
        problem_indices_all = [idx for idx in problem_samples_idx for _ in range(repeat_size)]
        population_samples_all = [element for element in population_samples_all for _ in range(repeat_size)]
        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        cond_vars = [element for element in cond_vars for _ in range(repeat_size)]
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)  # (num_problem_samples, num_cities, 2)

        return cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, weight_samples_all

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
        cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, weight_samples_all = self.get_cond_vars()
        # print('Problem Indices:', problem_indices_all)


        for t in range(self.num_vars):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cond_vars_tensor)  # returns shape: (batch,) and (batch,)
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
                for idx, design in enumerate(designs):
                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, design_obj = self.calc_reward(
                        design_bitstr,
                        problem_samples_all[idx],
                        population_samples_all[idx]
                    )
                    if design_obj.is_feasible is True:
                        all_dists.append(design_obj.objectives[0])
                        num_feasible += 1
                    else:
                        num_infeasible += 1
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

        # print('Time sample actor:', time.time() - curr_time)
        # if len(all_dists) > 0:
        #     print('Average distance:', np.mean(all_dists))
        # else:
        #     print('No feasible designs found')

        # -------------------------------------
        # Sample Critic
        # -------------------------------------
        curr_time = time.time()


        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer, cond_vars_tensor)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

        # print('Time sample critic:', time.time() - curr_time)
        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------

        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
            values = np.array(value_t[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor

            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            all_returns[idx] = ret_tensor

        advantage_mean, advantage_std = (
            np.mean(all_advantages),
            np.std(all_advantages),
        )
        all_advantages = (all_advantages - advantage_mean) / advantage_std

        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(all_actions, dtype=tf.int32)
        logprob_tensor = tf.convert_to_tensor(all_logprobs, dtype=tf.float32)
        advantage_tensor = tf.convert_to_tensor(all_advantages, dtype=tf.float32)
        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        return_tensor = tf.expand_dims(return_tensor, axis=-1)

        # -------------------------------------
        # Train Actor
        # -------------------------------------
        curr_time = time.time()

        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_tensor,
                action_tensor,
                logprob_tensor,
                advantage_tensor,
                cond_vars_tensor
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break
        kl = kl.numpy()
        entr = entr.numpy()
        policy_loss = policy_loss.numpy()
        actor_loss = actor_loss.numpy()

        # print('Time train actor:', time.time() - curr_time)
        # -------------------------------------
        # Train Critic
        # -------------------------------------
        curr_time = time.time()

        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor,
                cond_vars_tensor,
            )
        value_loss = value_loss.numpy()

        # Collect min population distances
        min_pop_distances = [pop.get_min_distance() for pop in self.populations]
        min_pop_distances = [dist for dist in min_pop_distances if dist is not None]
        min_distance = np.mean(min_pop_distances)
        min_distance = self.populations[-1].get_min_distance()
        if min_distance is None:
            min_distance = 100

        # print('Time train critic:', time.time() - curr_time)

        self.run_info['return'].append(np.mean(all_total_rewards))
        self.run_info['c_loss'].append(value_loss)
        self.run_info['kl'].append(kl)
        self.run_info['entropy'].append(entr)
        # self.run_info['num_feasible'] = num_feasible
        # self.run_info['num_infeasible'] = num_infeasible
        # self.run_info['avg_dist'].append(np.mean(all_dists))
        # self.run_info['min_dist'].append(min_distance)
        # self.run_info['problems'] = problem_indices_all


        # Update nfe
        self.nfe = self.get_total_nfe()

    # -------------------------------------
    # Reward
    # -------------------------------------

    def calc_reward(self, tour_bitstr, problem, population):
        tour_bitlst = [int(bit) for bit in tour_bitstr]

        design = Design(tour_bitlst, problem)
        design = population.add_design(design)
        if design.is_feasible is True:
            reward = -design.objectives[0]
        else:
            reward = -5
            unique_cities_visited = set()
            for i in range(len(tour_bitlst)):
                city = tour_bitlst[i]
                if city not in unique_cities_visited:
                    reward += 0.1
                    unique_cities_visited.add(city)
                else:
                    break
        reward = reward * 0.1
        return reward, design

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, cross_obs):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.actor([observation_input, cross_input])

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs

    def sample_critic(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value = self.critic([observation_input, parent_input])  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer
    ):
        with tf.GradientTape() as tape:
            pred_probs = self.actor([observation_buffer, parent_buffer])  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)

            # Total loss
            loss = 0

            # PPO Surrogate Loss
            ratio = tf.exp(
                logprobability - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
            loss += policy_loss

            # Entropy Term
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        policy_grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        #  KL Divergence
        pred_probs = self.actor([observation_buffer, parent_buffer])
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.critic(
                [observation_buffer, parent_buffer])  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return value_loss




if __name__ == '__main__':
    pop_size = 50
    ref_point = np.array([1, 1])

    # Single-Task Training
    actor_path, critic_path = None, None
    actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_5000')
    critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_5000')
    problems = [Model(problem)]
    pops = [Population(pop_size, ref_point, problem)]
    ppo = TspPPO(problems, pops, max_nfe, actor_path, critic_path, run_name=save_name)
    ppo.run()

    # Multi-Task Training
    # problems = load_problem_set()
    # problems = [Model(problem) for problem in problems]
    # pops = [Population(pop_size, ref_point, problem) for problem in problems]
    # ppo = TspPPO(problems, pops, max_nfe, run_name=save_name)
    # ppo.run()







