from copy import deepcopy
import numpy as np
import os
import tensorflow as tf
import time

from combench.core.algorithm import MultiTaskAlgorithm
from combench.nn.trussDecoder import get_models
from combench.algorithm import discounted_cumulative_sums
import random
import config

# ------- Run name
r_num = 5
save_name = 'ppo-mtl-cantilever-3x6-' + str(r_num)
load_name = 'ppo-mtl-cantilever-3x6-' + str(r_num)
metrics_num = 1
save_freq = 50
plot_freq = 20

NUM_PROCS = 32
DROPOUT = False

# ------- Sampling parameters
num_problem_samples = 6  # 1
num_weight_samples = 6  # 1
repeat_size = 1  # 3
global_mini_batch_size = num_problem_samples * repeat_size * num_weight_samples  # 4 * 4 * 4 = 64

# -------- Training Parameters
max_nfe = 1e15
clip_ratio = 0.2
target_kl = 0.005
entropy_coef = 0.04

# -------- Problem
opt_dir = ['max', 'min']
use_constraints = False
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager
from combench.models.truss.TrussModel import TrussModel as Model
from combench.models.truss.nsga2 import TrussPopulation as Population
from combench.models.truss.nsga2 import TrussDesign as Design
from combench.models.truss import train_problems, val_problems
for p in train_problems:
    truss.set_norms(p)

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
        self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            warmup_steps=100,
            initial_learning_rate=0.0,
            warmup_target=self.actor_learning_rate,
            decay_steps=1000,
            alpha=1.0
        )

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
        self.eval_manager = EvaluationProcessManager(NUM_PROCS)

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
                self.populations[0].plot_hv(self.save_dir)
                self.populations[self.last_updated_task].plot_population(self.save_dir)
                self.plot_metrics(['return', 'c_loss', 'kl', 'entropy'], sn=metrics_num)
                # print('Time for plotting:', time.time() - curr_time)
            if self.curr_epoch % save_freq == 0:
                self.save_models(self.curr_epoch)

        self.eval_manager.shutdown()
        self.save_models()

    def get_cond_vars(self):
        problem_indices = list(range(len(self.problems)))
        problem_samples_idx = random.sample(problem_indices, num_problem_samples)
        problem_samples = [self.problems[idx] for idx in problem_samples_idx]
        population_samples = [self.populations[idx] for idx in problem_samples_idx]
        weight_samples = random.sample(self.objective_weights, num_weight_samples)
        # weight_samples[0] = self.objective_weights[0]
        self.last_updated_task = problem_samples_idx[0]

        # Construct conditioning tensor
        cond_vars = []
        weight_samples_all = []
        problem_samples_all = []
        population_samples_all = []
        p_encoding_samples_all = []


        for x in range(global_mini_batch_size):
            weight = random.choice(self.objective_weights)
            sample_vars = [weight]
            cond_vars.append(sample_vars)
            weight_samples_all.append(weight)
            problem_sample_idx = random.choice(problem_indices)
            problem_samples_all.append(self.problems[problem_sample_idx])
            population_samples_all.append(self.populations[problem_sample_idx])
            p_encoding_samples_all.append(self.problems[problem_sample_idx].get_encoding())
            self.last_updated_task = problem_sample_idx

        # for weight in weight_samples:
        #     for idx, p in enumerate(problem_samples):
        #         sample_vars = [weight]
        #         cond_vars.append(sample_vars)
        #         weight_samples_all.append(weight)
        #         problem_samples_all.append(p)
        #         population_samples_all.append(population_samples[idx])
        #         p_encoding_samples_all.append(p.get_encoding())

        problem_samples_all = [element for element in problem_samples_all for _ in range(repeat_size)]
        problem_indices_all = [idx for idx in problem_samples_idx for _ in range(repeat_size)]
        population_samples_all = [element for element in population_samples_all for _ in range(repeat_size)]
        cond_vars = [element for element in cond_vars for _ in range(repeat_size)]
        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        p_encoding_samples_all = [element for element in p_encoding_samples_all for _ in range(repeat_size)]
        p_encoding_tensor = tf.convert_to_tensor(p_encoding_samples_all, dtype=tf.float32)
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)
        return cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all

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
        cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all = self.get_cond_vars()
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

                # TODO: Inline evals
                # for idx, design in enumerate(designs):
                #     # Record design
                #     design_bitstr = ''.join([str(bit) for bit in design])
                #     epoch_designs.append(design_bitstr)
                #
                #     # Evaluate design
                #     reward, design_obj = self.calc_reward(
                #         design_bitstr,
                #         problem_samples_all[idx],
                #         population_samples_all[idx],
                #         weight_samples_all[idx]
                #     )
                #     if design_obj.is_feasible is True:
                #         all_dists.append(design_obj.objectives[0])
                #         num_feasible += 1
                #     else:
                #         num_infeasible += 1
                #     all_rewards[idx].append(reward)
                #     all_total_rewards.append(reward)

                # TODO: Parallel evals
                # problem_to_design = {}  # Maps problems to the indices of their designs
                # for idx, design in enumerate(designs):
                #     problem = problem_samples_all[idx]
                #     problem_id = problem.model_id
                #     if problem_id not in problem_to_design:
                #         problem_to_design[problem_id] = [problem, []]
                #     problem_to_design[problem_id][1].append(idx)
                # problem_keys = list(problem_to_design.keys())
                # reconstructed_evals = [None for _ in range(len(designs))]
                # p_queues_all = []
                # for p_key in problem_keys:  # Batch evaluate each problem's designs
                #     designs_to_eval_idx = problem_to_design[p_key][1]
                #     designs_to_eval = [designs[idx] for idx in designs_to_eval_idx]
                #     problem = problem_to_design[p_key][0]
                #     p_queues = problem.evaluate_batch_async(designs_to_eval)
                #     p_queues_all.append(p_queues)
                # for idx, p_queues in enumerate(p_queues_all):
                #     evals = []
                #     for p_queue in p_queues:
                #         evals += p_queue.get()
                #     p_key = problem_keys[idx]
                #     designs_to_eval_idx = problem_to_design[p_key][1]
                #     for idx, eval in enumerate(evals):
                #         reconstructed_evals[designs_to_eval_idx[idx]] = eval
                # objectives = reconstructed_evals
                # evals = self.calc_reward_batch(designs, weight_samples_all, objectives, problem_samples_all, population_samples_all)
                # for idx, (reward, objs) in enumerate(evals):
                #     all_rewards[idx].append(reward)
                #     all_total_rewards.append(reward)

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
        value_t = self.sample_critic(critic_observation_buffer, cond_vars_tensor, p_encoding_tensor)  # (30, 31)
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
                cond_vars_tensor,
                p_encoding_tensor,
            )
            if abs(kl) > 1.5 * self.target_kl:
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
                p_encoding_tensor
            )
        value_loss = value_loss.numpy()

        # Collect min population distances
        # min_pop_distances = [pop.get_min_distance() for pop in self.populations]
        # min_pop_distances = [dist for dist in min_pop_distances if dist is not None]
        # min_distance = np.mean(min_pop_distances)
        # min_distance = self.populations[-1].get_min_distance()
        # if min_distance is None:
        #     min_distance = 100

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

    def calc_reward_batch(self, designs, weights, objectives, problems, populations):

        returns = []
        for idx, design in enumerate(designs):
            objs = list(objectives[idx])  # stiffness, volfrac
            weight = weights[idx]
            problem = problems[idx]
            population = populations[idx]

            # Validate Objectives
            if objs[0] == 0:  # If stiffness is 0, set volfrac to 1
                objs[1] = 1.0

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
            if reward > 100:
                print('Infeasible reward:', design, objs, reward)
                reward = 0.0

            # Create design and add to pop
            design_obj = Design(design, problem)
            design_obj.objectives = objs
            design_obj.is_feasible = True
            design_obj.weight = deepcopy(weight)
            design_obj.epoch = deepcopy(self.curr_epoch)
            design_obj = population.add_design(design_obj)


            # Add to returns
            returns.append([reward, design_obj])

        return returns


    def calc_reward(self, design_bitstr, problem, population, weight):

        design_bitlst = [int(bit) for bit in design_bitstr]
        design = Design(design_bitlst, problem)
        design = population.add_design(design)
        objs = design.evaluate()

        # If stiffness is 0, set volfrac to 1
        if objs[0] == 0:
            objs[1] = 1.0

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
            reward = design.feasibility_score * -0.01

        return reward, design





    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, cross_obs, p_encoding):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        # print('\n\n-----------------')
        # print('Observation:', observation_input.shape)
        # print('Cross Obs:', cross_obs.shape)
        # print('Inf Idx:', inf_idx)
        # print('P Encoding:', p_encoding.shape)
        return self._sample_actor(observation_input, cross_obs, inf_idx, p_encoding)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx, p_encoding):
        # print('sampling actor', inf_idx)
        pred_probs = self.actor([observation_input, cross_input, p_encoding], training=DROPOUT)  # shape (batch, seq_len, 2)

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

    def sample_critic(self, observation, parent_obs, p_encoding):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx, p_encoding)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx, p_encoding):
        t_value = self.critic([observation_input, parent_input, p_encoding])  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
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
            parent_buffer,
            p_encoding_buffer,
    ):
        with tf.GradientTape() as tape:
            pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer], training=DROPOUT)  # shape: (batch, seq_len, 2)
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
        pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer], training=DROPOUT)  # shape: (batch, seq_len, 2
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
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
            p_encoding_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.critic(
                [observation_buffer, parent_buffer, p_encoding_buffer])  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return value_loss




if __name__ == '__main__':
    pop_size = 50
    ref_point = np.array([0, 1])

    actor_path, critic_path = None, None
    # actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_650')
    # critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_650')
    problems = [Model(deepcopy(problem), num_procs=NUM_PROCS) for problem in train_problems]
    pops = [Population(pop_size, ref_point, problem) for idx, problem in enumerate(problems)]
    ppo = TrussPPO(problems, pops, max_nfe, actor_path, critic_path, run_name=save_name)
    ppo.run()







