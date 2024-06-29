from copy import deepcopy
import numpy as np
import os
import tensorflow as tf
from combench.core.algorithm import Algorithm
from combench.nn.trussDecoderSP import get_models
from combench.algorithm import discounted_cumulative_sums
import random
import time
import config

# ------- Run name
run_num = 3
load_name = 'ppoSP-cantilever-3x4'
save_name = 'ppoSP-cantilever-3x4-' + str(run_num)
metrics_num = 0
plot_freq = 30

# ------- Sampling parameters
num_weight_samples = 4  # 4
repeat_size = 3  # 3
global_mini_batch_size = num_weight_samples * repeat_size

# -------- Training Parameters
task_epochs = 800
max_nfe = 10000
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
from combench.models.truss import train_problems, val_problems
import multiprocessing as mp
mp.set_start_method('fork', force=True)


# -------- Set random seed for reproducibility
seed_num = 2
random.seed(seed_num)
tf.random.set_seed(seed_num)

class UnconstrainedPPO(Algorithm):

    def __init__(self, problem, population, max_nfe, actor_path=None, critic_path=None, run_name='ppo'):
        super().__init__(problem, population, run_name, max_nfe)
        self.designs = []
        self.nfe = 0
        self.unique_designs = []
        self.unique_designs_bitstr = set()
        self.actor_path = actor_path
        self.critic_path = critic_path

        # Objective Weights
        num_keys = 9
        self.objective_weights = list(np.linspace(0.05, 0.95, num_keys))
        # self.objective_weights = list(np.linspace(0.05, 0.75, num_keys))

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40

        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        # Get number of design variables
        self.num_vars = truss.rep.get_num_bits(self.problem.problem_formulation)
        self.cond_vars = 1  # Just objective weight for now

        self.actor, self.critic = get_models(self.actor_path, self.critic_path)

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

        # Update run info
        self.run_info['return'] = []
        self.run_info['c_loss'] = []
        self.run_info['kl'] = []
        self.run_info['entropy'] = []
        # self.run_info['overlaps'] = []

        # Evaluation Manager
        self.eval_manager = EvaluationProcessManager(12)

    def run(self):
        print('Running PPO')

        self.curr_epoch = 0
        while self.population.nfe < self.max_nfe:
            self.run_epoch()
            self.record()
            self.curr_epoch += 1
            if self.curr_epoch % plot_freq == 0:
                self.population.plot_hv(self.save_dir)
                self.population.plot_population(self.save_dir)
                self.plot_metrics(['return', 'c_loss', 'kl', 'entropy'], sn=metrics_num)

    def sample_normal_discrete(self, mean, std_dev):
        sample = np.random.normal(mean, std_dev)
        closest_value = min(self.objective_weights, key=lambda x: abs(x - sample))
        return closest_value

    def check_overlap(self, design_before, design_after):
        if 1 not in design_after or 1 not in design_before or design_after[-1] == 0:
            return False

        num_overlaps_bf = self.count_overlap(design_before)
        num_overlaps_af = self.count_overlap(design_after)
        if num_overlaps_bf != num_overlaps_af:
            return True
        else:
            return False

    def count_overlap(self, design):
        d = deepcopy(design)
        d += [0] * (config.num_vars - len(d))
        d_no = truss.rep.remove_overlapping_members(self.problem.problem_formulation, d)
        num_overlaps = abs(sum(d) - sum(d_no))
        return num_overlaps


    def get_cond_vars(self):
        weight_samples = random.sample(self.objective_weights, num_weight_samples)
        # weight_samples = [self.sample_normal_discrete(0.5, 0.2) for _ in range(num_weight_samples)]

        # Construct conditioning tensor
        cond_vars = []
        weight_samples_all = []
        p_encoding_samples_all = []
        for weight in weight_samples:
            sample_vars = [weight]
            cond_vars.append(sample_vars)
            weight_samples_all.append(weight)
            p_encoding_samples_all.append(self.problem.get_encoding())

        weight_samples_all = [element for element in weight_samples_all for _ in range(repeat_size)]
        cond_vars = [element for element in cond_vars for _ in range(repeat_size)]
        p_encoding_samples_all = [element for element in p_encoding_samples_all for _ in range(repeat_size)]
        p_encoding_tensor = tf.convert_to_tensor(p_encoding_samples_all, dtype=tf.float32)
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)
        return cond_vars_tensor, weight_samples_all, p_encoding_tensor

    def run_epoch(self):
        new_designs = []

        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Get conditioning variables
        cond_vars_tensor, weight_samples_all, p_encoding_tensor = self.get_cond_vars()

        # Sample actions
        sample_start_time = time.time()
        for t in range(self.num_vars):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cond_vars_tensor, p_encoding_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            designs_before = deepcopy(designs)
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
                start_eval_time = time.time()

                # TODO: Inline evals
                # for idx, design in enumerate(designs):
                #     # Record design
                #     design_bitstr = ''.join([str(bit) for bit in design])
                #     epoch_designs.append(design_bitstr)
                #
                #     # Evaluate design
                #     reward, objs = self.calc_reward(
                #         design_bitstr,
                #         weight_samples_all[idx]
                #     )
                #     all_rewards[idx].append(reward)
                #     all_total_rewards.append(reward)

                # TODO: Parallel evals
                for idx, design in enumerate(designs):
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)
                evals = self.calc_reward_batch(designs, weight_samples_all)
                for idx, (reward, objs) in enumerate(evals):
                    all_rewards[idx].append(reward)
                    all_total_rewards.append(reward)

                eval_time = time.time() - start_eval_time

            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    # has_overlap = self.check_overlap(designs_before[idx], designs[idx])
                    # if has_overlap is True:
                    #     all_rewards[idx].append(-0.01)
                    # else:
                    #     all_rewards[idx].append(reward)
                    all_rewards[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new

        sample_time = (time.time() - sample_start_time) - eval_time
        # print(f'Epoch {self.curr_epoch} - Sample Time: {sample_time:.2f}s - Eval Time: {eval_time:.2f}s')
        # overlaps = np.mean([self.count_overlap(design) for design in designs])


        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer, cond_vars_tensor, p_encoding_tensor)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

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

        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_tensor,
                action_tensor,
                logprob_tensor,
                advantage_tensor,
                cond_vars_tensor, p_encoding_tensor
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break
        kl = kl.numpy()
        entr = entr.numpy()
        policy_loss = policy_loss.numpy()
        actor_loss = actor_loss.numpy()

        # -------------------------------------
        # Train Critic
        # -------------------------------------

        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor,
                cond_vars_tensor, p_encoding_tensor
            )
        value_loss = value_loss.numpy()

        self.run_info['return'].append(np.mean(all_total_rewards))
        self.run_info['c_loss'].append(value_loss)
        self.run_info['kl'].append(kl)
        self.run_info['entropy'].append(entr)
        # self.run_info['overlaps'].append(overlaps)

        # Update nfe
        self.nfe = deepcopy(self.population.nfe)

    # -------------------------------------
    # Reward
    # -------------------------------------

    def calc_reward_batch(self, designs, weights):
        e_problems = [self.problem.problem_formulation for _ in range(len(designs))]
        e_designs = designs
        objectives = self.eval_manager.evaluate(e_problems, e_designs)
        # objectives = self.eval_manager.evaluate_store(e_problems, e_designs)

        returns = []
        for idx, design in enumerate(designs):
            objs = list(objectives[idx])
            weight = weights[idx]

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
            # Multiplier for term2 (volume fraction) to compete with stiffness
            # term2 = term2 * 2.0

            # Reward
            reward = term1 + term2

            # Create design and add to pop
            design_obj = Design(design, self.problem)
            design_obj.objectives = objs
            design_obj.is_feasible = True
            design_obj.weight = deepcopy(weight)
            design_obj.epoch = deepcopy(self.curr_epoch)
            design_obj = self.population.add_design(design_obj)


            # Add to returns
            returns.append([reward, design_obj])

        return returns

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, cross_obs, p_encoding):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx, p_encoding)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx, p_encoding):
        # print('sampling actor', inf_idx)
        pred_probs = self.actor([observation_input, cross_input, p_encoding])  # shape (batch, seq_len, 2)

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
            pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer])  # shape: (batch, seq_len, 2)
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
        pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer])
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

    problem_dict = val_problems[1]
    truss.set_norms(problem_dict)


    # Problem
    problem = Model(problem_dict)

    # Population
    pop_size = 50
    ref_point = np.array([0, 1])
    pop = Population(pop_size, ref_point, problem)

    # PPO
    actor_path, critic_path = None, None
    # actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_8000')
    # critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_8000')
    ppo = UnconstrainedPPO(problem, pop, max_nfe, actor_path=actor_path, critic_path=critic_path, run_name=save_name)
    ppo.run()





