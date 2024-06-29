from copy import deepcopy
import numpy as np
import os
import tensorflow as tf
import time

from combench.core.algorithm import MultiTaskAlgorithm
from combench.nn.trussDecoderVL import get_models
import random

# ------- Run name
r_num = 0
save_name = 'cantilever-NxN-pretrain-50res-flex' + str(r_num)
load_name = 'cantilever-NxN-pretrain-50res-flex' + str(r_num)
metrics_num = 0
save_freq = 50
plot_freq = 20

NUM_PROCS = 1

# ------- Sampling parameters
num_problem_samples = 3  # 1
num_weight_samples = 3  # 1
global_mini_batch_size = num_problem_samples * num_weight_samples  # 4 * 4 * 4 = 64

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


problem_set_orig = truss.problems.Cantilever.enumerate_res({
    'x_range': 4,
    'y_range': 4,
    'x_res_range': [2, 4],
    'y_res_range': [2, 4],
    'radii': 0.2,
    'y_modulus': 210e9
})
random.seed(22)
problem_set = random.sample(problem_set_orig, 12)
val_problems = problem_set[:8]
train_problems = problem_set[8:]
nn_dict = {}
for p in train_problems:
    nn = len(p['nodes'])
    if nn not in nn_dict:
        nn_dict[nn] = 1
    else:
        nn_dict[nn] += 1
    truss.set_norms(p)
# p_rep = truss.rep.random_sample_1(train_problems[0])
# truss.rep.viz(train_problems[0], p_rep, f_name='varlen_truss_example')
max_problem_node_count = max([len(p['nodes']) for p in train_problems])
max_num_vars = int((max_problem_node_count * (max_problem_node_count-1)) / 2)


# -------- Set random seed for reproducibility
seed_num = 3
random.seed(seed_num)
tf.random.set_seed(seed_num)



class TrussPPOVL(MultiTaskAlgorithm):

    def __init__(self, problems, populations, max_nfe, actor_path=None, critic_path=None, run_name='ppo'):
        super().__init__(problems, populations, run_name, max_nfe)
        self.val_run = True
        self.designs = []
        self.nfe = 0
        self.unique_designs = []
        self.unique_designs_bitstr = set()
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.problems_num_bits = [truss.rep.get_num_bits(problem.problem_formulation) for problem in self.problems]

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
        self.num_actions = 3
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
                self.populations[self.last_updated_task].plot_hv(self.save_dir)
                self.populations[self.last_updated_task].plot_population(self.save_dir)
                self.plot_metrics(['return', 'c_loss', 'kl', 'entropy'], sn=metrics_num)
                # print('Time for plotting:', time.time() - curr_time)
            if self.curr_epoch % save_freq == 0:
                self.save_models(self.curr_epoch)
        self.eval_manager.shutdown()
        self.save_models()



    def get_cond_vars(self):
        problem_indices = list(range(len(self.problems)))


        # Construct conditioning tensor
        cond_vars = []
        weight_samples_all = []
        problem_samples_all = []
        population_samples_all = []
        p_encoding_samples_all = []
        p_encoding_mask_samples_all = []
        problem_samples_idx = []

        for x in range(global_mini_batch_size):
            weight = random.choice(self.objective_weights)
            sample_vars = [weight]
            cond_vars.append(sample_vars)
            weight_samples_all.append(weight)
            problem_sample_idx = random.choice(problem_indices)
            problem_samples_idx.append(problem_sample_idx)
            problem_samples_all.append(self.problems[problem_sample_idx])
            population_samples_all.append(self.populations[problem_sample_idx])
            p_enc, p_enc_mask = self.problems[problem_sample_idx].get_padded_encoding(max_problem_node_count)
            p_encoding_samples_all.append(p_enc)
            p_encoding_mask_samples_all.append(p_enc_mask)

        p_encoding_tensor = tf.convert_to_tensor(p_encoding_samples_all, dtype=tf.float32)
        p_encoding_mask_tensor = tf.convert_to_tensor(p_encoding_mask_samples_all, dtype=tf.float32)
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)

        return cond_vars_tensor, problem_samples_all, population_samples_all, problem_samples_idx, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor


    def pad_to_len(self, lists_to_pad, pad_len, pad_val=0):
        padded_lists = []
        for list_to_pad in lists_to_pad:
            padded_list = list_to_pad + [pad_val] * (pad_len - len(list_to_pad))
            padded_lists.append(padded_list)
        return padded_lists


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
        action_masks = [[] for x in range(self.mini_batch_size)]
        designs_in_progress = [True for x in range(self.mini_batch_size)]
        complete_designs = [False for x in range(self.mini_batch_size)]
        all_dists = []
        num_feasible = 0
        num_infeasible = 0

        # Get conditioning variables
        cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor = self.get_cond_vars()
        problems_num_vars_all = [self.problems_num_bits[idx] for idx in problem_indices_all]
        max_mini_batch_vars = max(problems_num_vars_all)



        # print('Problem Indices:', problem_indices_all)
        # print('PROBLEM ENCODING:', p_encoding_tensor.shape)


        for t in range(max_mini_batch_vars):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cond_vars_tensor, p_encoding_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                reward = 0.0
                curr_design = designs[idx]
                curr_design_num_vars = problems_num_vars_all[idx]
                m_action = int(deepcopy(act))
                if len(curr_design) <= curr_design_num_vars and designs_in_progress[idx] is True:  # The full design and end token have not been generated
                    if len(curr_design) == curr_design_num_vars:  # Token generated in end token position
                        observation_new[idx].append(0)
                        if m_action == 2:
                            reward = 0.1
                        else:
                            reward = -0.1
                        designs_in_progress[idx] = False
                    else:  # Token generated in design position (0 or 1)
                        observation_new[idx].append(m_action + 2)
                        designs[idx].append(m_action)
                        if m_action == 2:  # End token generated in a design decision position
                            reward = -0.1
                            designs_in_progress[idx] = False
                        elif len(designs[idx]) == curr_design_num_vars:  # Evaluate the design
                            reward = 55  # TODO: PLACEHOLDER REWARD FOR EVAL LATER
                        else:
                            reward = 0.0

                    all_actions[idx].append(deepcopy(act))
                    all_logprobs[idx].append(action_log_prob[idx])
                    action_masks[idx].append(1)
                    all_rewards[idx].append(reward)
                else:  # The design has been fully constructed
                    observation_new[idx].append(0)
                    action_masks[idx].append(0)
                    all_rewards[idx].append(reward)

            if True not in designs_in_progress:
                done = True
            else:
                done = False

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
                break
            else:
                observation = observation_new



        # Evaluate generated designs
        critic_observation_buffer = deepcopy(observation_new)
        print('SETTING CRITIC OBS BUFFER', critic_observation_buffer)

        design_idx_to_eval = []
        for idx, design in enumerate(designs):
            des_rewards = all_rewards[idx]
            if 55 in des_rewards:
                design_idx_to_eval.append(idx)

        eval_designs = []
        eval_problems = []
        eval_populations = []
        eval_designs_weights = []
        for idx, design in enumerate(designs):
            if idx in design_idx_to_eval:
                eval_designs.append(design)
                eval_problems.append(problem_samples_all[idx].problem_formulation)
                eval_populations.append(population_samples_all[idx])
                eval_designs_weights.append(weight_samples_all[idx])
        if len(eval_designs) > 0:

            objectives = self.eval_manager.evaluate(eval_problems, eval_designs)
            print('OBJECTIVES:', objectives)
            evals = self.calc_reward_batch(eval_designs, eval_designs_weights, objectives, eval_problems,
                                           eval_populations)
            for idx, (reward, design_obj) in enumerate(evals):
                design_idx = design_idx_to_eval[idx]
                replace_idx = all_rewards[design_idx].index(55)
                all_rewards[design_idx][replace_idx] = reward


        # Pad lists

        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # print(tf.convert_to_tensor(critic_observation_buffer).shape)
        # exit(0)

        # Max critic obs len
        max_critic_obs_len = max([len(obs) for obs in critic_observation_buffer])
        critic_observation_buffer = self.pad_to_len(critic_observation_buffer, max_critic_obs_len, pad_val=0)

        # --- SINGLE CRITIC PREDICTION --- #
        all_values = []
        value_t = self.sample_critic(critic_observation_buffer, cond_vars_tensor, p_encoding_tensor)  # (30, 31)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            actions = all_actions[idx]
            if 2 in actions:  # This means the end token was generated
                end_token_idx = actions.index(2)
                value_token_idx = end_token_idx + 1
            else:
                value_token_idx = len(designs[idx])
            # print('VALUE TOKEN IDX:', value_token_idx, len(value), len(all_rewards[idx]))
            last_reward = value[value_token_idx]
            all_rewards[idx][value_token_idx] = last_reward

            # Retrieve values based on action mask
            action_mask = action_masks[idx]
            all_values.append(value[:sum(action_mask)])



        for idx, d in enumerate(designs):
            print('\n\nREWARD:', all_rewards[idx])
            print('VALUES:', all_values[idx])
            print('DESIGN:', d)

        # -------------------------------------
        # Calculate Advantage and Return
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
        pred_probs = self.actor([observation_input, cross_input, p_encoding],
                                training=False)  # shape (batch, seq_len, 2)

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




if __name__ == '__main__':
    pop_size = 50
    ref_point = np.array([0, 1])

    actor_path, critic_path = None, None
    # actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_650')
    # critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_650')
    problems = [Model(deepcopy(problem), num_procs=NUM_PROCS) for problem in train_problems]
    pops = [Population(pop_size, ref_point, problem) for idx, problem in enumerate(problems)]
    ppo = TrussPPOVL(problems, pops, max_nfe, actor_path, critic_path, run_name=save_name)
    ppo.run()











