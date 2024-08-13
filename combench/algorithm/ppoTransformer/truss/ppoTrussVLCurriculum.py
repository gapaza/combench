# """
#    _____                _            _                   _                           _
#   / ____|              (_)          | |                 | |                         (_)
#  | |    _   _ _ __ _ __ _  ___ _   _| |_   _ _ __ ___   | |     ___  __ _ _ __ _ __  _ _ __   __ _
#  | |   | | | | '__| '__| |/ __| | | | | | | | '_ ` _ \  | |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
#  | |___| |_| | |  | |  | | (__| |_| | | |_| | | | | | | | |___|  __/ (_| | |  | | | | | | | | (_| |
#   \_____\__,_|_|  |_|  |_|\___|\__,_|_|\__,_|_| |_| |_| |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
#                                                                                               __/ |
#                                                                                              |___/
# """


from copy import deepcopy
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import time
from combench.nn import beam_search

from combench.algorithm import discounted_cumulative_sums
from combench.core.algorithm import MultiTaskAlgorithm

# from combench.nn.trussDecoderVL import get_models
from combench.nn.trussDecoderVL2 import get_models

import random

# ------- Run name
save_start_epoch = 9000
load_name = 'cantilever1-mtl-' + str(21)
save_name = 'cantilever1-mtl-' + str(21)
metrics_num = 1
save_freq = 100
plot_freq = 100
val_freq = 100

NUM_PROCS = 32
REF_POINT = np.array([0, 1])
TRAIN_CALL = False
RAND_NODE_ENCODING = False

# ------- Sampling parameters
global_mini_batch_size = 128

# -------- Training Parameters
task_epochs = 800
max_nfe = 1e15
clip_ratio = 0.2
target_kl = 0.005  # was 0.005
entropy_coef = 0.02

# ------------------------------------------------
# Problem
# ------------------------------------------------

opt_dir = ['max', 'min']
use_constraints = False
from combench.models import truss
from combench.models.truss.eval_process import EvaluationProcessManager
from combench.models.truss.TrussModel import TrussModel as Model
from combench.models.truss.nsga2 import TrussPopulation as Population
from combench.models.truss.nsga2 import TrussDesign as Design

from combench.models.truss.studies.cantilever1 import train_problems, val_problems, oob_problems
val_problems_out = oob_problems

nn_dict = {}
for tp in train_problems:
    num_nodes = len(tp['nodes'])
    if num_nodes not in nn_dict:
        nn_dict[num_nodes] = []
    nn_dict[num_nodes].append(tp)
print('Number of nodes:', nn_dict.keys())
nn_list = list(nn_dict.keys())
nn_idx = [
    0,  # 4
    1,  # 6
    2,  # 8
    3,  # 9
    # 4,  # 12
    # 5,  # 16
]
train_problems = []
for nn in nn_idx:
    train_problems += nn_dict[nn_list[nn]]

print('Number of train problems:', len(train_problems))

# ------------------------------------------------
# Problem Assessment
# ------------------------------------------------

# for idx, vp in enumerate(val_problems):
#     p_rep = truss.rep.random_sample_1(vp)
#     truss.rep.viz(vp, p_rep, f_name='val_problem_' + str(idx) + '.png')
max_problem_node_count = max([len(p['nodes']) for p in train_problems])
max_num_vars = int((max_problem_node_count * (max_problem_node_count-1)) / 2)
train_problems_node_count = [len(p['nodes']) for p in train_problems]


# -------- Set random seed for reproducibility
seed_num = 1
random.seed(seed_num)
tf.random.set_seed(seed_num)


class TrussPPOVL(MultiTaskAlgorithm):

    def __init__(self, problems, populations, max_nfe, actor_path=None, critic_path=None, run_name='ppo', val_models=None, val_models_out=None):
        super().__init__(problems, populations, run_name, max_nfe)
        self.val_run = True
        self.designs = []
        self.nfe = 0
        self.unique_designs = []
        self.unique_designs_bitstr = set()
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.problems_num_bits = [truss.rep.get_num_bits(problem.problem_formulation) for problem in self.problems]
        self.val_models = val_models
        self.val_models_out = val_models_out

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0001  # 0.0001
        self.train_actor_iterations = 40  # was 250
        self.train_critic_iterations = 40  # was 40

        # Scheduler
        self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=1000,
            warmup_steps=10,
            warmup_target=self.actor_learning_rate,
            alpha=1.0,
        )

        # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
        self.critic_loss = tf.keras.losses.MeanSquaredError()

        # Get number of design variables
        self.num_vars = len(self.problems[0].random_design())
        self.actor, self.critic = get_models(self.actor_path, self.critic_path)

        # Objective Weights
        num_keys = 64
        self.objective_weights = list(np.linspace(0.00, 1.00, num_keys))

        # PPO Parameters
        self.gamma = 0.99
        self.lam = 0.95  # 0.95
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
        self.run_info['return_avg'] = []
        self.run_info['c_loss'] = []
        self.run_info['kl'] = []
        self.run_info['entropy'] = []
        self.run_info['epoch_evals'] = []
        self.run_info['val_hv'] = []
        self.run_info['pareto_sen'] = [
            [] for _ in range(len(self.val_models))
        ]


        # Evaluation Manager
        self.eval_manager = EvaluationProcessManager(NUM_PROCS)

    def save_models(self, epoch=None):
        if epoch:
            actor_save = self.actor_pretrain_save_path + '_' + str(epoch + save_start_epoch)
            critic_save = self.critic_pretrain_save_path + '_' + str(epoch + save_start_epoch)
        else:
            actor_save = self.actor_pretrain_save_path
            critic_save = self.critic_pretrain_save_path
        self.actor.save_weights(actor_save)
        self.critic.save_weights(critic_save)

    def run(self):
        print('Running TrussPPO')
        # self.run_val_epoch(self.val_models, 'val_hv')

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
            if self.curr_epoch % val_freq == 0:
                self.run_val_epoch(self.val_models, 'val_hv')
            if self.curr_epoch % plot_freq == 0:
                self.populations[self.last_updated_task].plot_hv(self.save_dir)
                self.populations[self.last_updated_task].plot_population(self.save_dir)
                self.plot_metrics(['return', 'return_avg', 'c_loss', 'kl', 'entropy', 'val_hv', 'pareto_sen'], sn=metrics_num)
                # print('Time for plotting:', time.time() - curr_time)
            if self.curr_epoch % save_freq == 0:
                self.save_models(self.curr_epoch)

        self.eval_manager.shutdown()
        self.save_models()

    def get_cond_vars(self):
        problem_indices = list(range(len(self.problems)))

        # problem_indices_all = list(range(len(self.problems)))
        # problem_lens = self.problems_num_bits
        # total_bits = sum(problem_lens)
        # problem_probs = [pl / total_bits for pl in problem_lens]


        # problem_lens_unique = list(set(problem_lens))
        # sample_problem_len = random.choice(problem_lens_unique)
        # problem_indices = []
        # for pi, pl in zip(problem_indices_all, problem_lens):
        #     if pl == sample_problem_len:
        #         problem_indices.append(pi)



        # Construct conditioning tensor
        cond_vars = []
        weight_samples_all = []
        problem_samples_all = []
        population_samples_all = []
        p_encoding_samples_all = []
        p_encoding_mask_samples_all = []
        problem_samples_idx = []
        for x in range(global_mini_batch_size):

            # Sample weight
            weight = random.choice(self.objective_weights)
            sample_vars = [weight]
            cond_vars.append(sample_vars)
            weight_samples_all.append(weight)

            # Sample problem
            problem_sample_idx = random.choice(problem_indices)
            # problem_sample_idx = np.random.choice(problem_indices_all, p=problem_probs)
            problem_samples_idx.append(problem_sample_idx)
            problem_samples_all.append(self.problems[problem_sample_idx])
            population_samples_all.append(self.populations[problem_sample_idx])
            p_enc, p_enc_mask = self.problems[problem_sample_idx].get_padded_encoding(max_problem_node_count, rand=RAND_NODE_ENCODING)
            p_encoding_samples_all.append(p_enc)
            p_encoding_mask_samples_all.append(p_enc_mask)

        # print('PROBLEM SAMPLES:', problem_samples_idx)
        p_encoding_tensor = tf.convert_to_tensor(p_encoding_samples_all, dtype=tf.float32)
        p_encoding_mask_tensor = tf.convert_to_tensor(p_encoding_mask_samples_all, dtype=tf.int32)
        cond_vars_tensor = tf.convert_to_tensor(cond_vars, dtype=tf.float32)

        return cond_vars_tensor, problem_samples_all, population_samples_all, problem_samples_idx, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor

    def pad_to_len(self, lists_to_pad, pad_len, pad_val=0):
        padded_lists = []
        for list_to_pad in lists_to_pad:
            if len(list_to_pad) < pad_len:
                padded_list = list(list_to_pad) + [pad_val] * (pad_len - len(list_to_pad))
            elif len(list_to_pad) > pad_len:
                padded_list = list_to_pad[:pad_len]
            else:
                padded_list = list_to_pad
            padded_lists.append(padded_list)
        return padded_lists

    def run_epoch(self):
        curr_time = time.time()
        epoch_designs = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]
        action_masks = [[] for x in range(self.mini_batch_size)]
        designs_in_progress = [True for x in range(self.mini_batch_size)]
        complete_designs = [False for x in range(self.mini_batch_size)]

        # Get conditioning variables
        cond_vars_tensor, problem_samples_all, population_samples_all, problem_indices_all, p_encoding_tensor, weight_samples_all, p_encoding_mask_tensor = self.get_cond_vars()
        problems_num_vars_all = [self.problems_num_bits[idx] for idx in problem_indices_all]
        # print('Bit Lengths:', list(set(problems_num_vars_all)))
        max_mini_batch_vars = max(problems_num_vars_all)

        for t in range(max_mini_batch_vars):
            action_log_prob, action, all_action_probs = self.sample_actor(observation, cond_vars_tensor, p_encoding_tensor, p_encoding_mask_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()
            observation_new = deepcopy(observation)

            # Iterate over actions
            for idx, act in enumerate(action.numpy()):
                reward = 0.0  # Default reward
                m_action = int(deepcopy(act))  # 0, 1, or 2 for end token
                curr_design = designs[idx]
                curr_design_num_vars = problems_num_vars_all[idx]


                in_progress = designs_in_progress[idx]
                if in_progress is False:
                    # ------------------------------
                    # Case 1: Design is complete
                    # ------------------------------
                    observation_new[idx].append(0)   # Pad input with zero
                    action_masks[idx].append(0)      # No action
                    reward = 0.0                     # No reward
                else:
                    observation_new[idx].append(m_action + 2)
                    designs[idx].append(m_action)
                    if len(designs[idx]) == curr_design_num_vars:  # Evaluate the design
                        reward = 55  # TODO: PLACEHOLDER REWARD FOR EVAL LATER
                        complete_designs[idx] = True
                        designs_in_progress[idx] = False
                    else:
                        reward = 0.0

                    all_actions[idx].append(deepcopy(act))
                    all_logprobs[idx].append(action_log_prob[idx])
                    action_masks[idx].append(1)
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


        # # Calculate pred len error
        # pred_len_errors = []
        # for idx, design in enumerate(designs):
        #     curr_design_num_vars = problems_num_vars_all[idx]
        #     diff = abs(len(design) - curr_design_num_vars)
        #     pred_len_errors.append(diff)
        # avg_len_error = np.mean(pred_len_errors)


        # -------------------------------------
        # Evaluate Designs
        # -------------------------------------

        critic_observation_buffer = deepcopy(observation_new)

        design_idx_to_eval = []
        for idx, design in enumerate(designs):
            des_rewards = all_rewards[idx]
            if 55 in des_rewards:
                design_idx_to_eval.append(idx)

        eval_designs = []
        eval_problems = []
        eval_problems_obj = []
        eval_populations = []
        eval_designs_weights = []
        for idx, design in enumerate(designs):
            if idx in design_idx_to_eval:
                eval_designs.append(design)
                eval_problems.append(problem_samples_all[idx].problem_formulation)
                eval_problems_obj.append(problem_samples_all[idx])
                eval_populations.append(population_samples_all[idx])
                eval_designs_weights.append(weight_samples_all[idx])
        if len(eval_designs) > 0:
            # print('Evaluating', len(eval_designs), 'designs')
            objectives = self.eval_manager.evaluate(eval_problems, eval_designs)
            evals = self.calc_reward_batch(eval_designs, eval_designs_weights, objectives, eval_problems_obj, eval_populations)
            eval_shown = False
            for idx, (reward, design_obj) in enumerate(evals):
                design_idx = design_idx_to_eval[idx]
                if eval_shown is False:
                    # print('\n----- Design:', all_actions[design_idx])
                    # print('- Reward-Pre:', design_idx, reward, all_rewards[design_idx])
                    eval_shown = True

                replace_idx = all_rewards[design_idx].index(55)
                all_rewards[design_idx][replace_idx] = reward

        all_rewards_flat = []
        eval_shown = False
        for idx, rewards in enumerate(all_rewards):
            all_rewards_flat.append(sum(rewards))
            if idx in design_idx_to_eval and eval_shown is False:
                # print('-- Evaluated:', idx, rewards)
                eval_shown = True


        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # Max critic obs len
        max_critic_obs_len = max([len(obs) for obs in critic_observation_buffer])
        critic_observation_buffer = self.pad_to_len(critic_observation_buffer, max_critic_obs_len, pad_val=0)

        # --- SINGLE CRITIC PREDICTION --- #
        all_values = []
        value_t = self.sample_critic(critic_observation_buffer, cond_vars_tensor, p_encoding_tensor, p_encoding_mask_tensor)  # (30, 31)
        # print('VALUE T:', value_t.shape)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            actions = all_actions[idx]
            actions_taken = len(actions)
            # print('ACTIONS TAKEN', actions_taken)
            last_reward = value[actions_taken]
            all_rewards[idx].append(last_reward)
            values_clip = value[:actions_taken+1]
            all_values.append(values_clip)


        all_values_mask = []
        for all_v in all_values:
            all_v_mask = [1 for _ in range(len(all_v))]
            all_values_mask.append(all_v_mask)
        all_values_mask = self.pad_to_len(all_values_mask, max_critic_obs_len, pad_val=0)


        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------


        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_advantages_flat = []
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):

            rewards = np.array(all_rewards[idx])
            values = np.array(all_values[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor
            all_advantages_flat.extend(deepcopy(adv_tensor))

            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            all_returns[idx] = ret_tensor

        adv_mean = np.mean(all_advantages_flat)
        adv_std = np.std(all_advantages_flat)

        all_advantages_norm = []
        for idx, traj_advantages in enumerate(all_advantages):
            traj_advantages_norm = []
            for idx2, adv in enumerate(traj_advantages):
                traj_advantages_norm.append((adv - adv_mean) / adv_std)
            all_advantages_norm.append(traj_advantages_norm)
        all_advantages = all_advantages_norm


        for idx, design in enumerate(designs):
            prob = problem_samples_all[idx].problem_formulation
            p_nn = truss.rep.get_num_bits(prob)
            # print('\n\n------------------- DESIGN:', idx, '(', len(design), '/', p_nn, ')', complete_designs[idx])
            # print('-------- Rep:', design)
            # print('---- Actions:', all_actions[idx])
            # print('---- Rewards:', all_rewards[idx])
            # print('---- LgProbs:', all_logprobs[idx])
            # print('----- Values:', all_values[idx])
            # print('- Advantages:', all_advantages[idx])
            # print('---- Returns:', all_returns[idx])
            # print('-- CriticObs:', critic_observation_buffer[idx])
            # print('--  ActorObs:', observation[idx])


        # -------------------------------------
        # Create Tensors
        # -------------------------------------
        # - pad all relevant inputs to max_mini_batch_vars

        # Actor Inputs
        all_actions_masks_pad = self.pad_to_len(action_masks, max_mini_batch_vars, pad_val=0)
        all_actions_masks_tensor = tf.convert_to_tensor(all_actions_masks_pad, dtype=tf.int32)
        all_actions_pad = self.pad_to_len(all_actions, max_mini_batch_vars, pad_val=0)
        all_actions_tensor = tf.convert_to_tensor(all_actions_pad, dtype=tf.int32)
        all_logprobs_pad = self.pad_to_len(all_logprobs, max_mini_batch_vars, pad_val=0)
        all_logprobs_tensor = tf.convert_to_tensor(all_logprobs_pad, dtype=tf.float32)
        all_observations_pad = self.pad_to_len(observation, max_mini_batch_vars, pad_val=0)
        all_observations_tensor = tf.convert_to_tensor(all_observations_pad, dtype=tf.float32)
        all_advantages_pad = self.pad_to_len(all_advantages, max_mini_batch_vars, pad_val=0)
        all_advantages_tensor = tf.convert_to_tensor(all_advantages_pad, dtype=tf.float32)

        # Critic Inputs
        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        all_returns_pad = self.pad_to_len(all_returns, max_critic_obs_len, pad_val=0)
        all_returns_tensor = tf.convert_to_tensor(all_returns_pad, dtype=tf.float32)
        all_returns_tensor = tf.expand_dims(all_returns_tensor, axis=-1)

        # -------------------------------------
        # Train Actor
        # -------------------------------------

        # print('\n\n----------- INPUTS')
        # print('Observations:', all_observations_tensor.shape)
        # print('Actions:', all_actions_tensor.shape)
        # print('LogProbs:', all_logprobs_tensor.shape)
        # print('Advantages:', all_advantages_tensor.shape)
        # print('CondVars:', cond_vars_tensor.shape)
        # print('P Encoding:', p_encoding_tensor.shape)
        # print('Action Masks:', all_actions_masks_tensor.shape)
        # print('P Encoding Mask:', p_encoding_mask_tensor.shape)

        policy_update_itr = 0
        for i in range(self.train_actor_iterations):

            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                all_observations_tensor,
                all_actions_tensor,
                all_logprobs_tensor,
                all_advantages_tensor,
                cond_vars_tensor,
                p_encoding_tensor,
                all_actions_masks_tensor,
                p_encoding_mask_tensor
            )
            if abs(kl) > 1.5 * self.target_kl:
                # Early Stopping
                break
        kl = kl.numpy()
        entr = entr.numpy()
        policy_loss = policy_loss.numpy()
        actor_loss = actor_loss.numpy()
        # print('finished training actor')


        # -------------------------------------
        # Train Critic
        # -------------------------------------

        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,  # Observations
                cond_vars_tensor,  # Weight
                p_encoding_tensor,  # Problem Nodes
                p_encoding_mask_tensor,  # Attention mask
                all_returns_tensor,
                all_values_mask,
            )
        value_loss = value_loss.numpy()

        self.run_info['return'].append(np.mean(all_rewards_flat))
        self.run_info['return_avg'].append(np.mean(self.run_info['return']))
        self.run_info['c_loss'].append(value_loss)
        self.run_info['kl'].append(kl)
        self.run_info['entropy'].append(entr)
        self.run_info['epoch_evals'].append(len(eval_designs))

        # Update nfe
        self.nfe = self.get_total_nfe()



    def run_val_epoch(self, val_models, val_key):
        val_models_hvs = []
        print('Val problems', end=': ')
        for idx, v_model in enumerate(val_models):
            v_problem = v_model.problem_formulation
            # weights = list(deepcopy(self.objective_weights))
            weights = list(np.linspace(0.00, 1.00, 100))
            # weights = [item for item in weights for i in range(3)]

            # GREEDY SEARCH
            designs = self.actor.generate(v_problem, w=weights)
            designs_str = [''.join([str(d) for d in design]) for design in designs]
            designs_str_unique = list(set(designs_str))
            num_designs = len(designs_str_unique)
            in_weights = weights

            # BEAM SEARCH
            # designs, in_weights = beam_search(self.actor, v_problem, weights, beam_width=10)
            # num_designs = len(designs)

            eval_problems = [v_problem for _ in range(len(designs))]
            population = Population(30, REF_POINT, v_model)
            eval_populations = [population for _ in range(len(designs))]
            objectives = self.eval_manager.evaluate(eval_problems, designs)
            evals = self.calc_reward_batch(designs, in_weights, objectives, eval_problems, eval_populations)

            # # print('DESIGNS FOR VAL', idx, ':', len(designs))
            # eval_problems = [v_problem for _ in range(len(designs))]
            # eval_designs = designs
            # # eval_designs_weights = tf.convert_to_tensor([[w] for w in weights])
            # population = Population(30, REF_POINT, v_model)
            # eval_populations = [population for _ in range(len(designs))]
            # objectives = self.eval_manager.evaluate(eval_problems, eval_designs)
            # evals = self.calc_reward_batch(eval_designs, eval_designs_weights, objectives, eval_problems, eval_populations)
            # Find the hv of the population
            hv = population.calc_hv()
            val_models_hvs.append(hv)
            # print('HV:', hv)
            # if last iteration
            if idx == len(val_models) - 1:
                print(f'({hv:.3f}, {num_designs}, {v_problem["y_res"]}x{v_problem["x_res"]})')
            else:
                print(f'({hv:.3f}, {num_designs}, {v_problem["y_res"]}x{v_problem["x_res"]})', end=', ')
            self.run_info['pareto_sen'][idx].append(num_designs)


        self.run_info[val_key].append(np.mean(val_models_hvs))

    # -------------------------------------
    # Calc Reward
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
            r_coef = 1.0
            reward = (term1 + term2) * r_coef
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

    def sample_actor(self, observation, cross_obs, p_encoding, p_encoding_mask):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx, p_encoding, p_encoding_mask)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # shape=(global_mini_batch_size, 1)
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx, p_encoding, p_encoding_mask):
        # print('sampling actor', inf_idx)
        pred_probs = self.actor([observation_input, cross_input, p_encoding, p_encoding_mask], training=TRAIN_CALL)  # shape (batch, seq_len, 2)

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

    def sample_critic(self, observation, parent_obs, p_encoding, p_encoding_mask):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx, p_encoding, p_encoding_mask)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx, p_encoding, p_encoding_mask):
        t_value = self.critic([observation_input, parent_input, p_encoding, p_encoding_mask])  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value




    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer,
            p_encoding_buffer,
            action_mask_buffer,
            encoding_attn_mask
    ):
        action_mask_buffer = tf.cast(action_mask_buffer, tf.float32)
        logprobability_buffer = logprobability_buffer * action_mask_buffer
        with tf.GradientTape() as tape:
            pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer, encoding_attn_mask], training=TRAIN_CALL)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            logprobability = logprobability * action_mask_buffer

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
            # pred_probs_m = pred_probs * action_mask_buffer[:, :, tf.newaxis]
            # pred_log_probs_m = pred_log_probs * action_mask_buffer[:, :, tf.newaxis]
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        policy_grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        #  KL Divergence
        pred_probs = self.actor([observation_buffer, parent_buffer, p_encoding_buffer, encoding_attn_mask], training=TRAIN_CALL)  # shape: (batch, seq_len, 2
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        logprobability = logprobability * action_mask_buffer
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def train_critic(
            self,
            observation_buffer,
            parent_buffer,
            p_encoding_buffer,
            p_encoding_mask,
            return_buffer,
            pred_mask,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.critic(
                [observation_buffer, parent_buffer, p_encoding_buffer, p_encoding_mask])  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = self.critic_loss(return_buffer, pred_values, sample_weight=pred_mask)

        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return value_loss

import config


if __name__ == '__main__':

    pop_size = 50
    ref_point = np.array([0, 1])

    actor_path, critic_path = None, None
    if save_start_epoch > 0:
        actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_' + str(save_start_epoch))
        critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_' + str(save_start_epoch))
    train_models = [Model(deepcopy(problem), num_procs=NUM_PROCS) for problem in train_problems]

    val_models = [Model(deepcopy(problem), num_procs=NUM_PROCS) for problem in val_problems]
    val_models_out = [Model(deepcopy(problem), num_procs=NUM_PROCS) for problem in val_problems_out]



    pops = [Population(pop_size, ref_point, problem) for idx, problem in enumerate(train_models)]
    ppo = TrussPPOVL(train_models, pops, max_nfe, actor_path, critic_path, run_name=save_name, val_models=val_models, val_models_out=val_models_out)
    ppo.run()











