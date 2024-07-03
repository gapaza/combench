import tensorflow as tf
from combench.models.truss.problems.cantilever import get_problems
from combench.models import truss
from copy import deepcopy
import config
from combench.nn.trussDecoderVL import get_models
import os
import numpy as np

from combench.models.truss.TrussModel import TrussModel as Model
from combench.models.truss.nsga2 import TrussPopulation as Population
from combench.models.truss.nsga2 import TrussDesign as Design
from combench.models.truss.eval_process import EvaluationProcessManager


# ----------------------------
# Analyze Pareto
# ----------------------------

def analyze_pareto(model, problem, eval_manager):
    seq_len = truss.rep.get_num_bits(problem)
    weights = list(np.linspace(0.0, 1.0, 10))
    truss_model = Model(problem)

    print('\n\n------------------ Analyzing '+str(len(problem['nodes']))+' Node Problem')
    print('Weights:', weights)
    print('--------')


    # 1. Calc hv from greedy search
    designs_greedy = model.generate(problem, w=weights)
    designs_str = [''.join([str(d) for d in design]) for design in designs_greedy]
    designs_str_unique = list(set(designs_str))
    num_designs_greedy = len(designs_str_unique)
    designs_weights = weights
    pop_greedy = Population(30, np.array([0, 1]), truss_model)
    eval_problems = [problem for _ in range(len(designs_greedy))]
    eval_populations = [pop_greedy for _ in range(len(designs_greedy))]
    objectives = eval_manager.evaluate(eval_problems, designs_greedy)
    evals = calc_reward_batch(designs_greedy, designs_weights, objectives, eval_problems, eval_populations)
    hv_greedy = pop_greedy.calc_hv()

    print('Greedy Designs:', num_designs_greedy)
    print('Greedy Design Example:', designs_str_unique[0])
    print('Greedy HV:', hv_greedy)
    print('--------')


    # 2. Calc hv from beam search
    beam_widths = [2, 5, 10, 20, 50, 100]
    beam_width_hvs = []
    beam_width_sens = []
    for beam_width in beam_widths:
        designs_beam, designs_weights = beam_search(actor, problem, weights, beam_width=beam_width)
        num_designs_beam = len(designs_beam)
        pop_beam = Population(30, np.array([0, 1]), truss_model)
        eval_problems = [problem for _ in range(len(designs_beam))]
        eval_populations = [pop_beam for _ in range(len(designs_beam))]
        objectives = eval_manager.evaluate(eval_problems, designs_beam)
        evals = calc_reward_batch(designs_beam, designs_weights, objectives, eval_problems, eval_populations)
        hv_beam = pop_beam.calc_hv()
        beam_width_hvs.append(hv_beam)
        beam_width_sens.append(num_designs_beam)
        print(beam_width, 'Beam:', num_designs_beam, hv_beam)



# ----------------------------
# Beam Search
# ----------------------------

def beam_search(model, problem, weights, beam_width=5):
    seq_len = truss.rep.get_num_bits(problem)
    problem_encoding, problem_mask = truss.rep.get_problem_encoding_padded(problem, seq_len)


    num_weights = len(weights)

    start_token_id = 1

    beam_observations = [
        [[start_token_id] for _ in range(1)] for _ in range(num_weights)
    ]
    beam_designs = [
        [[] for _ in range(1)] for _ in range(num_weights)
    ]
    beam_log_probs = [
        [[] for _ in range(1)] for _ in range(num_weights)
    ]

    # batch size is total number of observations across all beams
    batch_size = 0
    for b_obs in beam_observations:
        batch_size += len(b_obs)

    # print('Batch size', batch_size)


    for seq_idx in range(seq_len):
        # weights = [weight for _ in range(batch_size)]
        # weights_full = [[i] for i in weights]
        # weights_tensor = tf.constant(weights_full, dtype=tf.float32)

        problem_encoding_t = tf.convert_to_tensor(problem_encoding, dtype=tf.float32)
        problem_encoding_t = tf.expand_dims(problem_encoding_t, axis=0)
        problem_encoding_t = tf.tile(problem_encoding_t, [batch_size, 1, 1])

        problem_mask_t = tf.convert_to_tensor(problem_mask, dtype=tf.int32)
        problem_mask_t = tf.expand_dims(problem_mask_t, axis=0)
        problem_mask_t = tf.tile(problem_mask_t, [batch_size, 1])



        # Old observations
        # observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)

        # Combine all observations from all beams
        observations_tensor = []
        weights_tensor = []
        beam_indices = []
        for idx, b_obs in enumerate(beam_observations):
            observations_tensor.extend(b_obs)
            weights_tensor.extend([[weights[idx]] for _ in range(len(b_obs))])
            beam_indices.extend([idx for _ in range(len(b_obs))])
        observations_tensor = tf.convert_to_tensor(observations_tensor, dtype=tf.float32)
        weights_tensor = tf.convert_to_tensor(weights_tensor, dtype=tf.float32)
        # print('Observation tensor:', observations_tensor)
        # print('Weights tensor:', weights_tensor)




        probs = model([observations_tensor, weights_tensor, problem_encoding_t, problem_mask_t])
        # print('Probs', probs)
        probs_inf = probs[:, seq_idx, :]
        log_probs_inf = tf.math.log(probs_inf + 1e-10)

        expanded_beam_log_probs = [
            [] for _ in range(num_weights)
        ]
        expanded_beam_observations = [
            [] for _ in range(num_weights)
        ]
        expanded_beam_designs = [
            [] for _ in range(num_weights)
        ]


        cnt = 0
        for weight_idx in range(len(beam_observations)):
            # print('beam log probs:', weight_idx, beam_log_probs)

            beam_obs = beam_observations[weight_idx]
            b_log_probs = beam_log_probs[weight_idx]
            b_designs = beam_designs[weight_idx]
            for i in range(len(beam_obs)):
                des_curr_log_probs = b_log_probs[i]
                des_curr_obs = beam_obs[i]
                des_curr = b_designs[i]
                log_prob_zero = log_probs_inf[cnt, 0].numpy()
                log_prob_one = log_probs_inf[cnt, 1].numpy()

                expanded_beam_observations[weight_idx].append(deepcopy(des_curr_obs) + [0 + 2])
                expanded_beam_log_probs[weight_idx].append(deepcopy(des_curr_log_probs) + [log_prob_zero])
                expanded_beam_designs[weight_idx].append(deepcopy(des_curr) + [0])

                expanded_beam_observations[weight_idx].append(deepcopy(des_curr_obs) + [1 + 2])
                expanded_beam_log_probs[weight_idx].append(deepcopy(des_curr_log_probs) + [log_prob_one])
                expanded_beam_designs[weight_idx].append(deepcopy(des_curr) + [1])

                cnt += 1


        # zip the expanded observations and log probs and sort
        for weight_idx in range(len(beam_observations)):
            beam_expanded_zip = list(zip(expanded_beam_observations[weight_idx], expanded_beam_log_probs[weight_idx], expanded_beam_designs[weight_idx]))
            beam_expanded_zip = sorted(beam_expanded_zip, key=lambda x: sum(x[1]), reverse=True)
            expanded_beam_observations[weight_idx], expanded_beam_log_probs[weight_idx], expanded_beam_designs[weight_idx] = zip(*beam_expanded_zip)

            # Check if beam width is exceeded
            if len(expanded_beam_observations[weight_idx]) > beam_width:
                expanded_beam_observations[weight_idx] = expanded_beam_observations[weight_idx][:beam_width]
                expanded_beam_log_probs[weight_idx] = expanded_beam_log_probs[weight_idx][:beam_width]
                expanded_beam_designs[weight_idx] = expanded_beam_designs[weight_idx][:beam_width]

        # Update the beam observations
        beam_observations = deepcopy(expanded_beam_observations)
        beam_log_probs = deepcopy(expanded_beam_log_probs)
        beam_designs = deepcopy(expanded_beam_designs)

        batch_size = 0
        for b_obs in beam_observations:
            batch_size += len(b_obs)

    # Get beam designs for each weight
    all_designs = []
    all_unique_designs = []
    all_designs_set = set()
    all_unique_weights = []
    for weight_idx in range(len(beam_observations)):
        all_designs.extend(beam_designs[weight_idx])
        beam_designs_str = [''.join([str(i) for i in d]) for d in beam_designs[weight_idx]]
        unique_beam_designs = list(set(beam_designs_str))

        # print('\nUnique designs for weight', weights[weight_idx])
        for d in unique_beam_designs:
            if d not in all_designs_set:
                all_designs_set.add(d)
                all_unique_designs.append(d)
                all_unique_weights.append(weights[weight_idx])
                # print(d)
    return all_unique_designs, all_unique_weights


# ----------------------------
# Helpers
# ----------------------------


def calc_reward_batch(designs, weights, objectives, problems, populations):
    opt_dir = ['max', 'min']
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
        design_obj.epoch = deepcopy(0)
        design_obj = population.add_design(design_obj)


        # Add to returns
        returns.append([reward, design_obj])

    return returns




#  Val problems: (0.267, 2), (0.349, 10), (0.405, 26), (0.313, 7), (0.359, 36), (0.415, 48), (0.428, 25), (0.440, 49), (0.448, 66)

# --- Validation 1
# Greedy: (10, .267, 3), (100, 0.267, 3), (200, 0.267, 3)
# Beam: (10-weight, 3-beam, 0.267-hv, 3-designs), (10-weight, 5-beam, 0.267-hv, 3-designs), (10-weight, 10-beam, 0.267-hv, 3-designs), (10-weight, 20-beam, 0.267-hv, 3-designs)





if __name__ == '__main__':
    r_num = 4
    load_name = 'cantilever-NxM-mtl-' + str(r_num)
    actor_path = os.path.join(config.results_dir, load_name, 'pretrained', 'actor_weights_23050')
    critic_path = os.path.join(config.results_dir, load_name, 'pretrained', 'critic_weights_23050')
    actor, critic = get_models(actor_path, critic_path)


    beam_width = 3
    actions = 2

    train_problems, val_problems, val_problems_out = get_problems()
    eval_manager = EvaluationProcessManager(32)



    # 1. Set norms
    for idx, v_problem in enumerate(val_problems):
        truss.set_norms(v_problem)

    # 2. Analyze pareto for each problem
    for idx, v_problem in enumerate(val_problems):
        v_problem = val_problems[idx]
        results = analyze_pareto(actor, v_problem, eval_manager)

        # if idx > 2:
        #     eval_manager.shutdown()
        #     exit(0)



