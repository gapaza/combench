import os
import tensorflow as tf

# disable gpu
# tf.config.set_visible_devices([], 'GPU')

#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'combench')

results_dir = os.path.join(root_dir, 'results')
database_dir = os.path.join(root_dir, 'database')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(database_dir):
    os.makedirs(database_dir)

plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

plots_test_dir = os.path.join(plots_dir, 'test')
if not os.path.exists(plots_test_dir):
    os.makedirs(plots_test_dir)

plots_problems_dir = os.path.join(plots_dir, 'problems')
if not os.path.exists(plots_problems_dir):
    os.makedirs(plots_problems_dir)

plots_nsga2_dir = os.path.join(plots_dir, 'nsga2')
if not os.path.exists(plots_nsga2_dir):
    os.makedirs(plots_nsga2_dir)

plots_features_dir = os.path.join(plots_dir, 'features')
if not os.path.exists(plots_features_dir):
    os.makedirs(plots_features_dir)



sidenum = 3  # 3 | 4 | 5 | 6
sidenum_nvar_map = {2: 6, 3: 30, 4: 108, 5: 280, 6: 600, 7: 1134, 8: 1960, 9: 3168, 10: 4860, 11: 7150, 12: 10164, 13: 14040, 14: 18928, 15: 24990, 16: 32400, 17: 41344, 18: 52020, 19: 64638, 20: 79420}
# num_vars = sidenum_nvar_map[sidenum]

# For 3x3 cantilever truss problem
num_vars = 120  # 36





























