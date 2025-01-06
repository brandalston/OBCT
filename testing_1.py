import model_runs
from Benchmark_Methods import FB_OCT#, DL85, GOSDTg


numerical_datasets = ['iris', 'blood', 'climate', 'ionosphere']
categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = True
file = 'paper_results.csv'
log_file = False

# Accuracy and solution time (gap) results w/ warm starts

# run 1
heights = [2]
set_a = ['balance_scale', 'breast_cancer', 'car', 'hayes_roth', 'house_votes_84', 'iris',
         'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [3]
set_b = ['balance_scale', 'hayes_roth', 'iris', 'monk1', 'monk3', 'soybean_small']
FB_OCT.main(["-d", set_b, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [4]
set_c = ['iris', 'monk1', 'soybean_small']
FB_OCT.main(["-d", set_c, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [5]
set_d = ['iris', 'kr_vs_kp', 'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe']
FB_OCT.main(["-d", set_d, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
"""
# run 2
heights = [2]
set_a = ['blood', 'fico_binary', 'ionosphere']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [3]
set_b = ['ionosphere', 'monk2']
FB_OCT.main(["-d", set_b, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [4]
set_c = ['fico_binary', 'hayes_roth', 'monk3']
FB_OCT.main(["-d", set_c, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [5]
set_d = ['house_votes_84']
FB_OCT.main(["-d", set_d, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 3
heights = [2]
set_a = ['climate', 'kr_vs_kp']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [3]
set_b = ['blood', 'breast_cancer', 'car']
FB_OCT.main(["-d", set_b, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 4
heights = [3]
set_a = ['climate', 'fico_binary', 'house_votes_84', 'kr_vs_kp', 'spect']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 5
heights = [3]
set_a = ['tic_tac_toe']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [4]
set_b = ['balance_scale', 'blood', 'breast_cancer', 'car']
FB_OCT.main(["-d", set_b, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 6
heights = [4]
set_a = ['climate', 'house_votes_84', 'ionosphere', 'kr_vs_kp', 'monk2']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 7
heigts = [4]
set_a = ['spect', 'tic_tac_toe']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
heights = [5]
set_b = ['balance_scale', 'blood', 'breast_cancer']
FB_OCT.main(["-d", set_b, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])
# run 8
heights = [5]
set_a = ['car', 'climate', 'fico_binary', 'hayes_roth', 'ionosphere']
FB_OCT.main(["-d", set_a, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])

categorical_datasets = ['climate']
heights = [5]
time_limit = 3600
extras = None
rand_states = [138]
tuning = False
file = 'paper_results.csv'
log_file = False
FB_OCT.main(["-d", categorical_datasets, "-h", heights, "-m", ['Benders'], "-c", tuning,
             "-t", time_limit, "-r", rand_states, "-f", file, "-l", log_file])
"""