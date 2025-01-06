import model_runs
from Benchmark_Methods import FB_OCT#, DL85, GOSDTg

"""
Experiments found in our manuscript
"""

numerical_datasets = ['iris', 'blood', 'climate', 'ionosphere']
categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
heights = [2, 3, 4, 5]
models = ['SCF', 'MCF', 'POKE', 'CUT', 'POKE-GRB', 'CUT-GRB',
          'POKE-FRAC1-ROOT', 'POKE-FRAC2-ROOT', 'POKE-FRAC3-ROOT',
          'CUT-FRAC1-ROOT', 'CUT-FRAC2-ROOT', 'CUT-FRAC3-ROOT']
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = True
file = 'paper_results.csv'
log_file = False

# Accuracy and solution time (gap) results w/ warm starts
model_runs.main(["-d", numerical_datasets + categorical_datasets, "-h", heights, "-m", models, "-c", tuning,
                 "-t", time_limit, "-r", rand_states, "-f", file, "-l", False, "-e", None])

FB_OCT.main(["-d", numerical_datasets+categorical_datasets, "-h", heights, "-m", ['Flow', 'Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])

DL85.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
            "-t", time_limit, "-r", rand_states, "-f", file])

GOSDTg.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
             "-t", time_limit, "-r", rand_states, "-f", file])

# Accuracy and solution time (gap) results w/out warm starts
tuning = False
model_runs.main(["-d", numerical_datasets + categorical_datasets, "-h", heights, "-m", models, "-c", tuning,
                 "-t", time_limit, "-r", rand_states, "-f", file, "-l", False, "-e", None])

FB_OCT.main(["-d", numerical_datasets+categorical_datasets, "-h", heights, "-m", ['Flow', 'Benders'], "-c", tuning,
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])

DL85.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
            "-t", time_limit, "-r", rand_states, "-f", file])

GOSDTg.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
             "-t", time_limit, "-r", rand_states, "-f", file])

# Bi-objective results
models = ['SCF', 'MCF', 'POKE, CUT']
priorities = ['data', 'equal']
model_runs.multiobj(["-d", ['breast_cancer'], "-h", [5], "-m", models, "-t", 3600,
                     "-p", priorities, "-r", rand_states, "-f", file, "-l", log_file])

# Pareto frontier results
models = ['FlowOCT', 'SCF', 'MCF', 'POKE', 'CUT']
data_names = ['soybean_small', 'monk1', 'monk3', 'house_votes_84']
model_runs.pareto(["-d", data_names, "-h", 5, "-m", models,
                   "-t", 3600, "-r", rand_states, "-f", None])
