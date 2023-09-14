import model_runs
from Benchmark_Methods import DL85, GOSDTg, FB_OCT

numerical_datasets = ['iris', 'blood', 'climate', 'ionosphere']
categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
heights = [2, 3, 4, 5]
models = ['MCF1', 'MCF2', 'CUT1-GRB', 'CUT2-GRB', 'CUT1-ALL', 'CUT2-ALL',
          'CUT1-FRAC1-ROOT', 'CUT1-FRAC2-ROOT', 'CUT1-FRAC3-ROOT',
          'CUT2-FRAC1-ROOT', 'CUT2-FRAC2-ROOT', 'CUT2-FRAC3-ROOT']
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'paper_results.csv'
log_file = False

# Accuracy and solution time (gap) results
model_runs.main(["-d", numerical_datasets + categorical_datasets, "-h", heights, "-m", models,
                 "-t", time_limit, "-r", rand_states, "-f", file, "-l", False, "-e", None])

FB_OCT.main(["-d", numerical_datasets+categorical_datasets, "-h", heights, "-m", ['Flow', 'Benders'],
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", log_file])

DL85.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
            "-t", time_limit, "-r", rand_states, "-f", file])

GOSDTg.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
             "-t", time_limit, "-r", rand_states, "-f", file])

# Bi-objective results
models = ['MCF1', 'MCF2', 'CUT1, CUT2']
priorities = ['data', 'equal']
model_runs.biobjective(["-d", ['breast_cancer'], "-h", [5], "-m", models, "-t", 3600,
                     "-p", priorities, "-r", rand_states, "-f", file, "-l", log_file])

# Pareto frontier results
models = ['FlowOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
data_names = ['soybean_small', 'monk1', 'monk3', 'house_votes_84']
model_runs.pareto(["-d", data_names, "-h", 5, "-m", models,
                   "-t", 3600, "-r", rand_states, "-f", None])
