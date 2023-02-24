import model_runs, pareto_runs
from Benchmark_Methods import DL8_5, GOSDTg, FB_OCT

# Paper solution time, gap, and accuracy results
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
consol_log = False

model_runs.main(["-d", numerical_datasets + categorical_datasets, "-h", heights, "-m", models, "-t", time_limit,
                 "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

model_runs.main(["-d", numerical_datasets + categorical_datasets, "-h", heights, "-m", models, "-t", time_limit,
                 "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

FB_OCT.main(["-d", numerical_datasets+categorical_datasets, "-h", heights, "-m", ['Flow', 'Benders'],
             "-t", time_limit, "-e", extras, "-r", rand_states, "-f", file, "-l", consol_log])

DL8_5.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
            "-t", time_limit, "-r", rand_states, "-f", file])

GOSDTg.main(["-d", numerical_datasets+categorical_datasets, "-h", heights,
             "-t", time_limit, "-r", rand_states, "-f", file])


# Paper pareto frontier results
height = 5
models = ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
repeats = 5
data_names = ['soybean_small', 'monk1', 'monk3', 'house-votes-84']
file = None
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", repeats, "-f", file])
