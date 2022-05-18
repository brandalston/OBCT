import model_runs
import pareto_runs

'''
# MODEL RUN EXAMPLE
data_names = ['soybean-small_enc','monk1_enc','monk3_enc','monk2_enc','house-votes-84_enc',
              'hayes-roth_enc','breast-cancer_enc','balance-scale_enc','spect_enc',
              'tic-tac-toe_enc','kr-vs-kp_enc','car_evaluation_enc','fico_binary_enc']
heights = [4]
models = ['CART']
time_limit = 3600
extras = ['max_features-25']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'results.csv'
plot_fig = False
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])
'''
time_limit = 3600
rand_states = [138, 15, 89, 42, 0]
file = 'warm_start_results.csv'
heights = [5]
data_names = ['fico_binary_enc']
models = ['MCF2']
extras = None
tuning = 'warm_start'
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])
'''
# PARETO FRONTIER EXAMPLE
height = 5
models = ['AGHA', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
repeats = 5
data_names = ['soybean-small_enc', 'monk1_enc', 'monk3_enc', 'house-votes-84_enc']
file = 'pareto.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", repeats, "-f", file])
'''