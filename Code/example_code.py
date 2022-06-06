import model_runs
import pareto_runs


# MODEL RUN EXAMPLE
"""
data_names = ['soybean-small_enc','monk3_enc','balance-scale_enc','car_evaluation_enc']
heights = [3,4,5]
models = ['FlowOCT', 'BendersOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
time_limit = 3600
extras = ['regularization-3']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'test_results.csv'
data_names = ['soybean-small_enc']
models = ['CUT1']
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

# PARETO FRONTIER EXAMPLE
height = 2
models = ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
rand_states = [15, 78, 0]
data_names = ['hayes-roth_enc', 'house-votes-84_enc']
file = 'pareto_test.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", rand_states, "-f", file])

"""

data_names = ['soybean-small_enc','monk3_enc','monk1_enc','hayes-roth_enc','monk2_enc',
              'house-votes-84_enc','spect_enc','breast-cancer_enc','balance-scale_enc',
              'tic-tac-toe_enc','car_evaluation_enc','kr-vs-kp_enc','fico_binary_enc']
heights = [2,3,4,5]
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'test_results.csv'
models = ['MCF2']
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])
