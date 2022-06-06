import model_runs
import pareto_runs

time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'test_results.csv'
models = ['MCF2']
consol_log = False
heights = [2,3,4,5]
data_names = ['soybean-small_enc','monk3_enc','monk1_enc','house-votes-84_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

heights = [2]
data_names = ['hayes-roth_enc','monk2_enc','spect_enc','breast-cancer_enc',
              'balance-scale','tic-tac-toe_enc','car_evaluation_enc',
              'kr-vs-kp_enc','fico_binary_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

heights = [3]
data_names = ['hayes-roth_enc','monk2_enc','spect_enc','balance-scale_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])
