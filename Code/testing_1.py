import model_runs
import pareto_runs

time_limit = 3600
extras = None
tuning = None
file = 'test_results.csv'
models = ['MCF2']
consol_log = False
heights = [2]
data_names = ['tic-tac-toe_enc']
rand_states = [89, 42, 0]
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

rand_states = [138, 15, 89, 42, 0]
data_names = ['car_evaluation_enc', 'kr-vs-kp_enc', 'fico_binary_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])
