import model_runs
import pareto_runs

time_limit = 3600
rand_states = [0]
file = 'warm_start_results.csv'
heights = [3]
data_names = ['tic-tac-toe_enc']
models = ['CUT2']
extras = None
tuning = 'warm_start'
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])

rand_states = [138, 15, 89, 42, 0]
heights = [4,5]
data_names = ['tic-tac-toe_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])


rand_states = [138, 15, 89, 42, 0]
heights = [3,4,5]
data_names = ['car_evaluation_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])
