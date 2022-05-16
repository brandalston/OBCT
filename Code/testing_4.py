import model_runs
import pareto_runs

time_limit = 3600
rand_states = [42, 0]
file = 'warm_start_results.csv'
heights = [4]
data_names = ['tic-tac-toe_enc']
models = ['CUT2']
tuning = 'warm_start'
extras = None
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])

rand_states = [138, 15, 89, 42, 0]
heights = [5]
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
