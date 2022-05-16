import model_runs
import pareto_runs


time_limit = 3600
file = 'warm_start_results.csv'
rand_states = [89, 42, 0]
heights = [5]
data_names = ['kr-vs-kp_enc']
models = ['CUT1']
extras = None
tuning = 'warm_start'
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])

rand_states = [138, 15, 89, 42, 0]
heights = [3,4,5]
data_names = ['fico_binary_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])
