import model_runs
import pareto_runs

time_limit = 3600
rand_states = [138]
file = 'warm_start_results.csv'
heights = [5]
data_names = ['spect_enc']
models = ['MCF2']
extras = None
tuning = 'warm_start'
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", False, "-l", False])
