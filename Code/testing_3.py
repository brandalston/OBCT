import model_runs
import pareto_runs


time_limit = 3600
extras = None
tuning = None
file = 'test_results.csv'
models = ['MCF2']
consol_log = False
data_names = ['balance-scale_enc']
heights = [5]
rand_states = [138, 15, 89, 42, 0]
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])
