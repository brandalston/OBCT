import model_runs
import pareto_runs

time_limit = 3600
extras = None
rand_states = [89, 42, 0]
tuning = None
file = 'test_results.csv'
models = ['MCF2']
consol_log = False
heights = [5]
data_names = ['car_evaluation_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

