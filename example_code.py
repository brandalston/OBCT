import model_runs
import pareto_runs

data_names = ['soybean_small', 'monk3', 'car', 'iris', 'climate']
heights = [3,4,5]
models = ['MCF1', 'MCF2', 'CUT1-ALL', 'CUT2-FRAC-3']
time_limit = 3600
extras = ['max_features-25']
rand_seed = [13, 58, 94, None]
tuning = None
file = 'example_code_output.csv'
consol_log = False

model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_seed, "-c", tuning, "-f", file, "-l", consol_log])

height = 4
models = ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
rand_states = [15, 78, 0]
data_names = ['hayes_roth', 'house_votes_84']
file = 'pareto_test.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", rand_states, "-f", file])
