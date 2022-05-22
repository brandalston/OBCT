import model_runs
import pareto_runs


# MODEL RUN EXAMPLE
data_names = ['soybean-small_enc','monk3_enc','balance-scale_enc','car_evaluation_enc']
heights = [3,4,5]
models = ['FlowOCT', 'BendersOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
time_limit = 3600
extras = ['max_features-25']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'test_results.csv'
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

