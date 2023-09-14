import model_runs


# Basic standard testing example
data_names = ['soybean_small', 'monk3', 'car', 'iris', 'climate']
heights = [3,4,5]
models = ['MCF1', 'MCF2', 'CUT1-ALL', 'CUT2-FRAC-3']
time_limit = 3600
extras = ['max_features-25']
seeds = [13, 58, 94, None]
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", seeds, "-f", 'example_code_output.csv', "-l", False])


# Bi-objective example
height = 4
models = ['FlowOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
seeds = [15, 78, 0]
priorities = ['data','equal']
data_names = ['hayes_roth', 'house_votes_84']
model_runs.biobjective(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-p", priorities,
                        "-r", seeds, "-f", 'biobj_example.csv', "-l", False])


# Pareto frontier example
models = ['MCF1', 'MCF2', 'CUT1-GRB', 'CUT2-ALL']
seeds = [15, 78, 0]
data_names = ['hayes_roth', 'house_votes_84']
model_runs.pareto(["-d", data_names, "-h", 5, "-m", models, "-t", 3600, "-r", seeds, "-f", 'pareto_example.csv'])
