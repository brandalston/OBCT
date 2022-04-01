import model_runs
import pareto_runs

# '''
# MODEL RUN EXAMPLE
data_names = ['soybean-small_enc','monk3_enc','breast-cancer_enc','tic-tac-toe_enc']
heights = [2]
models = ['MCF1','MCF2','CUT1-ALL','CUT2-FRAC1','FlowOCT','BendersOCT','CART']
time_limit = 3600
extras = ['max_features-25']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'results.csv'
plot_fig = False
consol_log = False
data_names = ['soybean-small','soybean-small_enc']
models = ['CART']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])

'''
# PARETO FRONTIER EXAMPLE
height = 5
models = ['AGHA', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
repeats = 5
data_names = ['soybean-small_enc', 'monk1_enc', 'monk3_enc', 'house-votes-84_enc']
file = 'pareto.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", repeats, "-f", file])
'''