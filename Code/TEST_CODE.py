import model_runs
import pareto_runs

# '''
# MODEL RUN EXAMPLE
data_names = ['soybean-small_enc','monk3_enc','breast-cancer_enc','tic-tac-toe_enc']
heights = [2,3,4,5]
models = ['MCF1','MCF2','CUT1-ALL','CUT2-FRAC1','FlowOCT','BendersOCT','CART']
time_limit = 3600
extras = ['max_features-25']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'results.csv'
plot_fig = False
consol_log = False

models = ['CART','CART_STR']
data_names = ['soybean-small','soybean-small_enc','monk3','monk3_enc','monk1','monk1_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])
data_names = ['hayes-roth','hayes-roth_enc','monk2','monk2_enc','house-votes-84','house-votes-84_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])
data_names = ['spect','spect_enc','breast-cancer','breast-cancer_enc','balance-scale','balance-scale_enc']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])
data_names = ['tic-tac-toe','tic-tac-toe_enc','car_evaluation','car_evaluation_enc','kr-vs-kp','kr-vs-kp_enc', 'fico_binary', 'fico_binary_enc']
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