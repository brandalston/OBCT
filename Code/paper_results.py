import model_runs
import pareto_runs


# Paper solution time, gap, and accuracy results
data_names = ['soybean-small_enc','monk3_enc','monk1_enc','hayes-roth_enc','monk2_enc',
              'house-votes-84_enc','spect_enc','breast-cancer_enc','balance-scale_enc',
              'tic-tac-toe_enc','car_evaluation_enc','kr-vs-kp_enc','fico_binary_enc']
heights = [2,3,4,5]
models = ['FlowOCT','BendersOCT','MCF1','MCF2',
          'CUT1-GRB','CUT2-GRB','CUT1-ALL','CUT2-ALL',
          'CUT1-FRAC1-ROOT','CUT1-FRAC2-ROOT','CUT1-FRAC3-ROOT',
          'CUT2-FRAC1-ROOT','CUT2-FRAC2-ROOT','CUT2-FRAC3-ROOT']
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = None
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit,
    "-e", extras, "-r", rand_states, "-c", tuning, "-f", file, "-l", consol_log])

# Paper pareto frontier results
height = 5
models = ['FOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']
repeats = 5
data_names = ['soybean-small_enc', 'monk1_enc', 'monk3_enc', 'house-votes-84_enc']
file = 'pareto_test.csv'
pareto_runs.main(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-r", repeats, "-f", file])

