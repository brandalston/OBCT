
import model_runs
import pareto_runs

'''
data_names = ['house-votes-84_enc','soybean-small_enc','monk1_enc']
heights = [2,3,4,5]
models = ['FlowOCT','BendersOCT','AGHA','MCF1','MCF2',
          'CUT1','CUT2-GRB','CUT1-ALL',
          'CUT2-FRAC1-ROOT', 'CUT1-FRAC2-ROOT'
          'CUT1-BOTH-I1-F3']
time_limit = 3600
extras = None
rand_states = [138, 15, 89, 42, 0]
file = 'test.csv'
tuning = None
model_runs.main(["-d",data_names,"-h",heights,"-m",models,"-t",time_limit,"-e",extras,"-r",rand_states,"-f",file,"-c",tuning, "-g", False])
'''

height = 5
models = ['AGHA','MCF1','MCF2','CUT1','CUT2']
models = []
repeats = 5
data_names = ['soybean-small_enc','monk1_enc']
file = 'pareto.csv'
pareto_runs.main(["-d",data_names,"-h",height,"-m",models,"-t",3600,"-r",repeats,"-f",file])
