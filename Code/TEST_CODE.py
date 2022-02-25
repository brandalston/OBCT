
import model_runs
import pareto_runs


'''
models = ['FlowOCT','BendersOCT','AGHA','MCF1','MCF2',
          'CUT1','CUT2-GRB','CUT1-ALL',
          'CUT2-FRAC1-ROOT', 'CUT1-FRAC2-ROOT'
          'CUT1-BOTH-I1-F3']
time_limit = 3600
repeats = 5
extras = None
file = 'test.csv'
data_names = ['hayes-roth_enc','soybean-small_enc','monk1_enc']
heights = [2,3,4,5]
tuning = 'warm_start'
model_runs.main(["-d",data_names,"-h",heights,"-m",models,"-t",time_limit,"-e",extras,"-r",repeats,"-f",file,"-c",tuning])
'''

height = 5
models = ['AGHA','MCF1','MCF2','CUT1','CUT2']
repeats = 5
data_names = ['house-votes-84_enc']
file = 'pareto.csv'
pareto_runs.main(["-d",data_names,"-h",height,"-m",models,"-t",3600,"-r",repeats,"-f",file])
