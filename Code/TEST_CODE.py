
import model_runs
import pareto_runs


### MODEL RUN EXAMPLE
data_names = ['monk1_enc']
heights = [2]
models = ['FlowOCT','BendersOCT','MCF1','MCF2','CUT1','CUT2']
time_limit = 3600
extras = ['fixing','max_features-15']
rand_states = [138]
tuning = None
file = 'results.csv'
plot_fig = False
consol_log = True
model_runs.main(["-d",data_names,"-h",heights,"-m",models,"-t",time_limit,"-e",extras,"-r",rand_states,"-c", tuning,"-f",file,"-p",plot_fig,"-l", consol_log])

### PARETO FRONTIER EXAMPLE
height = 5
models = ['AGHA','MCF1','MCF2','CUT1','CUT2']
repeats = 5
data_names = ['soybean-small_enc','monk1_enc']
file = 'pareto.csv'
pareto_runs.main(["-d",data_names,"-h",height,"-m",models,"-t",3600,"-r",repeats,"-f",file])
