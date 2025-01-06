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
models = ['FlowOCT', 'SCF', 'MCF', 'POKE', 'CUT']
seeds = [15, 78, 0]
priorities = ['data','equal']
data_names = ['hayes_roth', 'house_votes_84']
model_runs.multiobj(["-d", data_names, "-h", height, "-m", models, "-t", 3600, "-p", priorities,
                     "-r", seeds, "-f", 'biobj_example.csv', "-l", False])


# Pareto frontier example
models = ['SCF', 'MCF', 'POKE-GRB', 'CUT-ALL']
seeds = [15, 78, 0]
data_names = ['hayes_roth', 'house_votes_84']
model_runs.pareto(["-d", data_names, "-h", 5, "-m", models, "-t", 3600, "-r", seeds, "-f", 'pareto_example.csv'])


# call OBCT directly without using model_runs.py functions
from OBCT import OBCT
from TREE import TREE
import UTILS
from sklearn.model_selection import train_test_split

data = UTILS.get_data('ionosphere')
rand_state = 15
train_set, test_set = train_test_split(data, train_size=0.65, random_state=rand_state)

tree = TREE(h=5)
opt_model = OBCT(data=train_set, tree=tree, target='target', model='CUT-GRB', name='ionosphere',
                 time_limit=3600, warm_start={'use': False, 'time': 0})
opt_model.formulation()
opt_model.extras()
opt_model.model.update()
opt_model.model.optimize()
UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                    rand_state=rand_state, results_file=None, data_name='ionosphere')

