#!/usr/bin/python
import os
import time
import pandas as pd
import getopt
import sys
import csv
from os.path import exists
from sklearn.model_selection import train_test_split
from OBCT import OBCT
from TREE import TREE
import UTILS as OU
import SPEED_UP as OSP
import RESULTS as OR


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    repeats = None
    file_out = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "repeats=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-m", "--models"):
            modeltypes = arg
        elif opt in ("-r", "--repeats"):
            repeats = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out-Acc', 'In-Acc', 'Sol-Time', 'Gap', 'ObjVal',
                       '# CB', 'User Cuts', 'Cuts/CB', 'CB-Time', 'INT-CB-time', 'FRAC-CB-TIME', 'CB-Eps',
                       'Model', 'Time Limit', 'Rand. State',
                       '% Fixed', 'Calibration', 'CC',
                       'Single Feature Use', 'Level Tree', 'Max Features', 'Super Feature']
    output_path = os.getcwd() + '/results_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_pareto.csv'
    else:
        output_name = str(file_out)
    out_file = output_path + output_name
    if not exists(out_file):
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()
    target = 'target'
    repeats = repeats
    rand_states = [138, 15, 89, 42, 0]

    for file in data_files:
        # pull dataset to train model with
        data, encoding_map = OU.get_data(file.replace('.csv', ''), target)
        for h in heights:
            for i in range(repeats):
                print('\n\nDataset: ' + str(file) + ', H: ' + str(h) + ', Iteration: ' + str(i) + '. Run Start: ' + str(
                      time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=rand_states[i])
                for modeltype in modeltypes:
                    WSV = None
                    for num_features in range(1, 2 ** h):
                        extras = [f'num_features-{num_features}']
                        tree = TREE(h=h)
                        opt_model = OBCT(data=train_set, tree=tree, target=target, model=modeltype,
                                         time_limit=time_limit, encoding_map=encoding_map, model_extras=extras,
                                         unreachable=None, warm_start=WSV, name=file)
                        opt_model.formulation()
                        opt_model.extras()
                        opt_model.model.update()
                        opt_model.optimization()
                        OR.node_assign(opt_model, tree)
                        OR.tree_check(tree)
                        OR.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                         rand_state=rand_states[i], results_file=out_file, fig_file=None)
                        model_wsm_acc, model_wsm_assgn = OR.model_acc(tree=tree, target=target,
                                                                      data=train_set)
                        WSV = {'tree': tree.DG_prime.nodes(data=True), 'data': model_wsm_assgn}
        pareto_data = pd.read_csv(out_file, na_values='?')
        OR.pareto_frontier(pareto_data, modeltypes)
