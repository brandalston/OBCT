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
import RESULTS as OR
from FlowOCT import FlowOCT
from BendersOCT import BendersOCT
import FlowOCTutils
from FlowOCTTree import Tree as FlowOCTTree

def main(argv):
    print(argv)
    data_files = None
    height = None
    time_limit = None
    modeltypes = None
    repeats = None
    file_out = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:",
                                   ["data_files=", "height=", "timelimit=",
                                    "models=", "repeats=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            height = arg
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
                       '# CB', 'User Cuts', 'Cuts/CB', 'CB-Time', 'INT-CB-Time', 'FRAC-CB-Time', 'CB-Eps',
                       'Model', 'Time Limit', 'Rand. State',
                       '% Fixed', 'Calibration', 'CC',
                       'Single Feature Use', 'Level Tree', 'Max Features', 'Super Feature']
    output_path = os.getcwd() + '/results_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(height) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_pareto.csv'
    else:
        output_name = str(file_out)
    out_file = output_path + output_name

    if file_out is None:
        with open(out_file, mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"')
            results_writer.writerow(summary_columns)
            results.close()

    target = 'target'
    repeats = repeats
    rand_states = [138, 15, 89, 42, 0]

    for file in data_files:
        # pull dataset to train model with
        data, encoding_map = OU.get_data(file.replace('.csv', ''), target)
        for i in range(repeats):
            print('\n\nDataset: ' + str(file) + ', H: ' + str(height) + ', Iteration: ' + str(i) + '. Run Start: ' + str(
                  time.strftime("%I:%M %p", time.localtime())))
            train_set, test_set = train_test_split(data, train_size=0.5, random_state=rand_states[i])
            for modeltype in modeltypes:
                WSV = None
                for num_features in range(1, 2 ** height):
                    extras = [f'num_features-{num_features}']
                    tree = TREE(h=height)
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
                    if 'AGHA' == modeltype: WSV = {'tree': tree.DG_prime.nodes(data=True), 'data': False}

        # Generate pareto plot of models using run averages of pareto.csv file
        pareto_data = pd.pareto_data = pd.read_csv(f'results_files/{file_out}', na_values='?')
        file_data = pareto_data[pareto_data['Data'] == file.replace('.csv', '')]
        frontier_avg = pd.DataFrame(columns=summary_columns)
        for model in ['AGHA','MCF1','MCF2','CUT1','CUT2','FlowOCT','BendersOCT']:
            sub_data = file_data.loc[file_data['Model'] == model]
            for feature in sub_data['Max Features'].unique():
                subsub_data = sub_data.loc[sub_data['Max Features'] == feature]
                frontier_avg = frontier_avg.append({
                    'Data': file.replace('.csv', ''), 'H': int(subsub_data['H'].mean()),
                    '|I|': int(subsub_data['|I|'].mean()),
                    'Out-Acc': 100 * subsub_data['Out-Acc'].mean(), 'In-Acc': 100 * subsub_data['In-Acc'].mean(),
                    'Sol-Time': subsub_data['Sol-Time'].mean(),
                    'Gap': 100 * subsub_data['Gap'].mean(), 'ObjVal': subsub_data['ObjVal'].mean(),
                    '# CB': subsub_data['# CB'].mean(), 'User Cuts': subsub_data['User Cuts'].mean(),
                    'Cuts/CB': subsub_data['Cuts/CB'].mean(),
                    'CB-Time': subsub_data['CB-Time'].mean(), 'INT-CB-Time': subsub_data['INT-CB-Time'].mean(),
                    'FRAC-CB-Time': subsub_data['FRAC-CB-Time'].mean(),
                    'CB-Eps': subsub_data['CB-Eps'].mean(), 'Model': model, 'Time Limit': time_limit,
                    'Rand State': 'None',
                    '% Fixed': 0, 'Calibration': False, 'CC': False, 'Single Feature Use': False,
                    'Level Tree': False,
                    'Max Features': float(feature), 'Super Feature': False
                }, ignore_index=True)
        OR.pareto_plot(frontier_avg)
