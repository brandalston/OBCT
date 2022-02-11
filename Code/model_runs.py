#!/usr/bin/python
import os
import time
from doctest import master

import pandas as pd
import getopt
import sys
import csv
from os.path import exists
from sklearn.model_selection import train_test_split
import UTILS as OU
from TREE import TREE
from OBCT import OBCT
import SPEED_UP as OSP
import RESULTS as OR
from FlowOCTTree import Tree as FlowOCTTree
from FlowOCT import FlowOCT
import FlowOCTutils

def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    model_extras = None
    repeats = None
    file_out = None

    rand_states = [138, 15, 89, 42, 0]

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:e:r:f:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "extras=", "repeats=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-m", "--model"):
            modeltypes = arg
        elif opt in ("-e", "--extras"):
            model_extras = arg
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
    output_path = os.getcwd() + '/results_data_files/'
    fig_path = os.getcwd() + '/results_figures'
    if file_out is None:
        output_name = str(data_files) + '_' + 'H' + ':' + str(heights) + '_' + str(modeltypes) + \
                  '_' + 'T:' + str(time_limit) + '_' + str(model_extras) + '.csv'
    else: output_name = str(file_out)
    out_file = output_path + output_name
    if not exists(out_file):
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    # Using logger we log the output of the console in a text file
    # sys.stdout = OU.logger(output_path + output_name + '.txt')

    '''We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    repeats = repeats

    for file in data_files:
        # pull dataset to train model with
        data, encoding_map = OU.get_data(file.replace('.csv', ''), target)
        for h in heights:
            for i in range(repeats):
                print('\n\nDataset: '+str(file)+', H: '+str(h)+', Iteration: '+str(i)+'. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=rand_states[i])
                calibration_set, test_set = train_test_split(test_set, train_size=0.5, random_state=rand_states[i])
                model_set = pd.concat([train_set, calibration_set])
                WSV, unreachable = None, {'data': {}, 'tree': {}}
                if model_extras is not None:
                    if 'fixing' in model_extras: unreachable = OSP.fixing(TREE(h), model_set)
                for modeltype in modeltypes:
                    if 'OCT' not in modeltype:
                        if model_extras is not None:
                            if 'calibration' in model_extras:
                                best_tree, best_feats, best_acc = {}, 0, 0
                                wsm = None
                                for num_features in range(1, 2 ** h):
                                    extras = [f'max_features-{num_features}']
                                    cal_tree = TREE(h=h)
                                    opt_model = OBCT(data=calibration_set, tree=cal_tree, target=target, model=modeltype,
                                                     time_limit=time_limit, encoding_map=encoding_map, warm_start=wsm,
                                                     model_extras=extras, unreachable=unreachable,  name=file)
                                    opt_model.formulation()
                                    opt_model.model.update()
                                    opt_model.extras()
                                    opt_model.optimization()
                                    OR.node_assign(opt_model, cal_tree.DG_prime)
                                    OR.tree_check(cal_tree)
                                    cal_acc, cal_assign = OR.model_acc(tree=cal_tree, model=opt_model, data=calibration_set)
                                    wsm = {'tree': cal_tree.DG_prime.nodes(data=True), 'data': cal_assign}
                                    if cal_acc > best_acc:
                                        best_feats, best_acc = num_features, cal_acc
                                        best_tree = cal_tree
                                    model_wsm_acc, model_wsm_assgn = OR.model_acc(tree=best_tree, model=opt_model,
                                                                                  data=model_set)
                                    WSV = {'tree': best_tree.DG_prime.nodes(data=True), 'data': model_wsm_assgn}
                                if any((match := elem).startswith('max_features') for elem in model_extras):
                                    model_extras[model_extras.index(match)] = 'max_features-'+str(best_feats)
                                else: model_extras.append('max_features-'+str(best_feats))

                        tree = TREE(h=h)
                        opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype,
                                         time_limit=time_limit, encoding_map=encoding_map, model_extras=model_extras,
                                         unreachable=unreachable, warm_start=WSV, name=file)
                        opt_model.formulation()
                        if model_extras is not None: opt_model.extras()
                        opt_model.model.update()
                        opt_model.optimization()
                        # lp_name = output_path+'_'+str(file)+'_'+str(h)+'_'+str(modeltype)+'_'+'T:'+str(time_limit)+'_'+str(model_extras)
                        # opt_model.model.write(lp_name + '.lp')
                        fig_file = fig_path + str(file) + str(h) + str(modeltype) + str(time_limit) + '.png'
                        OR.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                         rand_state=rand_states[i], results_file=out_file, fig_file=fig_file)
                    else:
                        print('Model: FlowOCT')
                        OCT_tree = FlowOCTTree(d=h)
                        primal = FlowOCT(data=model_set, label=target, tree=OCT_tree, _lambda=0, time_limit=time_limit, mode='classification')
                        primal.create_primal_problem()
                        primal.model.update()
                        primal.model.optimize()
                        b_value = primal.model.getAttr("X", primal.b)
                        beta_value = primal.model.getAttr("X", primal.beta)
                        p_value = primal.model.getAttr("X", primal.p)
                        train_acc = FlowOCTutils.get_acc(primal, train_set, b_value, beta_value, p_value)
                        test_acc = FlowOCTutils.get_acc(primal, test_set, b_value, beta_value, p_value)
                        if primal.model.RunTime < time_limit:
                            print('Optimal solution found in ' + str(round(primal.model.Runtime, 2)) + 's. (' + str(
                                time.strftime("%I:%M %p", time.localtime())) + ')\n')
                        else:
                            print('Time limit reached. (', time.strftime("%I:%M %p", time.localtime()), ')\n')
                        with open(out_file, mode='a') as results:
                            results_writer = csv.writer(results, delimiter=',', quotechar='"')
                            results_writer.writerow(
                                [file.replace('.csv',''), h, len(model_set),
                                 test_acc, train_acc, primal.model.Runtime, primal.model.MIPGap,
                                 len({i for i in model_set.index if primal.z[i, 1].x > .5}),
                                 0, 0, 0, 0, 0, 0, 0, modeltype, time_limit, rand_states[i],
                                 0, False, False, False, 'None', 'None', False])
                            results.close()


if __name__ == "__main__":
    main(sys.argv[1:])


def fixing(data_names, models, heights, file_out):
    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', '# Fixed', 'Total Vars', '% Fixed', 'Model']
    output_path = os.getcwd() + '/results_data_files/'
    if file_out is None:
        output_name = str(data_names) + '_' + 'H' + ':' + str(heights) + '_' + str(models)
    else:
        output_name = str(file_out)
    out_file = output_path + output_name
    if not exists(out_file):
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    rand_states = [138, 15, 89, 42, 0]
    for name in data_names:
        data, encoding_map = OU.get_data(name.replace('.csv', ''), 'target')
        for h in heights:
            for i in range(1):
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=rand_states[i])
                calibration_set, test_set = train_test_split(test_set, train_size=0.5, random_state=rand_states[i])
                model_set = pd.concat([train_set, calibration_set])
                unreachable = OSP.fixing(TREE(h), model_set)
                for model in models:
                    print('\n\nDataset: ' + str(name) + ', H: ' + str(h) + ', Iteration: ' + str(
                        i) + '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    tree = TREE(h=h)
                    opt_model = OBCT(data=model_set, tree=tree, target='target', model=model,
                                     time_limit=3600, encoding_map=encoding_map, model_extras=['fixing'],
                                     unreachable=unreachable, warm_start=None, name=name)
                    opt_model.formulation()
                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [name.replace('.csv', ''), h, len(data),
                             opt_model.fixedvars, opt_model.totalvars, opt_model.fixed,
                             model])
                        results.close()
