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
from FlowOCTTree import Tree as FlowOCTTree
from FlowOCT import FlowOCT
import FlowOCTutils
from BendersOCT import BendersOCT


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    model_extras = None
    repeats = None
    file_out = None
    tuning = None

    rand_states = [138, 15, 89, 42, 0]

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:e:r:f:c:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "extras=", "repeats=", "results_file=", "tuning="])
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
        elif opt in ("-c", "--tuning"):
            tuning = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time', 'MIP_Gap', 'Obj_Bound', 'Obj_Val', 'Model',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB', 'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', '%_Fixed', 'Calibration', 'CC',
                       'Single_Feature_Use', 'Level_Tree', 'Max_Features', 'Super_Feature']
    output_path = os.getcwd() + '/results_files/'
    fig_path = os.getcwd() + '/results_figures/'

    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_' + str(model_extras) + '.csv'
    else:
        output_name = str(file_out)
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
                print('\n\nDataset: '+str(file)+', H: '+str(h)+','
                      'Iteration: '+str(i)+'. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=rand_states[i])
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=rand_states[i])
                model_set = pd.concat([train_set, cal_set])
                WSV, unreachable = None, {'data': {}, 'tree': {}}
                if model_extras is not None:
                    # Fix DV of models if applicable
                    if 'fixing' in model_extras:
                        unreachable = OSP.fixing(TREE(h), model_set)
                for modeltype in modeltypes:
                    if 'OCT' not in modeltype:
                        # Calibrate number of maximum branching nodes using pareto frontier and warm starts
                        # Use a 25% calibration set for process
                        if 'calibration' == tuning:
                            print('Calibrating number of maximum branching features')
                            best_tree, best_feats, best_acc = {}, 0, 0
                            wsm = None
                            for num_features in range(1, 2 ** h):
                                # Solve model with number of branching features = k for k in [1, B]
                                extras = [f'num_features-{num_features}']
                                cal_tree = TREE(h=h)
                                cal_model = OBCT(data=cal_set, tree=cal_tree, target=target, model=modeltype,
                                                 time_limit=time_limit, encoding_map=encoding_map, warm_start=wsm,
                                                 model_extras=extras, unreachable=unreachable,  name=file)
                                cal_model.formulation()
                                cal_model.model.update()
                                cal_model.extras()
                                cal_model.optimization()
                                OR.node_assign(cal_model, cal_tree)
                                OR.tree_check(cal_tree)
                                cal_acc, cal_assign = OR.model_acc(tree=cal_tree, target=target, data=cal_set)
                                wsm = {'tree': cal_tree.DG_prime.nodes(data=True), 'data': cal_assign}
                                if cal_acc > best_acc:
                                    best_feats, best_acc = num_features, cal_acc
                                    best_tree = cal_tree
                                model_wsm_acc, model_wsm_assgn = OR.model_acc(tree=best_tree, target=target,
                                                                              data=model_set)
                                WSV = {'tree': best_tree.DG_prime.nodes(data=True), 'data': model_wsm_assgn}
                            # Assign calibrated number of branching nodes as model extra
                            if model_extras is not None:
                                if any((match := elem).startswith('max_features') for elem in model_extras):
                                    model_extras[model_extras.index(match)] = 'max_features-' + str(best_feats)
                                else: model_extras.append('max_features-'+str(best_feats))
                            else: model_extras = ['max_features-'+str(best_feats)]
                        # Warm start model using the best of random path and level trees
                        elif 'warm_start' == tuning:
                            print('Generating warm start values')
                            cal_tree = TREE(h=h)
                            WSV = OU.random_tree(target=target, data=data, tree=cal_tree)
                        # Generate tree and necessary structure information
                        tree = TREE(h=h)
                        # Model with 75% training set, applicable model extras and optimization time limit
                        opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype,
                                         time_limit=time_limit, encoding_map=encoding_map, model_extras=model_extras,
                                         unreachable=unreachable, warm_start=WSV, name=file)
                        # Add connectivity constraints according to model type
                        opt_model.formulation()
                        # Add any model extras if applicable
                        if model_extras is not None:
                            opt_model.extras()
                        # Optimize model with callback if applicable
                        opt_model.model.update()
                        opt_model.optimization()
                        # Generate model performance metrics and save to .csv file in .../results_files/
                        # Generate .png figure of assigned tree if applicable and save in .../results_figures/
                        fig_file = fig_path + str(file) + '_H:' + str(h) + '_' + str(modeltype) + '_T:' + str(
                            time_limit) + '_' + str(model_extras) + '.png'
                        OR.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                         rand_state=rand_states[i], results_file=out_file, fig_file=fig_file)
                        # Uncomment to write consol log .txt file to .../results_files/ folder
                        consol_log_file = output_path+'_'+str(file)+'_'+str(h)+'_'+str(modeltype)+'_'+'T:'+str(
                                          time_limit)+'_'+str(model_extras)+'.txt'
                        sys.stdout = OU.consol_log(consol_log_file)
                    else:
                        OCT_tree = FlowOCTTree(d=h)
                        # FlowOCT model
                        if 'Flow' in modeltype:
                            print('Model: FlowOCT')
                            primal = FlowOCT(data=model_set, label=target, tree=OCT_tree,
                                             _lambda=0, time_limit=time_limit, mode='classification')
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
                                    [file.replace('.csv', ''), h, len(model_set),
                                     test_acc, train_acc, primal.model.Runtime,
                                     primal.model.MIPGap, primal.model.ObjBound, primal.model.ObjVal, modeltype,
                                     0, 0, 0, 0, 0, 0, 0, time_limit, rand_states[i],
                                     0, False, False, False, 'None', 'None', False])
                                results.close()
                        # BendersOCT model
                        elif 'Benders' in modeltype:
                            print('Model: BendersOCT')
                            master = BendersOCT(data=model_set, label=target, tree=OCT_tree,
                                                _lambda=0, time_limit=time_limit, mode='classification')
                            master.create_master_problem()
                            master.model.update()
                            master.model.optimize(FlowOCTutils.mycallback)
                            b_value = master.model.getAttr("X", master.b)
                            beta_value = master.model.getAttr("X", master.beta)
                            p_value = master.model.getAttr("X", master.p)
                            train_acc = FlowOCTutils.get_acc(master, train_set, b_value, beta_value, p_value)
                            test_acc = FlowOCTutils.get_acc(master, test_set, b_value, beta_value, p_value)
                            if master.model.RunTime < time_limit:
                                print('Optimal solution found in ' + str(round(master.model.Runtime, 2)) + 's. (' + str(
                                    time.strftime("%I:%M %p", time.localtime())) + ')\n')
                            else:
                                print('Time limit reached. (', time.strftime("%I:%M %p", time.localtime()), ')\n')
                            with open(out_file, mode='a') as results:
                                results_writer = csv.writer(results, delimiter=',', quotechar='"')
                                results_writer.writerow(
                                    [file.replace('.csv', ''), h, len(model_set),
                                     test_acc, train_acc, master.model.Runtime,
                                     master.model.MIPGap, master.model.ObjBound, master.model.ObjVal, modeltype,
                                     0, 0, 0, 0, 0, 0, 0, time_limit, rand_states[i],
                                     0, False, False, False, 'None', 'None', False])
                                results.close()


if __name__ == "__main__":
    main(sys.argv[1:])
