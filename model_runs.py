#!/usr/bin/python
import os, time, getopt, sys, csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import SPEED_UP
from OBCT import OBCT
from TREE import TREE
import UTILS


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    model_extras = None
    rand_states = None
    file_out = None
    tuning = False
    log_files = None
    weight = 0

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:e:c:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "extras=", "calibration=",
                                    "results_file=", "log_files="])
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
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-e", "--extras"):
            model_extras = arg
        elif opt in ("-c", "--calibration"):
            tuning = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg
    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Tuning_Used', 'Lambda', 'Calibration_Time', 'Rand_State', 'Time_Limit',
                       'Num_CB', 'User_Cuts', 'Total_CB_Time', 'MIP_Node_Time', 'MIP_Sol_Time',
                       'Single_Feature_Use', 'Max_Features', 'Regularization']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_' + str(model_extras) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    # Using logger we log the output of the console in a text file
    # sys.stdout = UTILS.logger(output_path + output_name + '.txt')

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red'
                          'glass', 'image', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets:
            binarization = 'all-candidates'
        else:
            binarization = False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.75, random_state=i)
                cal_set, train_set = train_test_split(train_set, train_size=0.2, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                for modeltype in modeltypes:
                    cb_type = modeltype[5:]
                    if 'CUT' in modeltype and len(cb_type) == 0:
                        cb_type = 'ALL'
                    print('\n' + str(file) + ', H_' + str(h) + ', ' + str(modeltype) + ', Rand_' + str(i)
                          + '. Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
                    # Log .lp and .txt files name
                    WSV = {'use': False, 'time': 0}
                    if tuning:
                        wsm_time_start = time.perf_counter()
                        best_tree, best_acc = {}, 0
                        lambda_WSV = None
                        for cal_lambda in np.linspace(0, .9, 10):
                            # Calibrate model with number of Num-Tree-size features = k for k in [1, B]
                            cal_tree = TREE(h=h)
                            cal_model = OBCT(data=cal_set, tree=cal_tree, target=target, model=modeltype, name=file,
                                             time_limit=0.5 * time_limit, warm_start=lambda_WSV, weight=cal_lambda)
                            cal_model.formulation()
                            if lambda_WSV is not None:
                                cal_model.warm_start()
                            cal_model.model.update()
                            print('test:', round(cal_lambda, 2), str(time.strftime("%I:%M:%S %p", time.localtime())))
                            if 'CF' in modeltype: cal_model.model.optimize()
                            if 'CUT' or 'POKE' in modeltype:
                                if 'GRB' or 'ALL' in cb_type:
                                    cal_model.model.optimize()
                                if 'FRAC' in cb_type:
                                    # User cb.Cut FRAC S-Q cuts
                                    cal_model.model.Params.PreCrush = 1
                                    if '1' in cb_type: cal_model.model.optimize(SPEED_UP.frac1)
                                    if '2' in cb_type: cal_model.model.optimize(SPEED_UP.frac2)
                                    if '3' in cb_type: cal_model.model.optimize(SPEED_UP.frac3)
                            UTILS.node_assign(cal_model, cal_tree)
                            UTILS.tree_check(cal_tree)
                            cal_acc, cal_assign = UTILS.model_acc(tree=cal_tree, target=target, data=cal_set)
                            lambda_WSV = {'tree': cal_tree, 'data': cal_assign,
                                          'time': cal_model.model.RunTime, 'best': False}
                            if cal_acc > best_acc:
                                weight, best_acc, best_tree = cal_lambda, cal_acc, cal_tree
                        cal_time = time.perf_counter() - wsm_time_start
                        model_wsm_acc, model_wsm_assgn = UTILS.model_acc(tree=best_tree, target=target, data=model_set)
                        WSV = {'tree': best_tree, 'data': model_wsm_assgn, 'time': cal_time, 'best': True, 'use': True}
                        print('Tuning time:', cal_time)
                    if log_files:
                        log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) + '_W:' + str(weight) \
                              + '_T:' + str(time_limit) + '_R:' + str(i) + '_E:' + str(model_extras)
                    else:
                        log = False
                    # Generate tree and necessary structure information
                    tree = TREE(h=h)
                    # Model with 75% training set, applicable model extras and optimization time limit
                    opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype, weight=weight,
                                     time_limit=time_limit, warm_start=WSV, name=file, log=log)
                    # Add connectivity constraints according to model type
                    opt_model.formulation()
                    # Update with warm start values if applicable
                    if WSV['use']: opt_model.warm_start()
                    # Add any model extras if applicable
                    if model_extras is not None: opt_model.extras()
                    opt_model.model.update()
                    # Optimize model with callback if applicable
                    print(f"Optimizing full model w/ lamda: {weight}. Start:",
                          str(time.strftime("%I:%M:%S %p", time.localtime())))
                    if 'CF' in modeltype: opt_model.model.optimize()
                    if 'CUT' or 'POKE' in modeltype:
                        if 'GRB' or 'ALL' in cb_type:
                            opt_model.model.optimize()
                        if 'FRAC' in cb_type:
                            # User cb.Cut FRAC S-Q cuts
                            opt_model.model.Params.PreCrush = 1
                            if '1' in cb_type: opt_model.model.optimize(SPEED_UP.frac1)
                            if '2' in cb_type: opt_model.model.optimize(SPEED_UP.frac2)
                            if '3' in cb_type: opt_model.model.optimize(SPEED_UP.frac3)
                        """
                        if 'BOTH' in cb_type:
                            # User cb.Lazy FRAC and INT S-Q cuts
                            opt_model.model.Params.LazyConstraints = 1
                            opt_model.model.Params.PreCrush = 1
                            opt_model.model.optimize(CALLBACKS.both)
                        if 'INT' in cb_type:
                            # User cb.Lazy INT S-Q cuts
                            opt_model.model.Params.LazyConstraints = 1
                            if '1' in cb_type: opt_model.model.optimize(CALLBACKS.int1)
                            if '2' in cb_type: opt_model.model.optimize(CALLBACKS.int2)
                            if '3' in cb_type: opt_model.model.optimize(CALLBACKS.int3)
                        """

                    if opt_model.model.Runtime < time_limit:
                        print(f'Optimal solution found in {round(opt_model.model.Runtime, 4)}s '
                              f'({time.strftime("%I:%M:%S %p", time.localtime())})')
                    else:
                        print(f'Time limit reached. ({time.strftime("%I:%M:%S %p", time.localtime())})')

                    if opt_model.model._numcb > 0:
                        opt_model.model._avgcuts = opt_model.model._numcuts / opt_model.model._numcb
                    else:
                        opt_model.model._avgcuts = 0
                    # UTILS.dv_results(opt_model.model, tree, opt_model.features, opt_model.classes, opt_model.datapoints)

                    # Generate model performance metrics and save to .csv file in .../results_files/
                    UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                        rand_state=i, results_file=out_file, data_name=str(file))
                    # Write LP file
                    if log_files: opt_model.model.write(log + '.lp')


def multiobj(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    priorities = None
    rand_states = None
    file_out = None
    log_files = None
    model_extras = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:p:e:f:l:",
                                   ["data_files=", "heights=", "timelimit=", "models=",
                                    "rand_states=", "priority=", "extras=", "results_file=", "log_files"])
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
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-p", "--priority"):
            priorities = arg
        elif opt in ("-e", "--extras"):
            model_extras = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', 'Calibration', 'Single_Feature_Use', 'Max_Features']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/biobj_log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    # Using logger we log the output of the console in a text file
    # sys.stdout = UTILS.logger(output_path + output_name + '.txt')

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red'
                          'glass', 'image', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets:
            binarization = 'all-candidates'
        else:
            binarization = False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                print('\nDataset: ' + str(file) + ', H: ' + str(h) + ', ' 'Rand State: ' + str(i)
                      + '. Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.75, random_state=i)
                cal_set, train_set = train_test_split(train_set, train_size=0.2, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                WSV, unreachable = None, {'data': {}, 'tree': {}}
                for modeltype in modeltypes:
                    for priority in priorities:
                        # Log .lp and .txt files name
                        if log_files:
                            log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) \
                                  + '_T:' + str(time_limit) + '_R:' + str(i) + '_E:' + str(priority)
                        else:
                            log = False
                        # Calibrate number of maximum Num-Tree-size nodes using pareto frontier and warm starts
                        # Generate tree and necessary structure information
                        tree = TREE(h=h)
                        # Model with 75% training set, applicable model extras and optimization time limit
                        opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype + '-biobj_data',
                                         time_limit=time_limit, warm_start=WSV,
                                         name=file, log=log, priority=priority)
                        # Add connectivity constraints according to model type
                        opt_model.formulation()
                        # Add any model extras if applicable
                        if model_extras is not None: opt_model.extras()
                        # Update with warm start values if applicable
                        if WSV is not None: opt_model.warm_start()
                        # Optimize model with callback if applicable
                        opt_model.model.update()
                        if priority is not None:
                            opt_model.model.optimize(SPEED_UP.biobj)
                        else:
                            opt_model.model.optimize()
                        # Generate model performance metrics and save to .csv file in .../results_files/
                        UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                            rand_state=i, results_file=out_file, data_name=str(file))
                        # Write LP file
                        # if log_files: opt_model.model.write(log+'.lp')


def pareto(argv):
    print(argv)
    data_files = None
    height = None
    time_limit = None
    modeltypes = None
    rand_states = None
    file_out = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:",
                                   ["data_files=", "height=", "timelimit=",
                                    "models=", "rand_states=", "results_file="])
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
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps']
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
    print(list(range(1, 2 ** int(height))))
    for file in data_files:
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''))
        for i in rand_states:
            print(
                '\n\nDataset: ' + str(file) + ', H: ' + str(height) + ', Iteration: ' + str(i) + '. Run Start: ' + str(
                    time.strftime("%I:%M %p", time.localtime())))
            train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
            for modeltype in modeltypes:
                WSV = None
                for num_features in list(range(1, 2 ** int(height))):
                    extras = [f'num_features-{num_features}']
                    tree = TREE(h=height)
                    opt_model = OBCT(data=train_set, tree=tree, target=target, model=modeltype,
                                     time_limit=time_limit, model_extras=extras, warm_start=WSV, name=file)
                    opt_model.formulation()
                    if num_features != 1:
                        opt_model.warm_start()
                    opt_model.extras()
                    opt_model.model.update()
                    opt_model.model.optmize()
                    UTILS.node_assign(opt_model, tree)
                    UTILS.tree_check(tree)
                    UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                        rand_state=i, results_file=out_file, data_name=str(file))
                    model_wsm_acc, model_wsm_assgn = UTILS.model_acc(tree=tree, target=target, data=train_set)
                    WSV = {'tree': tree, 'data': model_wsm_assgn}
        # Generate pareto plot of models using run averages of pareto.csv file
        pareto_data = pd.pareto_data = pd.read_csv(os.getcwd() + '/results_files/pareto_test.csv', na_values='?')
        file_data = pareto_data[pareto_data['Data'] == file.replace('.csv', '')]
        frontier_avg = pd.DataFrame(columns=summary_columns)
        for model in ['FlowOCT', 'SCF', 'MCF', 'POKE', 'CUT']:
            sub_data = file_data.loc[file_data['Model'] == model]
            for feature in sub_data['Max_Features'].unique():
                subsub_data = sub_data.loc[sub_data['Max_Features'] == feature]
                frontier_avg = frontier_avg.append({
                    'Data': file.replace('.csv', ''), 'H': int(subsub_data['H'].mean()), '|I|': int(subsub_data['|I|'].mean()),
                    'Out_Acc': 100 * subsub_data['Out_Acc'].mean(), 'In_Acc': 100 * subsub_data['In_Acc'].mean(),
                    'Sol_Time': subsub_data['Sol_Time'].mean(), 'MIP_Gap': 100 * subsub_data['MIP_Gap'].mean(),
                    'Obj_Val': 100 * subsub_data['Obj_Val'].mean(), 'Obj_Bound': subsub_data['Obj_Bound'].mean(),
                    'Model': model, 'Num_CB': subsub_data['Num_CB'].mean(), 'User_Cuts': subsub_data['User_Cuts'].mean(),
                    'Total_CB_Time': subsub_data['Total_CB_Time'].mean(), 'CB_Eps': subsub_data['CB_Eps'].mean()
                }, ignore_index=True)
        for plot_type in ['out_acc', 'in_acc', 'time']:
            UTILS.pareto_plot(frontier_avg, type=plot_type)


if __name__ == "__main__":
    main(sys.argv[1:])

if __name__ == "__multiobj__":
    main(sys.argv[1:])

if __name__ == "__pareto__":
    main(sys.argv[1:])