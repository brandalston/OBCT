from Benchmark_Methods.FlowOCTmain.FlowOCTTree import Tree as OCT_Tree
from Benchmark_Methods.FlowOCTmain.FlowOCT import FlowOCT
import Benchmark_Methods.FlowOCTmain.FlowOCTutils as FlowOCTutils
from Benchmark_Methods.FlowOCTmain.BendersOCT import BendersOCT
from sklearn.model_selection import train_test_split
import os, time, getopt, sys, csv
import pandas as pd
import numpy as np
import UTILS


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    file_out = None
    log_files = None
    tuning = False

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:c:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "calibration=",
                                    "results_file=", "log_files"])
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
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg
        elif opt in ("-c", "--tuning"):
            tuning = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Lambda', 'Time_Limit', 'Calibration_Time', 'Rand_State', 'Tuning_Used',
                       'Num_CB', 'User_Cuts', 'Total_CB_Time']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
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
    # sys.stdout = OU.logger(output_path + output_name + '.txt')

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red',
                          'glass', 'image_segmentation','ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets: binarization = 'all-candidates'
        else: binarization = False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.75, random_state=i)
                cal_set, train_set = train_test_split(train_set, train_size=0.2, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                # Log files
                for modeltype in modeltypes:
                    print('\n'+str(file) + ', H_' + str(h) + ', ' + str(modeltype) + 'OCT, Rand_' + str(i)
                          + ', Tuning_'+str(tuning)+'. Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
                    # Log .lp and .txt files name
                    # FlowOCT model
                    if 'Flow' in modeltype:
                        weight, cal_time, best_cal_model = 0, 0, None
                        if tuning:
                            wsm_time_start = time.perf_counter()
                            best_tree, best_acc, lambda_WSV = {}, 0, None
                            for cal_lambda in np.linspace(0, .9, 10):
                                # Calibrate model with number of Num-Tree-size features = k for k in [1, B]
                                cal_tree = OCT_Tree(d=h)
                                cal_model = FlowOCT(data=model_set, label=target, tree=cal_tree, _lambda=cal_lambda,
                                                    time_limit=0.5*time_limit, mode='classification')
                                cal_model.create_primal_problem()
                                if lambda_WSV is not None:
                                    ws_branching, ws_leaf, ws_pruned = FlowOCTutils.get_tree_topology(lambda_WSV)
                                    cal_model.warm_start(ws_branching, ws_leaf, ws_pruned)
                                cal_model.model.update()
                                print('test:', round(cal_lambda,2), str(time.strftime("%I:%M:%S %p", time.localtime())))
                                cal_model.model.optimize()
                                cal_acc = FlowOCTutils.get_acc(cal_model, train_set,
                                                               cal_model.model.getAttr("X", cal_model.b),
                                                               cal_model.model.getAttr("X", cal_model.beta),
                                                               cal_model.model.getAttr("X", cal_model.p))
                                lambda_WSV = {'model': cal_model, 'b': cal_model.model.getAttr("X", cal_model.b),
                                              'beta': cal_model.model.getAttr("X", cal_model.beta),
                                              'p': cal_model.model.getAttr("X", cal_model.p), 'cal_acc': cal_acc}
                                if cal_acc > best_acc:
                                    weight, best_acc, best_cal_model = cal_lambda, cal_acc, cal_model
                            cal_time = time.perf_counter() - wsm_time_start
                            best_WSV = {'model': best_cal_model, 'b': best_cal_model.model.getAttr("X", best_cal_model.b),
                                        'beta': best_cal_model.model.getAttr("X", best_cal_model.beta),
                                        'p': best_cal_model.model.getAttr("X", best_cal_model.p)}
                            best_ws_branching, best_ws_leaf, best_ws_pruned = FlowOCTutils.get_tree_topology(best_WSV)
                            print(f'Total tuning time {round(cal_time,4)}')
                        OCT_tree = OCT_Tree(d=h)
                        primal = FlowOCT(data=model_set, label=target, tree=OCT_tree, _lambda=weight,
                                         time_limit=time_limit, mode='classification')
                        primal.create_primal_problem()
                        if tuning: primal.warm_start(best_ws_branching, best_ws_leaf, best_ws_pruned)
                        primal.model.update()
                        print(f'Optimizing Model w/ lambda: {weight}. Start:',str(time.strftime("%I:%M:%S %p", time.localtime())))
                        primal.model.optimize()
                        if primal.model.RunTime < time_limit:
                            print('Optimal solution found in ' + str(round(primal.model.Runtime, 2)) + 's. (' + str(
                                time.strftime("%I:%M:%S %p", time.localtime())) + ')\n')
                        else:
                            print('Time limit reached. (', time.strftime("%I:%M:%S %p", time.localtime()), ')\n')
                        b_value = primal.model.getAttr("X", primal.b)
                        beta_value = primal.model.getAttr("X", primal.beta)
                        p_value = primal.model.getAttr("X", primal.p)
                        train_acc = FlowOCTutils.get_acc(primal, train_set, b_value, beta_value, p_value)
                        test_acc = FlowOCTutils.get_acc(primal, test_set, b_value, beta_value, p_value)
                        with open(out_file, mode='a') as results:
                            results_writer = csv.writer(results, delimiter=',', quotechar='"')
                            results_writer.writerow(
                                [file.replace('.csv', ''), h, len(model_set),
                                 test_acc, train_acc, primal.model.Runtime,
                                 primal.model.MIPGap, primal.model.ObjVal, primal.model.ObjBound, 'FlowOCT',
                                 tuning, weight, cal_time, i, time_limit])
                            results.close()
                        # if log_files: primal.model.write(log + '.lp')
                    # BendersOCT model
                    elif 'Benders' in modeltype:
                        weight, cal_time, best_cal_model = 0, 0, None
                        if tuning:
                            wsm_time_start = time.perf_counter()
                            best_tree, best_acc, lambda_WSV = {}, 0, None
                            for cal_lambda in np.linspace(0, .9, 10):
                                # Calibrate model with number of Num-Tree-size features = k for k in [1, B]
                                cal_tree = OCT_Tree(d=h)
                                cal_model = BendersOCT(data=model_set, label=target, tree=cal_tree, _lambda=cal_lambda,
                                                       time_limit=0.5*time_limit, mode='classification')
                                cal_model.create_master_problem()
                                if lambda_WSV is not None:
                                    ws_branching, ws_leaf, ws_pruned = FlowOCTutils.get_tree_topology(lambda_WSV)
                                    cal_model.warm_start(ws_branching, ws_leaf, ws_pruned)
                                cal_model.model.update()
                                print('test:', round(cal_lambda,2), str(time.strftime("%I:%M:%S %p", time.localtime())))
                                cal_model.model.optimize(FlowOCTutils.mycallback)
                                cal_acc = FlowOCTutils.get_acc(cal_model, train_set,
                                                               cal_model.model.getAttr("X", cal_model.b),
                                                               cal_model.model.getAttr("X", cal_model.beta),
                                                               cal_model.model.getAttr("X", cal_model.p))
                                lambda_WSV = {'model': cal_model, 'b': cal_model.model.getAttr("X", cal_model.b),
                                              'beta': cal_model.model.getAttr("X", cal_model.beta),
                                              'p': cal_model.model.getAttr("X", cal_model.p), 'cal_acc': cal_acc, 'cal_lambda': cal_model._lambda}
                                if cal_acc > best_acc:
                                    weight, best_acc, best_cal_model = cal_lambda, cal_acc, cal_model
                            cal_time = time.perf_counter() - wsm_time_start
                            best_WSV = {'model': best_cal_model,
                                        'b': best_cal_model.model.getAttr("X", best_cal_model.b),
                                        'beta': best_cal_model.model.getAttr("X", best_cal_model.beta),
                                        'p': best_cal_model.model.getAttr("X", best_cal_model.p)}
                            best_ws_branching, best_ws_leaf, best_ws_pruned = FlowOCTutils.get_tree_topology(best_WSV)
                            print(f'Total tuning time {round(cal_time,4)}')
                        OCT_tree = OCT_Tree(d=h)
                        master = BendersOCT(data=model_set, label=target, tree=OCT_tree, _lambda=weight,
                                            time_limit=time_limit, mode='classification')
                        master.create_master_problem()
                        if tuning: master.warm_start(best_ws_branching, best_ws_leaf, best_ws_pruned)
                        master.model.update()
                        print(f'Optimizing Model w/ lambda: {weight}. Start:', str(time.strftime("%I:%M:%S %p", time.localtime())))
                        master.model.optimize(FlowOCTutils.mycallback)
                        if master.model.RunTime < time_limit:
                            print('Optimal solution found in ' + str(round(master.model.Runtime, 2)) + 's. (' + str(
                                time.strftime("%I:%M:%S %p", time.localtime())) + ')\n')
                        else:
                            print('Time limit reached. (', time.strftime("%I:%M:%S %p", time.localtime()), ')\n')
                        b_value = master.model.getAttr("X", master.b)
                        beta_value = master.model.getAttr("X", master.beta)
                        p_value = master.model.getAttr("X", master.p)
                        train_acc = FlowOCTutils.get_acc(master, train_set, b_value, beta_value, p_value)
                        test_acc = FlowOCTutils.get_acc(master, test_set, b_value, beta_value, p_value)
                        with open(out_file, mode='a') as results:
                            results_writer = csv.writer(results, delimiter=',', quotechar='"')
                            results_writer.writerow(
                                [file.replace('.csv', ''), h, len(model_set),
                                 test_acc, train_acc, master.model.Runtime,
                                 master.model.MIPGap, master.model.ObjVal, master.model.ObjBound, 'BendersOCT',
                                 tuning, weight, cal_time, i, time_limit,
                                 master.model._callback_counter_integer, master.model._callback_counter_integer_success,
                                 master.model._total_callback_time_integer])
                            results.close()
                        # if log_files: master.model.write(log + '.lp')
