#!/usr/bin/python
import os
import time
import pandas as pd
import getopt
import sys
import csv
from sklearn.model_selection import train_test_split
from sklearn import tree as HEURTree
from OBCT import OBCT
from TREE import TREE
import UTILS as OU
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
    rand_states = None
    file_out = None
    tuning = None
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:e:c:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "extras=", "calibration=",
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
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', 'Calibration', 'Single_Feature_Use', 'Max_Features']
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
    # sys.stdout = OU.logger(output_path + output_name + '.txt')

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'

    for file in data_files:
        # pull dataset to train model with
        data = OU.get_data(file.replace('.csv', ''))

        for h in heights:
            for i in rand_states:
                print('\nDataset: '+str(file)+', H: '+str(h)+', '
                      'Rand State: '+str(i)+'. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                WSV, unreachable = None, {'data': {}, 'tree': {}}
                for modeltype in modeltypes:
                    # Log .lp and .txt files name
                    log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) + '_' + 'T:' + str(
                        time_limit) + '_E:' + str(model_extras)
                    if any([char.isdigit() for char in modeltype]):
                        # Calibrate number of maximum branching nodes using pareto frontier and warm starts
                        # Use a 25% calibration set for process
                        if tuning:
                            print('Calibrating number of maximum branching features')
                            best_tree, best_feats, best_acc = {}, 0, 0
                            wsm = None
                            for num_features in range(1, 2 ** h):
                                # Calibrate model with number of branching features = k for k in [1, B]
                                extras = [f'num_features-{num_features}']
                                cal_tree = TREE(h=h)
                                cal_model = OBCT(data=cal_set, tree=cal_tree, target=target, model=modeltype,
                                                 time_limit=time_limit, warm_start=wsm, model_extras=extras,  name=file)
                                cal_model.formulation()
                                cal_model.model.update()
                                cal_model.extras()
                                cal_model.optimization()
                                OU.node_assign(cal_model, cal_tree)
                                OU.tree_check(cal_tree)
                                cal_acc, cal_assign = OU.model_acc(tree=cal_tree, target=target, data=cal_set)
                                wsm = {'tree': cal_tree, 'data': cal_assign}
                                if cal_acc > best_acc:
                                    best_feats, best_acc = num_features, cal_acc
                                    best_tree = cal_tree
                                model_wsm_acc, model_wsm_assgn = OU.model_acc(tree=best_tree, target=target,
                                                                              data=model_set)
                                WSV = {'tree': best_tree, 'data': model_wsm_assgn}
                            # Assign calibrated number of branching nodes as model extra
                            if model_extras is not None:
                                if any((match := elem).startswith('max_features') for elem in model_extras):
                                    model_extras[model_extras.index(match)] = 'max_features-' + str(best_feats)
                                else: model_extras.append('max_features-'+str(best_feats))
                            else: model_extras = ['max_features-'+str(best_feats)]
                        # Generate tree and necessary structure information
                        tree = TREE(h=h)
                        # Model with 75% training set, applicable model extras and optimization time limit
                        opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype,
                                         time_limit=time_limit, model_extras=model_extras, warm_start=WSV,
                                         name=file, log=log+'.txt')
                        # Add connectivity constraints according to model type
                        opt_model.formulation()
                        # Add any model extras if applicable
                        if model_extras is not None:
                            opt_model.extras()
                        # Update with warm start values if applicable
                        if WSV is not None:
                            opt_model.warm_start()
                        # Optimize model with callback if applicable
                        opt_model.model.update()
                        opt_model.optimization()
                        # Generate model performance metrics and save to .csv file in .../results_files/
                        OU.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                         rand_state=i, results_file=out_file)
                        # Write LP file
                        if log_files:
                            opt_model.model.write(log+'.lp')

                    elif 'OCT' in modeltype:
                        OCT_tree = FlowOCTTree(d=h)
                        # FlowOCT model
                        if 'Flow' in modeltype:
                            print('Model: FlowOCT')
                            primal = FlowOCT(data=model_set, label=target, tree=OCT_tree,
                                             _lambda=0, time_limit=time_limit, mode='classification')
                            if log_files:
                                primal.model.Params.LogFile = log + '.txt'
                            primal.create_primal_problem()
                            primal.model.update()
                            print('Optimizing Model')
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
                                     0, 0, 0, 0, 0, 0, 0, time_limit, i, False, False, False])
                                results.close()
                            if log_files:
                                primal.model.write(log+'.lp')
                        # BendersOCT model
                        elif 'Benders' in modeltype:
                            print('Model: BendersOCT')
                            master = BendersOCT(data=model_set, label=target, tree=OCT_tree,
                                                _lambda=0, time_limit=time_limit, mode='classification')
                            if log_files:
                                master.model.Params.LogFile = log + '.txt'
                            master.create_master_problem()
                            master.model.update()
                            print('Optimizing Model')
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
                                     0, 0, 0, 0, 0, 0, 0, time_limit, i, False, False, False])
                                results.close()
                            if log_files:
                                master.model.write(log+'.lp')

                    elif 'CART' in modeltype:
                        if '_enc' in file:
                            file.replace('_enc', '')
                            cart_data = OU.get_data(file.replace('.csv', ''))
                        else: cart_data = data
                        cart_train_set, cart_test_set = train_test_split(cart_data, train_size=0.5, random_state=i)
                        cart_cal_set, cart_test_set = train_test_split(cart_test_set, train_size=0.5, random_state=i)
                        cart_model_set = pd.concat([cart_train_set, cart_cal_set])
                        cart_X_train, cart_Y_train = cart_model_set.drop('target', axis=1), cart_model_set['target']
                        cart_X_test, cart_Y_test = test_set.drop('target', axis=1), test_set['target']
                        # HEURISTIC
                        if 'STR' in modeltype:
                            print('Model: CART (Structured Tree)')
                            cart_tree = HEURTree.DecisionTreeClassifier(criterion='gini', max_depth=h,
                                                                        max_leaf_nodes=2**h, max_features=1)
                        else:
                            cart_tree = HEURTree.DecisionTreeClassifier(criterion='gini')
                        start = time.time()
                        cart_tree.fit(cart_X_train, cart_Y_train)
                        cart_train_acc = cart_tree.score(cart_X_train, cart_Y_train)
                        cart_test_acc = cart_tree.score(cart_X_test, cart_Y_test)
                        cart_time = (time.time() - start)
                        with open(out_file, mode='a') as results:
                            results_writer = csv.writer(results, delimiter=',', quotechar='"')
                            results_writer.writerow(
                                [file.replace('.csv', ''), h, len(cart_model_set),
                                 cart_test_acc, cart_train_acc, cart_time,
                                 'None', 'None', 'None', modeltype,
                                 0, 0, 0, 0, 0, 0, 0, time_limit, i, False, False, False])
                            results.close()


if __name__ == "__main__":
    main(sys.argv[1:])
