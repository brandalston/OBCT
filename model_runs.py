#!/usr/bin/python
from sklearn.model_selection import train_test_split
import os, time, getopt, sys, csv
import pandas as pd
from OBCT import OBCT
from TREE import TREE
import UTILS as OU


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
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                                                                                'glass', 'image_segmentation',
                          'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets: binarization = 'all-candidates'
        else: binarization = False
        # pull dataset to train model with
        data = OU.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                print('\nDataset: '+str(file)+', H: '+str(h)+', ' 'Rand State: '+str(i)
                      + '. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                WSV, unreachable = None, {'data': {}, 'tree': {}}
                for modeltype in modeltypes:
                    # Log .lp and .txt files name
                    if log_files: log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype)\
                                        + '_' + 'T:' + str(time_limit) + '_E:' + str(model_extras)
                    else: log = False
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
                            cal_model.extras()
                            cal_model.model.update()
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
                                     name=file, log=log)
                    # Add connectivity constraints according to model type
                    opt_model.formulation()
                    # Add any model extras if applicable
                    if model_extras is not None: opt_model.extras()
                    # Update with warm start values if applicable
                    if WSV is not None: opt_model.warm_start()
                    # Optimize model with callback if applicable
                    opt_model.model.update()
                    opt_model.optimization()
                    # Generate model performance metrics and save to .csv file in .../results_files/
                    OU.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                     rand_state=i, results_file=out_file)
                    # Write LP file
                    if log_files:
                        opt_model.model.write(log+'.lp')


if __name__ == "__main__":
    main(sys.argv[1:])
