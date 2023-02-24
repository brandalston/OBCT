from FlowOCTTree import Tree as OCT_Tree
from FlowOCT import FlowOCT
import FlowOCTutils
from BendersOCT import BendersOCT
from sklearn.model_selection import train_test_split
import os, time, getopt, sys, csv
import pandas as pd
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
        elif opt in ("-m", "--models"):
            modeltypes = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
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
                OCT_tree = OCT_Tree(d=h)
                # Log files
                for modeltype in modeltypes:
                    # Log .lp and .txt files name
                    if log_files:
                        log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) \
                              + '_' + 'T:' + str(time_limit) + '_E:' + str(model_extras)
                    else:
                        log = False
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
                        if primal.model.RunTime < time_limit:
                            print('Optimal solution found in ' + str(round(primal.model.Runtime, 2)) + 's. (' + str(
                                time.strftime("%I:%M %p", time.localtime())) + ')\n')
                        else:
                            print('Time limit reached. (', time.strftime("%I:%M %p", time.localtime()), ')\n')
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
                                 primal.model.MIPGap, primal.model.ObjBound, primal.model.ObjVal, modeltype,
                                 0, 0, 0, 0, 0, 0, 0, time_limit, i, False, False, False])
                            results.close()
                        if log_files:
                            primal.model.write(log + '.lp')
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
                        if master.model.RunTime < time_limit:
                            print('Optimal solution found in ' + str(round(master.model.Runtime, 2)) + 's. (' + str(
                                time.strftime("%I:%M %p", time.localtime())) + ')\n')
                        else:
                            print('Time limit reached. (', time.strftime("%I:%M %p", time.localtime()), ')\n')
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
                                 master.model.MIPGap, master.model.ObjBound, master.model.ObjVal, modeltype,
                                 0, 0, 0, 0, 0, 0, 0, time_limit, i, False, False, False])
                            results.close()
                        if log_files:
                            master.model.write(log + '.lp')
