#!/usr/bin/python
import os, time, getopt, sys, csv
import pandas as pd
from sklearn.model_selection import train_test_split
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
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:e:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "extras=",
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
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Time_Limit', 'Rand_State', 'Warm_Start_Use', 'Warm_Start_Time',
                       'CB_Eps', 'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'FRAC_CB_Time', 'INT_CB_Time',
                       'Max_Features', 'Single_Features', 'Regularization']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = f'{data_files}_H_{heights}_M_{modeltypes}_T_{time_limit}.csv'
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
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red',
                          'glass', 'image', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets: binarization = 'all-candidates'
        else: binarization = False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                print(f'\nDataset: {file}, H: {h}, Seed: {i}, Run Start: {time.strftime("%I:%M:%S %p", time.localtime())}')
                # 3:1 train test split data
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set, WSV = pd.concat([train_set, cal_set]), None
                for modeltype in modeltypes:
                    # Log .lp and .txt files name
                    if log_files: log = f'{log_path}{file}_h_{h}_m_{modeltype}_t_{time_limit}_r_{i}_e_{model_extras}'
                    else: log = False

                    # Generate tree and necessary structure information
                    tree = TREE(h=h)
                    # Model with 75% training set, applicable model extras and optimization time limit
                    opt_model = OBCT(data=model_set, tree=tree, target=target, model=modeltype,
                                     time_limit=time_limit, model_extras=model_extras, warm_start=WSV, log=log)
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
                    UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                        rand_state=i, results_file=out_file, data_name=file)
                    # Write LP file
                    if log_files: opt_model.model.write(log + '.lp')


def biobjective(argv):
    print(argv)
    data_files = None
    height = None
    time_limit = None
    modeltypes = None
    priorities = None
    rand_states = None
    file_out = None
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:p:f:l:",
                                   ["data_files=", "height=", "timelimit=",
                                    "models=", "rand_states=", "priority=",
                                    "results_file=", "log_files"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            height = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-m", "--model"):
            modeltypes = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-p", "--priority"):
            priorities = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc',
                       'Sum_S', 'Obj_Val', 'Num_Branch_Nodes', 'Sol_Num', 'Priority', 'Sol_Time',
                       'Model', 'Time_Limit', 'Rand_State', 'Warm_Start', 'Warm_Start_Time',
                       'CB_Eps', 'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'FRAC_CB_Time', 'INT_CB_Time']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = f'{data_files}_h_{height}_m_{modeltypes}_p_{priorities}_t_{time_limit}.csv'
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
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red',
                          'glass', 'image', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']

    for file in data_files:
        binarization = 'all-candidates' if file in numerical_datasets else False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=binarization)
        for i in rand_states:
            # 3:1 train test split data
            train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
            cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
            model_set = pd.concat([train_set, cal_set])
            for modeltype in modeltypes:
                for priority in priorities:
                    print(f'\nDataset: {file}, H: {height}, Priority: {priority}, Seed: {i}, '
                          f'Run Start: {time.strftime("%I:%M:%S %p", time.localtime())}')
                    # Log .lp and .txt files name
                    if log_files:
                        log = f'{log_path}file_H_{height}_M_{modeltype}_P_{priority}_T_{time_limit}_R_{i}'
                    else:
                        log = False
                    # Generate tree and necessary structure information
                    tree = TREE(h=height)
                    # Bi-objectiveModel with priority
                    opt_model = OBCT(data=model_set, tree=tree, target=target, time_limit=time_limit,
                                     model=modeltype + '-biobj', priority=priority, log=log)
                    # Add connectivity constraints according to model type
                    opt_model.formulation()
                    # Optimize model with callback if applicable
                    opt_model.model.update()
                    opt_model.optimization()
                    # Generate model performance metrics and save to .csv file in .../results_files/
                    UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                        rand_state=i, results_file=out_file, data_name=file)
                    # Write LP file
                    if log_files:
                        opt_model.model.write(log + '.lp')

        # Generate .png images on metrics of bi-objective average results (saved to .../figures/ folder)
        biobj_data = pd.read_csv(os.getcwd() + '/results_files/' + output_name, na_values='?')
        for priority in priorities:
            sub_data = biobj_data[(biobj_data['Data'] == file) & (biobj_data['Priority'] == priority)]
            sub_data['Model'] = sub_data['Model'].str.replace(r'-biobj', '', regex=True)
            sub_data['Model'] = sub_data['Model'].str.replace(r'-ALL', '', regex=True)
            biobj_avg = pd.DataFrame(columns=summary_columns)
            for model in modeltypes:
                model_data = sub_data.loc[sub_data['Model'] == model]
                for sol_num in range(1, model_data['Sol_Num'].max()+1):
                    model_subdata = model_data.loc[model_data['Sol_Num'] == sol_num]
                    biobj_avg = biobj_avg.append({
                        'Data': file.replace('.csv', ''), 'H': int(model_subdata['H'].mean()),
                        '|I|': int(model_subdata['|I|'].mean()), 'Out_Acc': 100 * model_subdata['Out_Acc'].mean(),
                        'In_Acc': 100 * model_subdata['In_Acc'].mean(), 'Sol_Time': model_subdata['Sol_Time'].mean(),
                        'Model': model, 'Obj_Val': model_subdata['Obj_Val'].mean(), 'Sum_S': model_subdata['Sum_S'].mean(),
                        'Num_Branch_Nodes': model_subdata['Num_Branch_Nodes'].mean(), 'Priority': priority, 'Sol_Num': sol_num,
                        'Num_CB': model_subdata['Num_CB'].mean(), 'User_Cuts': model_subdata['User_Cuts'].mean(),
                        'Cuts_per_CB': model_subdata['Cuts_per_CB'].mean(), 'Total_CB_Time': model_subdata['Total_CB_Time'].mean(),
                        'INT_CB_Time': model_subdata['INT_CB_Time'].mean(), 'FRAC_CB_Time': model_subdata['FRAC_CB_Time'].mean(),
                        'CB_Eps': model_subdata['CB_Eps'].mean()
                    }, ignore_index=True)
            for plot_type in ['out_acc','in_acc','tree_size','overfit','objval','in_v_branching','out_v_branching']:
                UTILS.biobj_plot(biobj_avg, file, height=height, type=plot_type, priority=priority)


def pareto(argv):
    print(argv)
    data_files = None
    height = 0
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
                       'Time_Limit', 'Rand_State', 'Warm_Start_Use', 'Warm_Start_Time',
                       'CB_Eps', 'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'FRAC_CB_Time', 'INT_CB_Time',
                       'Max_Features', 'Single_Features', 'Regularization']
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(height) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_pareto.csv'
    else:
        output_name = str(file_out)
    out_file = os.getcwd() + '/results_files/' + output_name

    if file_out is None:
        with open(out_file, mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"')
            results_writer.writerow(summary_columns)
            results.close()

    target = 'target'

    for file in data_files:
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''))
        for i in rand_states:
            train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
            for modeltype in modeltypes:
                WSV = None
                # generate pareto frontier by fixing number of branching vertices in tree
                for num_features in range(1, 2 ** height):
                    print(f'\nDataset: {file}, Pareto tree size: {num_features}, Seed: {i}, Run Start: {time.strftime("%I:%M:%S %p", time.localtime())}')
                    extras = [f'num_features-{num_features}']
                    tree = TREE(h=height)
                    opt_model = OBCT(data=train_set, tree=tree, target=target, model=modeltype,
                                     time_limit=time_limit, model_extras=extras, warm_start=WSV)
                    opt_model.formulation()
                    opt_model.extras()
                    opt_model.model.update()
                    opt_model.optimization()
                    UTILS.node_assign(opt_model, tree)
                    UTILS.tree_check(tree)
                    UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                        rand_state=i, results_file=out_file, data_name=file)
                    model_wsm_acc, model_wsm_assgn = UTILS.model_acc(tree=tree, target=target, data=train_set)
                    if 'FlowOCT' == modeltype: WSV = {'tree': tree, 'data': False}
                    else: WSV = {'tree': tree, 'data': model_wsm_assgn}

        # Generate pareto plot of models using run averages of pareto.csv file
        pareto_data = pd.read_csv(os.getcwd()+'/results_files/'+output_name, na_values='?')
        file_data = pareto_data[pareto_data['Data'] == file.replace('.csv', '')]
        frontier_avg = pd.DataFrame(columns=summary_columns)
        for model in ['FlowOCT', 'MCF1', 'MCF2', 'CUT1', 'CUT2']:
            sub_data = file_data.loc[file_data['Model'] == model]
            for feature in sub_data['Max_Features'].unique():
                subsub_data = sub_data.loc[sub_data['Max_Features'] == feature]
                frontier_avg = frontier_avg.append({
                    'Data': file.replace('.csv', ''), 'H': int(subsub_data['H'].mean()),
                    '|I|': int(subsub_data['|I|'].mean()), 'Out_Acc': 100 * subsub_data['Out_Acc'].mean(),
                    'In_Acc': 100 * subsub_data['In_Acc'].mean(), 'Sol_Time': subsub_data['Sol_Time'].mean(),
                    'MIP_Gap': 100 * subsub_data['MIP_Gap'].mean(), 'Model': model,
                    'Obj_Val': subsub_data['Obj_Val'].mean(), 'Obj_Bound': subsub_data['Obj_Bound'].mean(),
                    'Num_CB': subsub_data['Num_CB'].mean(), 'User_Cuts': subsub_data['User_Cuts'].mean(),
                    'Cuts_per_CB': subsub_data['Cuts_per_CB'].mean(), 'Total_CB_Time': subsub_data['Total_CB_Time'].mean(),
                    'INT_CB_Time': subsub_data['INT_CB_Time'].mean(), 'FRAC_CB_Time': subsub_data['FRAC_CB_Time'].mean(),
                    'CB_Eps': subsub_data['CB_Eps'].mean(), 'Max_Features': feature
                }, ignore_index=True)
        for plot_type in ['out_acc', 'in_acc', 'time']:
            UTILS.pareto_plot(frontier_avg, type=plot_type)


if __name__ == "__main__":
    main(sys.argv[1:])

if __name__ == "__multiobj__":
    main(sys.argv[1:])

if __name__ == "__pareto__":
    main(sys.argv[1:])