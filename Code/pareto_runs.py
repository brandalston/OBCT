#!/usr/bin/python
import os
import time
import pandas as pd
import getopt
import sys
import csv
from sklearn.model_selection import train_test_split
from OBCT import OBCT
from TREE import TREE
import UTILS as OU


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
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', 'Calibration', 'Single_Feature_Use', 'Max_Features']
    output_path = os.getcwd() + '/Code/results_files/'
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
        data = OU.get_data(file.replace('.csv', ''))
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
                                     time_limit=time_limit, model_extras=extras, warm_start=WSV, name=file)
                    opt_model.formulation()
                    opt_model.extras()
                    opt_model.model.update()
                    opt_model.optimization()
                    OU.node_assign(opt_model, tree)
                    OU.tree_check(tree)
                    OU.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                     rand_state=rand_states[i], results_file=out_file)
                    model_wsm_acc, model_wsm_assgn = OU.model_acc(tree=tree, target=target,
                                                                  data=train_set)
                    WSV = {'tree': tree, 'data': model_wsm_assgn}
                    if 'AGHA' == modeltype: WSV = {'tree': tree, 'data': False}

        # Generate pareto plot of models using run averages of pareto.csv file
        pareto_data = pd.pareto_data = pd.read_csv(os.getcwd()+'/Code/results_files/'+file_out, na_values='?')
        file_data = pareto_data[pareto_data['Data'] == file.replace('.csv', '')]
        frontier_avg = pd.DataFrame(columns=summary_columns)
        for model in ['AGHA','MCF1','MCF2','CUT1','CUT2','FlowOCT','BendersOCT']:
            sub_data = file_data.loc[file_data['Model'] == model]
            for feature in sub_data['Max_Features'].unique():
                subsub_data = sub_data.loc[sub_data['Max_Features'] == feature]
                frontier_avg = frontier_avg.append({
                    'Data': file.replace('.csv', ''), 'H': int(subsub_data['H'].mean()),
                    '|I|': int(subsub_data['|I|'].mean()),
                    'Out_Acc': 100 * subsub_data['Out_Acc'].mean(), 'In_Acc': 100 * subsub_data['In_Acc'].mean(),
                    'Sol_Time': subsub_data['Sol_Time'].mean(), 'MIP_Gap': 100 * subsub_data['MIP_Gap'].mean(),
                    'Obj_Bound': subsub_data['Obj_Bound'].mean(), 'Obj_Val': 100*subsub_data['Obj_Val'].mean(), 'Model': model,
                    'Num_CB': subsub_data['Num_CB'].mean(), 'User_Cuts': subsub_data['User_Cuts'].mean(),
                    'Cuts_per_CB': subsub_data['Cuts_per_CB'].mean(),
                    'Total_CB_Time': subsub_data['Total_CB_Time'].mean(), 'INT_CB_Time': subsub_data['INT_CB_Time'].mean(),
                    'FRAC_CB_Time': subsub_data['FRAC_CB_Time'].mean(), 'CB_Eps': subsub_data['CB_Eps'].mean(),
                    'Time_Limit': time_limit, 'Rand_State': 'None',
                    'Calibration': False, 'Single_Feature_Use': False, 'Max_Features': float(feature)
                }, ignore_index=True)
        OU.pareto_plot(frontier_avg)
