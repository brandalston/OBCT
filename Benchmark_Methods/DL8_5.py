'''
This file is the implementation of the DL8.5 model found in the paper ''[Learning optimal decision trees using caching branch-and-bound search](https://ojs.aaai.org/index.php/AAAI/article/view/5711)''.
and publicly available on https://dl85.readthedocs.io/en/latest/
Code is taken directly from https://dl85.readthedocs.io/en/latest/
All rights and ownership are to the original owners.
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time, csv, getopt, sys, os
from dl85 import DL85Classifier
import UTILS as OU


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    rand_states = None
    file_out = None
    try:
        opts, args = getopt.getopt(argv, "d:h:t:r:f:",
                                   ["data_files=", "heights=", "timelimit=", "rand_states=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB',
                       'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', 'Calibration', 'Single_Feature_Use', 'Max_Features']
    output_path = 'results_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_DL8.5' + \
                      '_T:' + str(time_limit)+'.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()
    for file in data_files:
        data = OU.get_data(file.replace('.csv', ''), binarization='all-candidates')
        for h in heights:
            for i in rand_states:
                print('\nDataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: ' + str(i) + '. Run Start: ' +
                      str(time.strftime("%I:%M %p", time.localtime())))
                # data split
                train_set, test_set = train_test_split(data, train_size=0.75, random_state=i)
                X_train, Y_train = train_set.drop('target', axis=1), train_set['target']
                X_test, Y_test = test_set.drop('target', axis=1), test_set['target']
                # initialize the classifier , train and predict
                clf = DL85Classifier(max_depth=h, time_limit=time_limit)
                clf.fit(X_train, Y_train)
                if clf.runtime_ < time_limit:
                    print('Optimal solution found in '+str(round(clf.runtime_, 4))+'s')
                else:
                    print('Time limit reached.'+str(time.strftime("%I:%M %p", time.localtime())))
                y_pred = clf.predict(X_test)
                with open(out_file, mode='a') as results:
                    results_writer = csv.writer(results, delimiter=',', quotechar='"')
                    results_writer.writerow(
                        [file.replace('.csv', ''), h, len(train_set),
                         accuracy_score(Y_test, y_pred), clf.accuracy_, clf.runtime_,
                         'N/A', 'N/A', 'N/A', 'DL8.5', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', time_limit, i,
                         'N/A', 'N/A', 'N/A'])
                    results.close()
