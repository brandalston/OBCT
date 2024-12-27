import numpy as np
import networkx as nx
import sys, csv, os, math, warnings
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from data_load import *
import TREE
warnings.filterwarnings("ignore")


def get_data(dataset, binarization=None):
    dataset2loadfcn = {
        'balance_scale': load_balance_scale,
        'banknote': load_banknote_authentication,
        'blood': load_blood_transfusion,
        'breast_cancer': load_breast_cancer,
        'car': load_car_evaluation,
        'kr_vs_kp': load_chess,
        'climate': load_climate_model_crashes,
        'house_votes_84': load_congressional_voting_records,
        'fico_binary': load_fico_binary,
        'glass': load_glass_identification,
        'hayes_roth': load_hayes_roth,
        'image': load_image_segmentation,
        'ionosphere': load_ionosphere,
        'iris': load_iris,
        'monk1': load_monk1,
        'monk2': load_monk2,
        'monk3': load_monk3,
        'parkinsons': load_parkinsons,
        'soybean_small': load_soybean_small,
        'spect': load_spect,
        'tic_tac_toe': load_tictactoe_endgame,
        'wine_red': load_wine_red,
        'wine_white': load_wine_white
    }

    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                                                                                'glass', 'image', 'ionosphere',
                          'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    already_processed = ['fico_binary']

    load_function = dataset2loadfcn[dataset]
    X, y = load_function()
    codes, uniques = pd.factorize(y)
    y = pd.Series(codes, name='target')

    if dataset in already_processed:
        X_new = X
    else:
        if dataset in numerical_datasets:
            if binarization is None:
                X_new, ct = preprocess(X, numerical_features=X.columns)
                X_new = pd.DataFrame(X_new, columns=X.columns)
            else:
                X_new, ct = preprocess(X, y=y, binarization=binarization, numerical_features=X.columns)
                cols = []
                for key in ct.transformers_[0][1].candidate_thresholds_:
                    for item in ct.transformers_[0][1].candidate_thresholds_[key]:
                        cols.append(f"{key}<={item}")
                X_new = pd.DataFrame(X_new, columns=cols)
        else:
            X_new, ct = preprocess(X, categorical_features=X.columns)
            X_new = pd.DataFrame(X_new, columns=ct.get_feature_names_out(X.columns))
            X_new.columns = X_new.columns.str.replace('cat__', '')
    X_new = X_new.astype(int)
    data_new = pd.concat([X_new, y], axis=1)
    return data_new


def preprocess(X, y=None, numerical_features=None, categorical_features=None, binarization=None):
    """ Preprocess a dataset.

    Numerical features are scaled to the [0,1] interval by default, but can also
    be binarized, either by considering all candidate thresholds for a
    univariate split, or by binning. Categorical features are one-hot encoded.

    Parameters
    ----------
    X
    X_test
    y_train : pandas Series of training labels, only needed for binarization
        with candidate thresholds
    numerical_features : list of numerical features
    categorical_features : list of categorical features
    binarization : {'all-candidates', 'binning'}, default=None
        Binarization technique for numerical features.
        all-candidates
            Use all candidate thresholds.
        binning
            Perform binning using scikit-learn's KBinsDiscretizer.
        None
            No binarization is performed, features scaled to the [0,1] interval.

    Returns
    -------
    X_train_new : pandas DataFrame that is the result of binarizing X
    """

    if numerical_features is None:
        numerical_features = []
    if categorical_features is None:
        categorical_features = []

    numerical_transformer = MinMaxScaler()
    if binarization == 'all-candidates':
        numerical_transformer = CandidateThresholdBinarizer()
    elif binarization == 'binning':
        numerical_transformer = KBinsDiscretizer(encode='onehot-dense')
    # categorical_transformer = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore') # Should work in scikit-learn 1.0
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ct = ColumnTransformer([("num", numerical_transformer, numerical_features),
                            ("cat", categorical_transformer, categorical_features)])
    X_train_new = ct.fit_transform(X, y)

    return X_train_new, ct


class CandidateThresholdBinarizer(TransformerMixin, BaseEstimator):
    """ Binarize continuous data using candidate thresholds.

    For each feature, sort observations by values of that feature, then find
    pairs of consecutive observations that have different class labels and
    different feature values, and define a candidate threshold as the average of
    these two observations’ feature values.

    Attributes
    ----------
    candidate_thresholds_ : dict mapping features to list of thresholds
    """

    def fit(self, X, y):
        """ Finds all candidate split thresholds for each feature.

        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names
        y : pandas Series with labels

        Returns
        -------
        self
        """
        X_y = X.join(y)
        self.candidate_thresholds_ = {}
        for j in X.columns:
            thresholds = []
            sorted_X_y = X_y.sort_values([j, y.name])  # Sort by feature value, then by label
            prev_feature_val, prev_label = sorted_X_y.iloc[0][j], sorted_X_y.iloc[0][y.name]
            for idx, row in sorted_X_y.iterrows():
                curr_feature_val, curr_label = row[j], row[y.name]
                if (curr_label != prev_label and
                        not math.isclose(curr_feature_val, prev_feature_val)):
                    thresh = (prev_feature_val + curr_feature_val) / 2
                    thresholds.append(thresh)
                prev_feature_val, prev_label = curr_feature_val, curr_label
            self.candidate_thresholds_[j] = thresholds
        return self

    def transform(self, X):
        """ Binarize numerical features using candidate thresholds.

        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names

        Returns
        -------
        Xb : pandas DataFrame that is the result of binarizing X
        """
        check_is_fitted(self)
        Xb = pd.DataFrame()
        for j in X.columns:
            for threshold in self.candidate_thresholds_[j]:
                binary_test_name = "{}<={}".format(j, threshold)
                Xb[binary_test_name] = (X[j] <= threshold)
        Xb.replace({"False": 0, "True": 1}, inplace=True)
        return Xb


def dv_results(model, tree, features, classes, datapoints):
    # Print assigned features, classes, and pruned nodes of tree
    for v in tree.DG_prime.nodes:
        for f in features:
            if model._B[v, f].x > .5:
                print('vertex ' + str(v) + ' assigned feature ' + str(f))
        for k in classes:
            if model._W[v, k].x > 0.5:
                print('vertex ' + str(v) + ' assigned class ' + str(k))
        if model._P[v].x < .5 and all(elem < .5 for elem in [model._B[v, f].x for f in features]):
            print('vertex ' + str(v) + ' pruned')

    # Print datapoint paths through tree
    for i in datapoints:
        for v in tree.DG_prime.nodes:
            if model._Q[i, v].x > 0.5:
                print('datapoint ' + str(i) + ' use vertex ' + str(v) + ' in source-terminal path')
            if model._S[i, v].x > 0.5:
                print('datapoint ' + str(i) + ' terminal vertex ' + str(v))


def node_assign(model, tree):
    # assign features, classes, and pruned nodes determined by model to tree
    for i in model.B.keys():
        if model.B[i].x > 0.5:
            # print(i[0], 'Num-Tree-size on', i[1])
            tree.DG_prime.nodes[i[0]]['branch on feature'] = i[1]
            tree.DG_prime.nodes[i[0]]['color'] = 'green'
    for i in model.W.keys():
        if model.W[i].x > 0.5:
            # print(i[0], 'class', i[1])
            tree.DG_prime.nodes[i[0]]['class'] = i[1]
            tree.DG_prime.nodes[i[0]]['color'] = 'yellow'
    for i in model.P.keys():
        if model.P[i].x < .5 and all(x < .5 for x in [model.B[i, f].x for f in model.features]):
            # print(i, 'pruned')
            tree.DG_prime.nodes[i]['pruned'] = 0
            tree.DG_prime.nodes[i]['color'] = 'red'


def tree_check(tree):
    # check for each class node v
    # all nodes n in ancestors of v are Num-Tree-size nodes
    # all children c of v are pruned
    class_nodes = {v: tree.DG_prime.nodes[v]['class']
                   for v in tree.DG_prime.nodes if 'class' in tree.DG_prime.nodes[v]}
    branch_nodes = {v: tree.DG_prime.nodes[v]['branch on feature']
                    for v in tree.DG_prime.nodes if 'branch on feature' in tree.DG_prime.nodes[v]}
    pruned_nodes = {v: tree.DG_prime.nodes[v]['pruned']
                    for v in tree.DG_prime.nodes if 'pruned' in tree.DG_prime.nodes[v]}
    for v in class_nodes.keys():
        if not (all(n in branch_nodes.keys() for n in tree.path[v][:-1])):
            return False
        if not (all(c in pruned_nodes.keys() for c in tree.child[v])):
            return False


def model_acc(tree, target, data):
    # get Num-Tree-size and class node and direct children of each node
    branching_nodes = nx.get_node_attributes(tree.DG_prime, 'branch on feature')
    class_nodes = nx.get_node_attributes(tree.DG_prime, 'class')
    acc = 0
    results = {i: [None, data.at[i, target], []] for i in data.index}
    # for each datapoint
    #   start at root
    #   while unassigned to class
    #   if current node is Num-Tree-size
    #      branch left or right
    #      new current node = according child
    #   elif current node is class
    #      assign datapoint to class
    #      check if correctly assigned
    for i in data.index:
        current_node = 0
        while results[i][0] is None:
            results[i][2].append(current_node)
            if current_node in branching_nodes and data.at[i, branching_nodes[current_node]] == 0:
                current_node = tree.successor[current_node][0]
            elif current_node in branching_nodes and data.at[i, branching_nodes[current_node]] == 1:
                current_node = tree.successor[current_node][1]
            elif current_node in class_nodes:
                results[i][0] = class_nodes[current_node]
                if class_nodes[current_node] == results[i][1]:
                    acc += 1
                    results[i].append('correct')
            else:
                results[i][0] = 'ERROR'
    return acc, results


def model_summary(opt_model, tree, test_set, rand_state, results_file, data_name):
    if 'biobj' not in opt_model.modeltype:
        # Log model results to .csv file and save .png if applicable
        node_assign(opt_model, tree)
        if tree_check(tree):
            print('Invalid Tree!!')
        test_acc, test_assignments = model_acc(tree=tree, target=opt_model.target, data=test_set)
        train_acc, train_assignments = model_acc(tree=tree, target=opt_model.target, data=opt_model.data)
        with open(results_file, mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"')
            results_writer.writerow(
                [data_name, tree.height, len(opt_model.datapoints),
                 test_acc / len(test_set), train_acc / len(opt_model.datapoints), opt_model.model.Runtime,
                 opt_model.model.MIPGap, opt_model.model.ObjVal, opt_model.model.ObjBound, opt_model.modeltype,
                 opt_model.warmstart['use'], opt_model.branch_weight, opt_model.warmstart['time'], rand_state, opt_model.time_limit,
                 opt_model.model._numcb, opt_model.model._numcuts, opt_model.model._cbtime,
                 opt_model.model._mipnodetime, opt_model.model._mipsoltime,
                 opt_model.max_features, opt_model.repeat_use, opt_model.regularization])
            results.close()
    else:
        # print(f'Problem has {opt_model.model.NumObj} objectives\nGurobi found {opt_model.model.SolCount} solutions')
        for s in range(opt_model.model.SolCount):
            # Set which solution we will query from now on
            opt_model.model.params.SolutionNumber = s
            branch_nodes = {k[0]: k[1] for (k, v) in opt_model.model.getAttr('Xn', opt_model.B).items() if v > 0.5}
            class_nodes = {k[0]: k[1] for (k, v) in opt_model.model.getAttr('Xn', opt_model.W).items() if v > 0.5}
            sum_s = len({k[0]: k[1] for (k, v) in opt_model.model.getAttr('Xn', opt_model.S).items() if v > 0.5})
            dummy_tree = TREE.TREE(h=opt_model.tree.height)
            for v in branch_nodes: dummy_tree.DG_prime.nodes[v]['branch on feature'] = branch_nodes[v]
            for v in class_nodes: dummy_tree.DG_prime.nodes[v]['class'] = class_nodes[v]
            test_acc, test_assignments = model_acc(tree=dummy_tree, target=opt_model.target, data=test_set)
            train_acc, train_assignments = model_acc(tree=dummy_tree, target=opt_model.target, data=opt_model.data)
            with open(results_file, mode='a') as results:
                results_writer = csv.writer(results, delimiter=',', quotechar='"')
                results_writer.writerow(
                    [data_name, tree.height, len(opt_model.datapoints), test_acc / len(test_set), train_acc / len(opt_model.datapoints),
                     sum_s, opt_model.model.ObjVal, len(branch_nodes), s + 1, opt_model.priority, opt_model.model.Runtime,
                     opt_model.modeltype, opt_model.time_limit, rand_state, 'N/A', 'N/A',
                     opt_model.eps, opt_model.model._numcb, opt_model.model._numcuts, opt_model.model._avgcuts,
                     opt_model.model._cbtime, opt_model.model._mipnodetime, opt_model.model._mipsoltime])
                results.close()


def pareto_plot(data, type='out_acc'):
    # Generate pareto frontier .png file
    models = data['Model'].unique()
    name = data['Data'].unique()[0]
    height = max(data['H'].unique())
    dom_points = []
    for model in models:
        sub_data = data.loc[data['Model'] == model]
        best_acc, max_features = -1, 0
        for i in sub_data.index:
            if type == 'out_acc':
                if (sub_data.at[i, 'Out_Acc']) > best_acc and (sub_data.at[i, 'Max_Features'] > max_features):
                    dom_points.append(i)
                    best_acc, max_features = sub_data.at[i, 'Out_Acc'], sub_data.at[i, 'Max_Features']
            elif type == 'in_acc':
                if (sub_data.at[i, 'In_Acc']) > best_acc and (sub_data.at[i, 'Max_Features'] > max_features):
                    dom_points.append(i)
                    best_acc, max_features = sub_data.at[i, 'In_Acc'], sub_data.at[i, 'Max_Features']
    domed_pts = list(set(data.index).difference(set(dom_points)))
    dominating_points = data.iloc[dom_points, :]
    if domed_pts: dominated_points = data.iloc[domed_pts, :]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax2.set_xticks([3.5, 7.5, 15.5, 31.5])
    ax2.set_xticklabels([f'h={h}' for h in [2, 3, 4, 5]])
    ax1.set_xlabel('Num. Branching Vertices')
    ax1.xaxis.set_ticks(np.arange(1, max(data['Max_Features'].unique()) + 1, 5))
    markers = {'MCF1': 'X', 'MCF2': 'p', 'CUT1': 's', 'CUT2': 'P', 'FlowOCT': '*'}
    colors = {'MCF1': 'blue', 'MCF2': 'orange', 'CUT1': 'green', 'CUT2': 'red', 'FlowOCT': 'k'}

    for model in models:
        for h in [2, 3, 4, 5]:
            plt.axvline(x=2 ** h - .5, color='k', linewidth=1)
        if type == 'time':
            ax1.scatter(dominating_points.loc[data['Model'] == model]['Max_Features'],
                        dominating_points.loc[data['Model'] == model]['Sol_Time'],
                        marker=markers[model], color=colors[model], label=model)
            if domed_pts: ax1.scatter(dominated_points.loc[data['Model'] == model]['Max_Features'],
                                      dominated_points.loc[data['Model'] == model]['Sol_Time'],
                                      marker=markers[model], color=colors[model])
        elif type == 'out_acc':
            ax1.scatter(dominating_points.loc[data['Model'] == model]['Max_Features'],
                        dominating_points.loc[data['Model'] == model]['Out_Acc'],
                        marker=markers[model], color=colors[model], label=model)
            if domed_pts: ax1.scatter(dominated_points.loc[data['Model'] == model]['Max_Features'],
                                      dominated_points.loc[data['Model'] == model]['Out_Acc'],
                                      marker=markers[model], color=colors[model], alpha=0.2)

        elif type == 'in_acc':
            ax1.scatter(dominating_points.loc[data['Model'] == model]['Max_Features'],
                        dominating_points.loc[data['Model'] == model]['In_Acc'],
                        marker=markers[model], color=colors[model], label=model)
            if domed_pts: ax1.scatter(dominated_points.loc[data['Model'] == model]['Max_Features'],
                                      dominated_points.loc[data['Model'] == model]['In_Acc'],
                                      marker=markers[model], color=colors[model], alpha=0.05)
        ax1.legend()
        if type == 'out_acc': ax1.set_ylabel('Out-of-Sample Acc. (%)')
        elif type == 'in_acc': ax1.set_ylabel('In-Sample Acc. (%)')
        elif type == 'time': ax1.set_ylabel('Solution Time (s)')

        name = name.replace('_enc', '')
        if type == 'out_acc': ax1.set_title(f'{str(name)} Out-of-Sample Acc. Pareto Frontier')
        elif type == 'in_acc': ax1.set_title(f'{str(name)} In-Sample Acc. Pareto Frontier')
        elif type == 'time': ax1.set_title(f'{str(name)} Solution Time Distribution')

    plt.savefig(os.getcwd()+f'/figures/pareto/{name}_h_{height}_{type}_pareto.png', dpi=300)
    plt.close()


def biobj_plot(raw_data, name, height, type=None, priority='data'):
    data = pd.DataFrame(columns=raw_data.columns)
    for model in raw_data['Model'].unique():
        model_data = raw_data[raw_data['Model'] == model]
        if type in ['in_acc', 'out_acc', 'objval']:
            model_data['Sol_Num'] = model_data['Sol_Num'].values[::-1]
        data = pd.concat([data, model_data])
    data['Overfit'] = data['Out_Acc']-data['In_Acc']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    markers = {'MCF1': 'X', 'MCF2': 'p', 'CUT1': 's', 'CUT2': 'P'}
    colors = {'MCF1': 'blue', 'MCF2': 'orange', 'CUT1': 'green', 'CUT2': 'red'}

    models = data['Model'].unique()
    for model in models:
        sub_data = data.loc[data['Model'] == model]
        if type == 'in_acc':
            ax1.scatter(sub_data['Sol_Num'],sub_data['In_Acc'],marker=markers[model],color=colors[model],label=model)
        elif type == 'out_acc':
            ax1.scatter(sub_data['Sol_Num'],sub_data['Out_Acc'],marker=markers[model],color=colors[model],label=model)
        elif type == 'tree_size':
            ax1.scatter(sub_data['Sol_Num'], sub_data['Num_Branch_Nodes'], marker=markers[model],
                        color=colors[model], label=model)
        elif type == 'overfit':
            ax1.scatter(sub_data['Sol_Num'], sub_data['Overfit'], marker=markers[model],
                        color=colors[model], label=model)
        elif type == 'in_v_branching':
            ax1.scatter(sub_data['Num_Branch_Nodes'], sub_data['In_Acc'], marker=markers[model],
                        color=colors[model], label=model)
        elif type == 'out_v_branching':
            ax1.scatter(sub_data['Num_Branch_Nodes'], sub_data['Out_Acc'], marker=markers[model],
                        color=colors[model], label=model)
        elif type == 'objval':
            ax1.scatter(sub_data['Sol_Num'], sub_data['Sum_S'], marker=markers[model], color=colors[model], label=model)
    ax1.legend()

    if type == 'in_acc':
        ax1.set_ylabel('In-Sample Acc. (%)')
        ax1.set_xlabel('Nth Solution Found in 1hr')
        ax1.set_title(f'{str(name)} Bi-objective Results In-Sample Acc.')
    elif type == 'out_acc':
        ax1.set_ylabel('Out-of-Sample Acc. (%)')
        ax1.set_xlabel('Nth Solution Found in 1hr')
        ax1.set_title(f'{str(name)} Bi-objective Results Out-of-Sample Acc.')
    elif type == 'tree_size':
        ax1.set_ylabel('Num. Branching Vertices')
        ax1.set_xlabel('Nth Solution Found in 1hr')
        ax1.set_title(f'{str(name)} Bi-objective Results Tree Size')
    elif type == 'overfit':
        ax1.set_xlabel('Nth Solution Found in 1hr')
        ax1.set_ylabel('Overfit %-age')
        ax1.set_title(f'{str(name)} Bi-objective Results Overfitting')
    elif type == 'in_v_branching':
        ax1.set_ylabel('In-Sample Acc. (%)')
        ax1.set_xlabel('Num. Branching Vertices')
        ax1.set_title(f'{str(name)} Bi-objective Results In-Acc v Tree Size')
    elif type == 'out_v_branching':
        ax1.set_ylabel('In-Sample Acc. (%)')
        ax1.set_xlabel('Num. Branching Vertices')
        ax1.set_title(f'{str(name)} Bi-objective Results Out-Acc v Tree Size')
    elif type == 'objval':
        ax1.set_ylabel('Instances Solved')
        ax1.set_xlabel('Nth Solution Found in 1hr')
        ax1.set_title(f'{str(name)} Bi-objective Results Obj. Value (1a)')

    plt.savefig(os.getcwd()+f'/figures/biobj_{priority}/{type}/{name}_h_{int(height)}_biobj_{priority}_{type}.png',dpi=300)
    fig.tight_layout()


class consol_log:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
