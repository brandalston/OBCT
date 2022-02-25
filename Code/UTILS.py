import numpy as np
import pandas as pd
import sys
import time
import random
import RESULTS as OR


def get_data(name, target):
    # Return dataset from 'name' in Pandas dataframe
    # dataset located in workspace folder named 'Datasets'
    # Remove any non-numerical features from dataset
    global data, encoding_map
    try:
        if '_enc' not in name:
            data = pd.read_csv('Datasets/'+name+'.csv', na_values='?')
            data, encoding_map = encode(data, target)
            return data, encoding_map
        else:
            data = pd.read_csv('Datasets/'+name+'.csv', na_values='?')
            return data, {}
    except:
        print("Dataset Not Found or Error in Encoding Process!")
    return


def encode(data, target, cont_bin_num=2):
    columns, data_types = data.columns[data.columns != target], data.dtypes
    new_data_columns, encoding_map = [], {column: [None, [], []] for column in columns}
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)
    # data_name = name.replace('_enc', '')
    sub_iter = 0
    for column in columns:
        if data_types[column] == int:
            encoding_map[column][0] = data_types[column]
            if len(data[column].unique()) == 2:
                encoding_map[column][1] = list(data[column].unique())
                encoding_map[column].append(sub_iter)
                encoding_map[column][2].append(f'{column}')
                new_data_columns.append(f'{column}')
                sub_iter += 1
            else:
                for item in data[column].unique():
                    encoding_map[column][1].append(item)
                    encoding_map[column][2].append(f'{column}.{item}')
                    encoding_map[column].append(sub_iter)
                    new_data_columns.append(f'{column}.{item}')
                    sub_iter += 1
        elif data_types[column] == float:
            encoding_map[column][0] = data_types[column]
            test_values = np.linspace(min(data[column]), max(data[column]), cont_bin_num + 1).tolist()
            encoding_map[column][1] = list(zip(test_values, test_values[1:]))
            for item in range(len(encoding_map[column][1])):
                encoding_map[column][2].append(f'{column}.{item}')
                encoding_map[column].append(sub_iter)
                new_data_columns.append(f'{column}.{item}')
                sub_iter += 1
        elif data_types[column] == str or object:
            encoding_map[column][0] = data_types[column]
            if len(data[column].unique()) == 2:
                encoding_map[column][1] = list(data[column].unique())
                encoding_map[column][2].append(f'{column}')
                encoding_map[column].append(sub_iter)
                new_data_columns.append(f'{column}')
                sub_iter += 1
            else:
                for item in data[column].unique():
                    encoding_map[column][1].append(item)
                    index = list(data[column].unique()).index(item)
                    encoding_map[column][2].append(f'{column}.{index}')
                    encoding_map[column].append(sub_iter)
                    new_data_columns.append(f'{column}.{index}')
                    sub_iter += 1
    else:
        new_data = pd.DataFrame(0, index=data.index, columns=new_data_columns)
        for i in data.index:
            for col in columns:
                if encoding_map[col][0] == int:
                    if len(data[col].unique()) == 2:
                        if data.at[i, col] == data[col].unique()[0]: new_data.at[i, col] = 1
                    else:
                        for sub_value in encoding_map[col][1]:
                            if data.at[i, col] == sub_value: new_data.at[i, f'{col}.{sub_value}'] = 1
                elif encoding_map[col][0] == float:
                    for pair in encoding_map[col][1]:
                        if pair[0] <= data.at[i, col] <= pair[1]: new_data.at[
                            i, f'{col}.{encoding_map[col][1].index(pair)}'] = 1
                elif encoding_map[col][0] == str or object:
                    if len(data[col].unique()) == 2:
                        if data.at[i, col] == data[col].unique()[0]: new_data.at[i, col] = 1
                    else:
                        for item in encoding_map[col][1]:
                            sub_col = list(data[col].unique()).index(item)
                            if data.at[i, col] == item: new_data.at[i, f'{col}.{sub_col}'] = 1
        encoding_map = {list(encoding_map.keys()).index(col): value for col, value in encoding_map.items()}
        new_data['target'] = data.target
    return new_data, encoding_map


def random_tree(target, data, tree, repeats=100, threshold=0):
    # Generate best randomly assigned height 'h' tree for dataset 'data'
    # Threshold is (0,1) float for 'best features' to use in random trees
    # generate 'repeats' # of trees
    # return best acc tree and data assignments for warm start

    # select 'best features' and unique classes
    features = data.columns[data.columns != target]
    classes = data[target].unique()
    selected_features = [feature for feature in features
                         if data.loc[:, feature].sum() / len(data) >= threshold]

    # generate random trees and store best accuracy tree
    level_start = time.perf_counter()
    level_acc, best_l_tree, level_data = -1, {}, {}
    for i in range(repeats):
        # randomly assign features to base tree
        l_tree = level_tree(tree, selected_features, classes)
        # generate acc results
        acc, data_assignments = OR.model_acc(tree=l_tree, target=target, data=data)
        if acc > level_acc:
            level_acc = acc
            best_l_tree = l_tree
            level_data = data_assignments
    level_time = time.perf_counter()-level_start
    # print(f'Level Time: {round(level_time, 4)}s. Acc: {level_acc}')

    path_start = time.perf_counter()
    path_acc, best_p_tree, path_data = -1, {}, {}
    for i in range(repeats):
        # randomly assign features to base tree
        p_tree = path_tree(tree, selected_features, classes, tree.path, tree.child)
        # generate acc results
        acc, data_assignments = OR.model_acc(tree=p_tree, target=target, data=data)
        if acc > path_acc:
            path_acc = acc
            best_p_tree = p_tree
            path_data = data_assignments
    path_time = time.perf_counter()-path_start
    # print(f'Path Time: {round(path_time, 4)}s. Acc: {path_acc}')

    # store and return best tree assignment and data results in dictionary for warm start
    if level_acc > path_acc:
        WSV = {'tree': best_l_tree.DG_prime.nodes(data=True), 'data': level_data, 'acc': level_acc, 'time': level_time}
    else:
        WSV = {'tree': best_p_tree.DG_prime.nodes(data=True), 'data': path_data, 'acc': path_acc, 'time': path_time}
    return WSV


def level_tree(base_tree, feature_list, classes):
    # clear any existing node assignments
    for n in base_tree.DG_prime.nodes():
        if 'class' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['class']
        if 'branch on feature' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['branch on feature']
        if 'pruned' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['pruned']
    node_assignments = {n: None for n in base_tree.DG_prime.nodes}

    max_level = max(base_tree.node_level.keys())
    # branch on root node
    base_tree.DG_prime.nodes[0]['branch on feature'] = random.choice(feature_list)
    node_assignments[0] = 'branch on feature'

    # For each level in tree
    #   For each node in the level
    #       If ancestor is branching feature randomly assign random class or feature
    #       If node is leaf node and ancestor is branching feature assign random class
    #       If ancestor is class or pruned assign pruned
    for level in base_tree.node_level:
        if level == 0: continue
        for node in base_tree.node_level[level]:
            if node_assignments[base_tree.direct_ancestor[node]] == 'branch on feature':
                if random.random() > .5 and level != max_level:
                    base_tree.DG_prime.nodes[node]['branch on feature'] = random.choice(feature_list)
                    node_assignments[node] = 'branch on feature'
                else:
                    base_tree.DG_prime.nodes[node]['class'] = random.choice(classes)
                    node_assignments[node] = 'class'
            else:
                base_tree.DG_prime.nodes[node]['pruned'] = 0
                node_assignments[node] = 'pruned'

    return base_tree


def path_tree(base_tree, feature_list, classes, path, child):
    # clear any existing node assignments
    for n in base_tree.DG_prime.nodes():
        if 'class' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['class']
        if 'branch on feature' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['branch on feature']
        if 'pruned' in base_tree.DG_prime.nodes[n]:
            del base_tree.DG_prime.nodes[n]['pruned']
    node_assignments = {n: None for n in base_tree.DG_prime.nodes}
    node_list = list(base_tree.DG_prime.nodes())

    # branch on root node
    base_tree.DG_prime.nodes[0]['branch on feature'] = random.choice(feature_list)
    node_assignments[0] = 'branch on feature'
    node_list.remove(0)

    while len(node_list) > 0:
        selected = random.choice(node_list)
        base_tree.DG_prime.nodes[selected]['class'] = random.choice(classes)
        node_assignments[selected] = 'class'
        for v in reversed(base_tree.path[selected][1:-1]):
            if node_assignments[v] is None:
                base_tree.DG_prime.nodes[v]['branch on feature'] = random.choice(feature_list)
                node_assignments[v] = 'branch on feature'
                node_list.remove(v)
            else: break
        for c in base_tree.child[selected]:
            if node_assignments[c] is None:
                base_tree.DG_prime.nodes[c]['pruned'] = 0
                node_assignments[c] = 'pruned'
                node_list.remove(c)
            else: break
        node_list.remove(selected)
    return base_tree


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