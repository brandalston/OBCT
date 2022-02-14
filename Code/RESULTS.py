import networkx as nx
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


def dv_results(model, tree, features, classes, datapoints):
    for v in tree.DG_prime.nodes:
        for f in features:
            if model._B[v, f].x > .5:
                print('vertex '+str(v)+' assigned feature '+str(f))
        for k in classes:
            if model._W[v, k].x > 0.5:
                print('vertex '+str(v)+' assigned class '+str(k))
        if model._P[v].x < .5 and 1.0 not in [model._B[v, f].x for f in features]:
            print('vertex '+str(v)+' pruned')
    # uncomment to print datapoint paths through tree
    '''
    for i in datapoints:
        for v in tree.DG_prime.nodes:
            if model._Q[i, v].x > 0.5:
                print('datapoint '+str(i)+' use vertex '+str(v)+' in source-terminal path')
            if model._S[i, v].x > 0.5:
                print('datapoint '+str(i)+' terminal vertex '+str(v))
    '''


def node_assign(model, tree):
    # assign features, classes, and pruned nodes determined by model to tree
    for i in model.B.keys():
        if model.B[i].x > 0.5:
            tree.DG_prime.nodes[i[0]]['branch on feature'] = i[1]
            tree.DG_prime.nodes[i[0]]['color'] = 'green'
            tree.labels[i[0]] = str(i[1])
            tree.color_map.append('green')
    for i in model.W.keys():
        if model.W[i].x > 0.5:
            tree.DG_prime.nodes[i[0]]['class'] = i[1]
            tree.DG_prime.nodes[i[0]]['color'] = 'yellow'
            tree.labels[i[0]] = str(i[1])
            tree.color_map.append('yellow')
    for i in model.P.keys():
        if model.P[i].x < .5 and all(x < .5 for x in [model.B[i, f].x for f in model.features]):
            tree.DG_prime.nodes[i]['pruned'] = 0
            tree.DG_prime.nodes[i]['color'] = 'red'
            tree.labels[i] = 'P'
            tree.color_map.append('red')


def tree_check(tree):
    # check for each class node v
    # all nodes n in ancestors of v are branching nodes
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
    # get branching and class node and direct children of each node
    branching_nodes = nx.get_node_attributes(tree.DG_prime, 'branch on feature')
    class_nodes = nx.get_node_attributes(tree.DG_prime, 'class')
    acc = 0
    results = {i: [None, data.at[i, target], []] for i in data.index}
    # for each datapoint
    #   start at root
    #   while unassigned to class
    #   if current node is branching
    #      yes: branch left or right
    #      new current node = child
    #   if current node is class
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


def model_summary(opt_model, tree, test_set, rand_state, results_file, fig_file):
    node_assign(opt_model, tree)
    if tree_check(tree=tree):
        print('Invalid Tree!!')
    if fig_file is not None:
        nx.draw(tree.DG_prime, pos=tree.pos, node_color=tree.color_map, labels=tree.labels, with_labels=True)
        plt.savefig(fig_file)
    test_acc, test_assignments = model_acc(tree=tree, target=opt_model.target, data=test_set)
    train_acc, train_assignments = model_acc(tree=tree, target=opt_model.target, data=opt_model.data)

    with open(results_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"')
        results_writer.writerow(
            [opt_model.dataname, tree.height, len(opt_model.datapoints),
             test_acc / len(test_set), train_acc / len(opt_model.datapoints), opt_model.model.Runtime,
             (opt_model.model.ObjBound - opt_model.model.ObjVal)/opt_model.model.ObjBound,
             opt_model.model.ObjBound, opt_model.model.ObjVal,
             opt_model.model._numcb, opt_model.model._numcuts, opt_model.model._avgcuts,
             opt_model.model._cbtime, opt_model.model._mipsoltime, opt_model.model._mipnodetime, opt_model.eps,
             opt_model.modeltype, opt_model.time_limit, rand_state,
             opt_model.fixed, opt_model.warmstart, opt_model.cc,
             opt_model.single_use, opt_model.level_tree, opt_model.max_features, opt_model.super_feature])
        results.close()


def pareto_frontier(data, models):
    dom_points = {}
    ticker = 0
    for model in models:
        sub_data = data.loc[data['Model'] == model]
        # print(f'\n\nPareto Frontier for model {model}')
        best_acc, max_features = -1, 0
        for i in sub_data.index:
            if (sub_data.at[i, 'Out-Acc']) > best_acc and (sub_data.at[i, 'Max Features'] > max_features):
                dom_points[ticker+i] = [sub_data.at[i, 'Max Features'], sub_data.at[i, 'Out-Acc'], model]
                # print(f'new dominating point: {sub_data.iloc[i, 4], sub_data.iloc[i, 2]}')
                best_acc, max_features = sub_data.at[i, 'Out-Acc'], sub_data.at[i, 'Max Features']
        ticker += 31

    dominating_points = pd.DataFrame.from_dict(dom_points, orient='index',
                                               columns=['Max Branching Nodes', 'Out-Acc', 'Model'])
    return dominating_points


def pareto_plot(data_name, results_file):
    for name in data_name:
        pareto_data = pd.read_excel(results_file, sheet_name=name)
        models = pareto_data['models'].unique()
        dominating_points = pareto_frontier(pareto_data, models)
        dominated_points = pareto_data.iloc[list(set(pareto_data.index).difference(set(dominating_points.index))), :]
        '''
        fig = plt.figure()
        axs = fig.add_subplot(111)
        ticker = 0
        colors = ['b', 'g', 'r', 'k', 'model']
        markers = ['s', 'p', 'P', '*', '^']
        for model in models:
            axs.scatter(dominating_points.loc[pareto_data['Model'] == model]['Max_Branches'],
                        dominating_points.loc[pareto_data['Model'] == model]['Acc'],
                        marker=markers[ticker], label=model, color=colors[ticker])

            z = np.polyfit(pareto_data.loc[pareto_data['Model'] == model]['Max_Branches'],
                           pareto_data.loc[pareto_data['Model'] == model]['Acc'], 5)
            p = np.poly1d(z)
            axs.plot(pareto_data.loc[pareto_data['Model'] == model]['Max_Branches'],
                     p(pareto_data.loc[pareto_data['Model'] == model]['Max_Branches']),
                     color=colors[ticker], alpha=0.5)
            axs.scatter(dominated_points.loc[dominated_points['Model'] == model]['Max_Branches'],
                        dominated_points.loc[dominated_points['Model'] == model]['Acc'],
                        marker=markers[ticker], color=colors[ticker], alpha=0.05)
            ticker += 1
        plt.legend(loc='lower right')
        plt.ylabel('Acc (%)')
        plt.xlabel('Max Branches in Height 5 Tree')
        plt.title(f'{name} Pareto Frontier')
        plt.savefig(f'results_figures/{name}_pareto.png', dpi=1000)
        '''
