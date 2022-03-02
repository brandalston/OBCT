import networkx as nx
import numpy as np
import csv
import matplotlib.pyplot as plt
import os


def dv_results(model, tree, features, classes, datapoints):
    # Print assigned features, classes, and pruned nodes of tree
    for v in tree.DG_prime.nodes:
        for f in features:
            if model._B[v, f].x > .5:
                print('vertex '+str(v)+' assigned feature '+str(f))
        for k in classes:
            if model._W[v, k].x > 0.5:
                print('vertex '+str(v)+' assigned class '+str(k))
        if model._P[v].x < .5 and 1.0 not in [model._B[v, f].x for f in features]:
            print('vertex '+str(v)+' pruned')

    # Print datapoint paths through tree
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
    # Log model results to .csv file and save .png if applicable
    node_assign(opt_model, tree)
    if tree_check(tree):
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
             opt_model.model.MIPGap, opt_model.model.ObjVal,
             opt_model.model._numcb, opt_model.model._numcuts, opt_model.model._avgcuts,
             opt_model.model._cbtime, opt_model.model._mipsoltime, opt_model.model._mipnodetime, opt_model.eps,
             opt_model.modeltype,
             opt_model.time_limit, rand_state, opt_model.fixed, opt_model.warmstart, opt_model.cc,
             opt_model.single_use, opt_model.level_tree, opt_model.max_features, opt_model.super_feature])
        results.close()


def pareto_frontier(data):
    # Generate pareto frontier
    models = data['Model'].unique()
    name = data['Data'].unique()[0]
    height = max(data['H'].unique())
    dom_points = []
    for model in models:
        sub_data = data.loc[data['Model'] == model]
        best_acc, max_features = -1, 0
        for i in sub_data.index:
            if (sub_data.at[i, 'Out-Acc']) > best_acc and (sub_data.at[i, 'Max Features'] > max_features):
                dom_points.append(i)
                best_acc, max_features = sub_data.at[i, 'Out-Acc'], sub_data.at[i, 'Max Features']
    domed_pts = list(set(data.index).difference(set(dom_points)))
    dominating_points = data.iloc[dom_points, :]
    if domed_pts: dominated_points = data.iloc[domed_pts, :]
    fig = plt.figure()
    axs = fig.add_subplot(111)
    markers = {'MCF1': 'X', 'MCF2': 'p', 'CUT1': 's', 'CUT2': 'P', 'AGHA': '*', 'FlowOCT': '*', 'BendersOCT': 'o'}
    colors = {'MCF1': 'blue', 'MCF2': 'orange', 'CUT1': 'green', 'CUT2': 'red', 'AGHA': 'k', 'FlowOCT': 'k', 'BendersOCT': 'm'}

    for model in models:
        axs.scatter(dominating_points.loc[data['Model'] == model]['Max Features'],
                    dominating_points.loc[data['Model'] == model]['Out-Acc'],
                    marker=markers[model], color=colors[model], label=model)
        if domed_pts: axs.scatter(dominated_points.loc[data['Model'] == model]['Max Features'],
                                  dominated_points.loc[data['Model'] == model]['Out-Acc'],
                                  marker=markers[model], color=colors[model], alpha=0.1)
        z = np.polyfit(data.loc[data['Model'] == model]['Max Features'],
                       data.loc[data['Model'] == model]['Out-Acc'], 3)
        p = np.poly1d(z)
        axs.plot(data.loc[data['Model'] == model]['Max Features'],
                 p(data.loc[data['Model'] == model]['Max Features']),
                 color=colors[model], alpha=0.5)
        axs.legend(loc='lower right')
        axs.set_xlabel('Num. Branching Features')
        axs.xaxis.set_ticks(np.arange(1, max(data['Max Features'].unique())+1, 5))
        axs.set_ylabel('Out-Acc. (%)')
        name = name.replace('_enc','')
        axs.set_title(f'{str(name)} Pareto Frontier')
    plt.savefig(os.getcwd() + '/results_figures/' + str(name) + ' H: '+ str(height)+' Pareto Frontier.png', dpi=300)
    plt.close()
