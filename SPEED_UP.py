'''
This file contains the utility functions necessary for our models Callback functions as well as the
models found in the paper ''[Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)''.
and publically available on https://github.com/pashew94/StrongTree/Code/StrongTree/utils.py
Code is taken directly from https://github.com/pashew94/StrongTree/Code/StrongTree/utils.py
All rights and ownership are to the original owners.
'''

import gurobipy as gp
from gurobipy import GRB
import time

def frac1(model, where):
    # Add all violating FRAC cuts in 1,v path of datapoint terminal node in branch and bound tree
    if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        start = time.perf_counter()
        if model._rootnode:
            # Only add cuts at root-node of branch and bound tree
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
        model._numcb += 1
        q_val = model.cbGetNodeRel(model._Q)
        s_val = model.cbGetNodeRel(model._S)
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        # print(f'Callback MIPNODE {model._numcb}: {model._numcuts} total user SQ1 frac lazy cuts')


def frac2(model, where):
    # Add the first found violating FRAC cut in 1,v path of datapoint terminal node in branch and bound tree
    if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        start = time.perf_counter()
        if model._rootnode:
            # Only add cuts at root-node of branch and bound tree
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
        model._numcb += 1
        q_val = model.cbGetNodeRel(model._Q)
        s_val = model.cbGetNodeRel(model._S)
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        # print(f'Callback MIPNODE {model._numcb}: {model._numcuts} total user SQ1 frac lazy cuts')


def frac3(model, where):
    # Add most violating FRAC cut in 1,v path of datapoint terminal node in branch and bound tree
    # If more than one most violating cut exists add the one closest to the root of DT
    if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        start = time.perf_counter()
        if model._rootnode:
            # Only add cuts at root-node of branch and bound tree
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
        model._numcb += 1
        q_val = model.cbGetNodeRel(model._Q)
        s_val = model.cbGetNodeRel(model._S)
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^(-model._eps):
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        # print(f'Callback MIPNODE {model._numcb}: {model._numcuts} total user SQ1 frac lazy cuts')


def int1(model, where):
    # Add all violating INT cuts in 1,v path of datapoint terminal node

    if where == GRB.Callback.MIPSOL or where == GRB.Callback.MULTIOBJ:
        print('IN MIPSOL')
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        print('Q')
        print(q_val)
        print('\nS')
        print(s_val)
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] > q_val[i, c]:
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c]:
                        model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT1 cuts')


def int2(model, where):
    # Add the first found INT cut in 1,v path of datapoint terminal node
    if where == GRB.Callback.MIPSOL or where == GRB.Callback.MULTIOBJ:
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] > q_val[i, c]:
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c]:
                        model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT2 cuts')


def int3(model, where):
    # Add the most violating INT cut in 1,v path of datapoint terminal node
    # If more than one most violating cut exists, add the one closest to the root of DT
    if where == GRB.Callback.MIPSOL or where == GRB.Callback.MULTIOBJ:
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if (s_val[i, v] > q_val[i, c] and
                            s_val[i, v] - q_val[i, c] == max(s_val[i, v] - q_val[i, d] for d in model._path[v][1:])):
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if (s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c] and
                            s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] ==
                            max(s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, d] for d in model._path[v][1:])):
                        model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT3 cuts')


def both(model, where):
    if 'F1' in model.ModelName:
        frac1(model, where)
    if 'F2' in model.ModelName:
        frac2(model, where)
    if 'F3' in model.ModelName:
        frac3(model, where)
    if 'I1' in model.ModelName:
        int1(model, where)
    if 'I2' in model.ModelName:
        int2(model, where)
    if 'I3' in model.ModelName:
        int3(model, where)


def get_node_status(grb_model, b, beta, p, n):
    '''
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a Num-Tree-size node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?

    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of Num-Tree-size decision variable b
    :param beta: The values of prediction decision variable w
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, Num-Tree-size, selected_feature, leaf, value

    pruned=1 iff the node is pruned
    Num-Tree-size = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    '''
    tree = grb_model.tree
    mode = grb_model.mode
    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        if mode == "regression":
            value = beta[n, 1]
        elif mode == "classification":
            for k in grb_model.labels:
                if beta[n, k] > 0.5:
                    value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # Num-Tree-size
            for f in grb_model.cat_features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def get_left_exp_integer(master, b, n, i):
    lhs = gp.quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = gp.quicksum(-master.m[i] * master.b[n, f] for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, p, beta, n, i):
    label_i = master.data.at[i, master.label]

    if master.mode == "classification":
        lhs = -1 * master.beta[n, label_i]
    elif master.mode == "regression":
        # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
        if master.m[i] * p[n] - label_i * p[n] + beta[n, 1] < master.m[i] * p[n] + label_i * p[n] - beta[n, 1]:
            lhs = -1 * (master.m[i] * master.p[n] - label_i * master.p[n] + master.beta[n, 1])
        else:
            lhs = -1 * (master.m[i] * master.p[n] + label_i * master.p[n] - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, p, beta, left, right, target, i):
    lhs = gp.LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, p, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(master, b, p, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(master, b, beta, p, current)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
            if master.mode == "regression":
                subproblem_value = master.m[i] - abs(current_value - label_i)
            elif master.mode == "classification" and beta[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = master.tree.get_left_children(current)

    return subproblem_value, left, right, target


def bendersoct(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    mode = model._master.mode

    local_eps = 0.0001
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        p = model.cbGetSolution(model._vars_p)
        beta = model.cbGetSolution(model._vars_beta)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            if mode == "classification":
                g_threshold = 0.5
            elif mode == "regression":
                g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem(model._master, b, p, beta, i)
                if mode == "classification" and subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)
                elif mode == "regression" and ((subproblem_value + local_eps) < g[i]):
                    added_cut = 1
                    lhs = get_cut_integer(model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time
"""
def lp1(model, where):
    # Add all violating INT cuts in 1,v path of datapoint terminal node
    if where == GRB.Callback.MIPSOL:
        print('IN MIPSOL')
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        print('Q')
        print(q_val)
        print('\nS')
        print(s_val)
        for i in model.datapoints:
            if sum(s_val[i, v] for v in model._V) > 1:
                model.cbLazy(gp.quicksum(model._S[i, v] for v in model._V) <= 1)
                model._numcuts += 1
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] > q_val[i, c]:
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c]:
                        model.cbLazy(
                            model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT1 cuts')


def lp2(model, where):
    # Add all violating INT cuts in 1,v path of datapoint terminal node
    if where == GRB.Callback.MIPSOL:
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        print('Q')
        print(q_val)
        print('\nS')
        print(s_val)
        for i in model.datapoints:
            if sum(s_val[i,v] for v in model._V) > 1:
                model.cbLazy(gp.quicksum(model._S[i, v] for v in model._V) <= 1)
                model._numcuts += 1
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] > q_val[i, c]:
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c]:
                        model.cbLazy(
                            model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT1 cuts')


def lp3(model, where):
    # Add all violating INT cuts in 1,v path of datapoint terminal node
    if where == GRB.Callback.MIPSOL:
        start = time.perf_counter()
        model._numcb += 1
        q_val = model.cbGetSolution(model._Q)
        s_val = {key: item for key, item in model.cbGetSolution(model._S).items() if item > .5}
        print('Q')
        print(q_val)
        print('\nS')
        print(s_val)
        for i in model.datapoints:
            if sum(s_val[i,v] for v in model._V) > 1:
                model.cbLazy(gp.quicksum(model._S[i, v] for v in model._V) <= 1)
                model._numcuts += 1
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if (s_val[i, v] > q_val[i, c] and
                            s_val[i, v] - q_val[i, c] == max(s_val[i, v] - q_val[i, d] for d in model._path[v][1:])):
                        model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if (s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) > q_val[i, c] and
                            s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] ==
                            max(s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, d] for d in
                                model._path[v][1:])):
                        model.cbLazy(
                            model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipsoltime += (end - start)
        # print(f'MIPSOL time: {time.perf_counter()-start}')
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} total user INT1 cuts')

def CUT1_vs_CUT2(model, where):
    if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        start = time.perf_counter()
        if model._rootnode:
            # Only add cuts at root-node of branch and bound tree
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
        model._numcb += 1
        q_val = model.cbGetNodeRel(model._Q)
        s_val = model.cbGetNodeRel(model._S)
        if 'CUT1' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    # if v not reachable through c, add cut
                    if (s_val[i, v] - q_val[i, c] > 10^(-model._eps) and
                            s_val[i, v] - q_val[i, c] == max(s_val[i, v] - q_val[i, d] for d in model._path[v][1:])):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
"""
