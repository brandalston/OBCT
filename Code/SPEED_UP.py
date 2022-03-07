import gurobipy as gp
from gurobipy import GRB
import time


def fixing(tree, data):
    # Generate unreachable nodes in tree using [0,1] count of datapoints and 'left, right' paths to vertices in the tree
    feats = list(data.columns)
    feats.remove('target')
    data['lefts'] = len(feats) - data[feats].sum(axis=1)
    data['rights'] = data[feats].sum(axis=1)
    data_lefts_rights = {i: [data.at[i, 'lefts'], data.at[i, 'rights']] for i in data.index}
    vertex_lefts_rights = {n: [] for n in tree.DG_prime.nodes if n != 0}
    for level in tree.node_level.keys():
        if level == 0: continue
        for n in tree.node_level[level]:
            if n % 2 == 1:
                for v in [n] + tree.child[n]: vertex_lefts_rights[v].append('left')
            else:
                for v in [n] + tree.child[n]: vertex_lefts_rights[v].append('right')

    data_unreachable = {i: [] for i in data.index}
    node_unreachable = {n: 0 for n in tree.DG_prime.nodes}

    for i in data.index:
        for v in tree.DG_prime.nodes:
            if v == 0: continue
            if data_lefts_rights[i][0] < vertex_lefts_rights[v].count('left') \
                    or data_lefts_rights[i][1] < vertex_lefts_rights[v].count('right'):
                data_unreachable[i].append(v)
                node_unreachable[v] += 1
    return {'data': {key: val for key, val in data_unreachable.items() if len(val) != 0},
            'tree': {key: val for key, val in node_unreachable.items() if val == len(data.index)}}


def warm_start(opt_model, warm_start_values):
    # Tree Assignment Warm Start Values
    # For each node in the warm start tree
    #    If branching node
    #      start value of branching feature = 1
    #      all other features and classes = 0
    #      pruned activation = 0
    #    If classification node
    #       start value of class = 1
    #       all other features and classes = 0
    #       pruned activation = 1
    #    If pruned node
    #       all classes and features = 0
    #       pruned activation = 0
    for n in opt_model.tree.B + opt_model.tree.L:
        if 'branch on feature' in warm_start_values['tree'][n]:
            for f in opt_model.features:
                if f == warm_start_values['tree'][n]['branch on feature']:
                    opt_model.B[n, f].start = 1.0
                else:
                    opt_model.B[n, f].start = 0.0
            for k in opt_model.classes: opt_model.W[n, k].start = 0.0
            opt_model.P[n].start = 0.0
        elif 'class' in warm_start_values['tree'][n]:
            for k in opt_model.classes:
                if k == warm_start_values['tree'][n]['class']:
                    opt_model.W[n, k].start = 1.0
                else:
                    opt_model.W[n, k].start = 0.0
            for f in opt_model.features: opt_model.B[n, f].start = 0.0
            opt_model.P[n].start = 1.0
        elif 'pruned' in warm_start_values['tree'][n]:
            opt_model.P[n].start = 0.0
            for k in opt_model.classes: opt_model.W[n, k].start = 0.0
            for f in opt_model.features: opt_model.B[n, f].start = 0.0

    # Dataset Warm Start Values
    # For each datapoint
    #    Find terminal node and check if datapoint is correctly assigned at node
    #       If yes, s_i,n = 1
    #       If no, s_i,n = 0
    #       otherwise s_i,n = 0
    #   Activate correct source-terminal path nodes for Q
    #       If node in source-terminal path of datapoint, q_i,n: start = 1
    #       otherwise, q_i,n = 0
    if warm_start_values['data']:
        for i in opt_model.datapoints:
            for n in opt_model.tree.B + opt_model.tree.L:
                if n == warm_start_values['data'][i][2][-1] and 'correct' in warm_start_values['data'][i]:
                    opt_model.S[i, n].start = 1.0
                elif n == warm_start_values['data'][i][2][-1]:
                    opt_model.S[i, n].start = 0.0
                else:
                    opt_model.S[i, n].start = 0.0
                if n in warm_start_values['data'][i][2]:
                    opt_model.Q[i, n].start = 1.0
                else:
                    opt_model.Q[i, n].start = 0.0
    return opt_model


def conflict(model, where):
    if where == GRB.Callback.MIPSOL:
        q_val = {key: item for key, item in model.cbGetSolution(model._Q).items() if item > .5}
        branch = {key: item for key, item in model.cbGetSolution(model._B).items() if item > .5}
        for (i, v) in q_val.keys():
            for (n, f) in branch.keys():
                if model.data.at[i, f] == (v % 2) and v in model.successor[n]:
                    model.cbLazy(model._Q[i, v] + model._B[n, f] <= 1)
                    model._numcuts += 1
        # print(f'Callback MIPSOL {model._numcb}: {model._numcuts} conflict lazy constraints')


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
    if 'conflict' in model._modelextras:
        conflict(model, where)


def int1(model, where):
    # Add all violating INT cuts in 1,v path of datapoint terminal node
    if where == GRB.Callback.MIPSOL:
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
    if where == GRB.Callback.MIPSOL:
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
    if where == GRB.Callback.MIPSOL:
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
                    if s_val[i, v] - q_val[i, c] > model._epsval:
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > model._epsval:
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        model._cutsequence[model._numcb] = (model._numcuts, model.cbGet(GRB.Callback.MIPNODE_NODCNT))
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
                    if s_val[i, v] - q_val[i, c] > model._epsval:
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > model._epsval:
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        model._cutsequence[model._numcb] = (model._numcuts, model.cbGet(GRB.Callback.MIPNODE_NODCNT))
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
                    # if v not reachable through c, add cut
                    if (s_val[i, v] - q_val[i, c] > model._epsval and
                            s_val[i, v] - q_val[i, c] == max(s_val[i, v] - q_val[i, d] for d in model._path[v][1:])):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    # if v not reachable through c, add cut
                    if (s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > model._epsval and
                            s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] ==
                            max(s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, d] for d in model._path[v][1:])):
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        model._cutsequence[model.cbGet(GRB.Callback.MIPNODE_NODCNT)] = model._numcuts
        # print(f'MIPNODE time: {time.perf_counter()-start}')
        # print(f'Callback MIPNODE {model._numcb}: {model._numcuts} total user SQ3 frac lazy cuts')
