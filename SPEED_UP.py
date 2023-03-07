import gurobipy as gp
from gurobipy import GRB
import time


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
                    if s_val[i, v] - q_val[i, c] > 10^-model._eps:
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^-model._eps:
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
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
                    if s_val[i, v] - q_val[i, c] > 10^-model._eps:
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    if s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^-model._eps:
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
                    # if v not reachable through c, add cut
                    if (s_val[i, v] - q_val[i, c] > 10^-model._eps and
                            s_val[i, v] - q_val[i, c] == max(s_val[i, v] - q_val[i, d] for d in model._path[v][1:])):
                        if model._lazycuts: model.cbLazy(model._S[i, v] <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] <= model._Q[i, c])
                        model._numcuts += 1
                        break
        if 'CUT2' in model.ModelName:
            for (i, v) in s_val.keys():
                for c in model._path[v][1:]:
                    # if v not reachable through c, add cut
                    if (s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] > 10^-model._eps and
                            s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, c] ==
                            max(s_val[i, v] + sum(s_val[i, u] for u in model._child[v]) - q_val[i, d] for d in model._path[v][1:])):
                        if model._lazycuts: model.cbLazy(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        else: model.cbCut(model._S[i, v] + gp.quicksum(model._S[i, u] for u in model._child[v]) <= model._Q[i, c])
                        model._numcuts += 1
                        break
        end = time.perf_counter()
        model._cbtime += (end - start)
        model._mipnodetime += (end - start)
        # print(f'MIPNODE time: {time.perf_counter()-start}')
        # print(f'Callback MIPNODE {model._numcb}: {model._numcuts} total user SQ3 frac lazy cuts')
