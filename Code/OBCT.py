from gurobipy import *
import SPEED_UP
import time


class OBCT:
    
    def __init__(self, data, tree, model, time_limit, target, name, warm_start=None, model_extras=None, log=None):
        """
        Parameters
        data: training data
        tree: input decision tree object of prespecified user height
        model: model to use for connectivity and optimization
        time_limit: gurobi time limit in seconds
        target: target column of training data
        name: name of dataset
        encoding_map: encoding map of one-hot-encoded features of training data
        warm_start: dictionary warm start values
        model_extras: list of model extras
        log: Gurobi log file
        """

        self.modeltype = model
        self.tree = tree
        self.data = data
        self.target = target
        self.datapoints = data.index
        self.dataname = name
        self.modelextras = model_extras
        self.time_limit = time_limit
        self.log = log

        print('Model: '+str(self.modeltype))
        # Binary Encoded Feature Set and Class Set
        self.features = self.data.columns[self.data.columns != self.target]
        self.classes = data[target].unique()

        # Decision Variables
        self.B = 0
        self.W = 0
        self.P = 0
        self.S = 0

        self.Q = 0
        self.Z = 0
        self.T = 0
        self.GEN = 0

        # model extras metrics
        self.repeat_use = 'None'
        self.max_features = 'None'
        self.regularization = 'None'

        # Gurobi optimization parameters
        self.cb_type = self.modeltype[5:]
        if 'CUT' in self.modeltype and len(self.cb_type) == 0:
            self.cb_type = 'GRB'
        self.tree.Lazycuts = False
        self.rootnode = False
        self.eps = 0
        self.cut_constraint = 0
        self.single_terminal = 0
        self.cumul_cut_constraint = 0

        """
        We assume epsilon value of 1e-4 for fractional separation
        Lazy cuts  for integer separation
        User cuts for fractional separation
            - ROOT: We consider fractional cuts at only the root node of the branch and bound tree
        Lazy cuts for integer and fractional separationa in tandem
        All integral lazy cuts are added only at the root node of the branch and bound tree (Lazy = 3 parameter in GRB)
        """
        if 'BOTH' in self.cb_type:
            print('User INT and FRAC lazy cuts')
            self.eps = -4
            if 'ROOT' in self.cb_type:
                self.rootnode = True
            self.tree.Lazycuts = True
        elif 'FRAC' in self.cb_type:
            self.eps = -4
            if 'ROOT' in self.cb_type:
                self.rootnode = True
            print('User FRAC cuts (ROOT: '+str(self.rootnode)+')')
        elif 'INT' in self.cb_type:
            print('User INT lazy = 3 cuts')
        elif 'ALL' in self.cb_type:
            print('ALL integral connectivity constraints')
        elif 'GRB' in self.cb_type:
            print('GRB lazy = 3 constraints')

        # Gurobi model
        self.model = Model(f'{self.modeltype}')
        self.model.Params.TimeLimit = time_limit
        self.model.Params.LogToConsole = 0
        # Use only 1 thread for testing purposes
        self.model.Params.Threads = 1
        # Save Gurobi log to file
        if self.log is not None:
            self.model.Params.LogFile = self.log

        # CUT-1,2 model callback metrics
        self.model._numcuts, self.model._numcb, self.model._cbtime = 0, 0, 0
        self.model._mipsoltime, self.model._mipnodetime, self.model._eps = 0, 0, self.eps
        self.model._cuttype, self.model._lazycuts, self.model._rootnode = self.cb_type, self.tree.Lazycuts, self.rootnode
        self.model._modeltype, self.model._path, self.model._child = self.modeltype, self.tree.path, self.tree.child

        # Warm start (if applicable)
        if warm_start is not None:
            self.warmstart = True
            self.wsv = warm_start
            self.calibration = 'warm_start'
        else:
            self.warmstart = False

    ###########################################
    # MIP FORMULATIONS
    ###########################################
    def formulation(self):
        """
        Formulation of MIP model with connectivity constraints according to model type chosen by user
        returns gurobi model
        """

        # Branching vertex vars
        self.B = self.model.addVars(self.tree.V, self.features, vtype=GRB.BINARY, name='B')
        # Classification vertex vars
        self.W = self.model.addVars(self.tree.V, self.classes, vtype=GRB.BINARY, name='W')
        # Pruned vertex vars
        self.P = self.model.addVars(self.tree.V, vtype=GRB.CONTINUOUS, lb=0, name='P')
        # Datapoint terminal vertex vars
        self.S = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='S')

        """ Model Objective and BASE constraints """
        # Objective: Maximize the number of correctly classified datapoints
        # Max sum(S[i,v], i in I, v in V\1)
        self.model.setObjective(
            quicksum(self.S[i, v] for i in self.datapoints for v in self.tree.V if v != 0),
            GRB.MAXIMIZE)

        # Pruned vertices not assigned to class
        # P[v] = sum(W[v,k], k in K) for v in V
        self.model.addConstrs(self.P[v] == quicksum(self.W[v, k] for k in self.classes)
                              for v in self.tree.V)

        # Vertices must be branched, assigned to class, or pruned
        # sum(B[v,f], f in F) + sum(P[u], u in path[v]) = 1 for v in V
        self.model.addConstrs(quicksum(self.B[v, f] for f in self.features) +
                              quicksum(self.P[u] for u in self.tree.path[v]) == 1
                              for v in self.tree.V)

        # Cannot branch on leaf vertex
        # B[v,f] = 0 for v in L, f in F
        for v in self.tree.L:
            for f in self.features:
                self.B[v, f].ub = 0

        # Terminal vertex of datapoint matches datapoint class
        # S[i,v] <= W[v,y_i=k] for v in V, k in K
        for v in self.tree.V:
            self.model.addConstrs(self.S[i, v] <= self.W[v, self.data.at[i, self.target]]
                                  for i in self.datapoints)

        """ MCF1 Model Connectivity Constraints """
        if 'MCF1' in self.modeltype:
            # Flow vars
            self.Z = self.model.addVars(self.datapoints, self.tree.DG_prime.edges, vtype=GRB.CONTINUOUS, lb=0, name='Z')
            # Source-terminal vertex vars
            self.Q = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='Q')
            # Sink vars
            self.T = self.model.addVars(self.datapoints, vtype=GRB.BINARY, name='T')

            # non-(root,t) vertex selected in datapoint 0-t path if vertex receives flow
            # Z[i,a(v),v] <= Q[i,v] for i in I, v in V\1
            for v in self.tree.V:
                if v == 0: continue
                self.model.addConstrs(quicksum(self.Z[i, u, v] for u in self.tree.DG.neighbors(v) if u < v) <= self.Q[i, v]
                                      for i in self.datapoints)

            # flow conservation at v
            # Z[i,a(v),v] = Z[i,v,l(v)] + Z[i,v,r(v) + S[i,v] for i in I, v in V\1
            # Z[i,1,l(1)] + Z[i,1,r(1)] + S[i,1] == Q[i,t] for i in I
            for v in self.tree.V:
                if v != 0:
                    self.model.addConstrs(quicksum(self.Z[i, u, v] for u in self.tree.DG.neighbors(v) if u < v) ==
                                          quicksum(self.Z[i, v, u] for u in self.tree.DG.neighbors(v) if v < u) + self.S[
                                              i, v]
                                          for i in self.datapoints)
                else:
                    self.model.addConstrs(
                        self.S[i, v] + quicksum(self.Z[i, v, u] for u in self.tree.DG.neighbors(v) if v < u) ==
                        self.T[i] for i in self.datapoints)

            # left right branching for vertices selected in 0-t path for datapoint
            # q[i,l(v)] <= sum(b[v,f], f if x[i,f]=0) for all i in I, v in N
            # q[i,r(v)] <= sum(b[v,f], f if x[i,f]=1) for all i in I, v in N
            for v in self.tree.B:
                for u in self.tree.DG_prime.neighbors(v):
                    if u % 2 == 1:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 0)
                            for i in self.datapoints)
                    elif u % 2 == 0:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 1)
                            for i in self.datapoints)

        """ MCF2 Model Connectivity Constraints """
        if 'MCF2' in self.modeltype:
            # Flow vars
            self.Z = self.model.addVars(self.datapoints, self.tree.V, self.tree.DG_prime.edges, vtype=GRB.CONTINUOUS,
                                        lb=0, name='Z')
            # Source-terminal vertex vars
            self.Q = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='Q')

            # if datapoint i correctly classified at vertex v,
            # then flow of type datapoint i heading towards vertex v originates from root
            # Z[iv,1,l(1)] + Z[iv,1,r(1)] = S[i,v] for all i in I, v in V\1
            for v in self.tree.V:
                if v == 0: continue
                self.model.addConstrs(
                    quicksum(self.Z[i, v, 0, u] for u in self.tree.DG.neighbors(0) if 0 < u) == self.S[i, v]
                    for i in self.datapoints)

            # flow conservation of datapoint i heading towards vertex v at vertex u
            # Z[iv,u,l(u)] + Z[iv,u,r(u)] - Z[iv,a(u),u] = 0 for all i in I, v in V\1, u in V\{i,v}
            for u in self.tree.V:
                self.model.addConstrs(
                    quicksum(self.Z[i, v, u, j] for j in self.tree.DG.neighbors(u) if (u != 0 and v != u < j)) -
                    quicksum(self.Z[i, v, j, u] for j in self.tree.DG.neighbors(u) if (j < u != v and u != 0)) == 0
                    for v in self.tree.V if v != 0 for i in self.datapoints)

            # if any flow of datapoint i enters v,
            # then v is selected on a 0-terminal path for datapoint i
            # sum(Z[iu,a(v),v], u in V\1) <= Q[i,v] for all in I, v in V\1
            for v in self.tree.V:
                if v == 0: continue
                self.model.addConstrs(
                    quicksum(
                        self.Z[i, u, j, v] for j in self.tree.DG.neighbors(v) if j < v for u in self.tree.V if u != 0) <=
                    self.Q[i, v] for i in self.datapoints)

            # if vertex v selected as terminal node for datapoint i
            # then ancestor of vertex v receives flow
            # Z[iv,a(v),v] == S[i,v] for all i in I, v in V
            for v in self.tree.V:
                if v == 0: continue
                self.model.addConstrs(self.Z[i, v, self.tree.path[v][-2], v] == self.S[i, v] for i in self.datapoints)

            # left right branching for vertices selected in 0-t path for datapoint
            # q[i,l(v)] <= sum(b[v,f], f if x[i,f]=0) for all i in I, v in N
            # q[i,r(v)] <= sum(b[v,f], f if x[i,f]=1) for all i in I, v in N
            for v in self.tree.B:
                for u in self.tree.DG_prime.neighbors(v):
                    if u % 2 == 1:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 0)
                            for i in self.datapoints)
                    elif u % 2 == 0:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 1)
                            for i in self.datapoints)

            # each datapoint has at most one terminal vertex
            # sum(s[i, v] for v in V]) == 1 for i in I
            self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V) <= 1
                                  for i in self.datapoints)

        """ CUT1 Model Connectivity Constraints """
        if 'CUT1' in self.modeltype:
            # Source-terminal vertex vars
            self.Q = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='Q')

            # left right branching for vertices selected in 0-t path for datapoint
            # q[i,l(v)] <= sum(b[v,f], f if x[i,f]=0) for all i in I, v in N
            # q[i,r(v)] <= sum(b[v,f], f if x[i,f]=1) for all i in I, v in N
            # if (self.modelextras is None) or ('conflict_constraints' not in self.modelextras):
            for v in self.tree.B:
                for u in self.tree.DG_prime.neighbors(v):
                    if u % 2 == 1:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 0)
                            for i in self.datapoints)
                    elif u % 2 == 0:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 1)
                            for i in self.datapoints)

            if 'BOTH' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(
                    quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                    for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
            elif 'FRAC' in self.cb_type:
                # terminal vertex of datapoint must be in reachable path
                # s[i,v] <= q[i,c] for i in I, for c in path[v], for v in V / {1}
                self.cut_constraint = self.model.addConstrs(self.S[i, v] <= self.Q[i, c] for i in self.datapoints
                                                            for v in self.tree.V if v != 0
                                                            for c in self.tree.path[v][1:])
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        for c in self.tree.path[v][1:]:
                            self.cut_constraint[i, v, c].lazy = 3
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(
                    quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                    for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
            elif 'INT' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(
                    quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                    for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
            elif 'GRB' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(
                    quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                    for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
                # terminal vertex of datapoint must be in reachable path
                # s[i,v] <= q[i,c] for i in I, for c in path[v], for v in V / {1}
                self.cut_constraint = self.model.addConstrs(self.S[i, v] <= self.Q[i, c] for i in self.datapoints
                                                            for v in self.tree.V if v != 0
                                                            for c in self.tree.path[v][1:])
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        for c in self.tree.path[v][1:]:
                            self.cut_constraint[i, v, c].lazy = 3
            elif 'ALL' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V) <= 1
                                      for i in self.datapoints)
                # terminal vertex of datapoint must be in reachable path
                # s[i,v] <= q[i,c] for i in I, for c in path[v], for v in V / {1}
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        self.model.addConstrs(self.S[i, v] <= self.Q[i, c] for c in self.tree.path[v][1:])

        """ CUT2 Model Connectivity Constraints """
        if 'CUT2' in self.modeltype:
            # Source-terminal vertex vars
            self.Q = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='Q')

            # left right branching for vertices selected in 0-t path for datapoint
            # q[i,l(v)] <= sum(b[v,f], f if x[i,f]=0) for all i in I, v in N
            # q[i,r(v)] <= sum(b[v,f], f if x[i,f]=1) for all i in I, v in N
            for v in self.tree.B:
                for u in self.tree.DG_prime.neighbors(v):
                    if u % 2 == 1:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 0)
                            for i in self.datapoints)
                    elif u % 2 == 0:
                        self.model.addConstrs(
                            self.Q[i, u] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 1)
                            for i in self.datapoints)

            # separation procedure
            if 'BOTH' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(
                    quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                    for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
            elif 'FRAC' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                                                             for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
                # terminal vertex of datapoint must be in reachable path for vertex and all children
                # s[i, v] + sum(s[i, u] for u in CHILD[v]) <= q[i, c] for i in I, for c in path[v], for v in V / {1}
                self.cumul_cut_constraint = self.model.addConstrs(
                    self.S[i, v] + quicksum(self.S[i, u] for u in self.tree.child[v])
                    <= self.Q[i, c] for i in self.datapoints
                    for v in self.tree.V if v != 0
                    for c in self.tree.path[v][1:])
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        for c in self.tree.path[v][1:]:
                            self.cumul_cut_constraint[i, v, c].lazy = 3
            elif 'INT' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                                                             for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
            elif 'GRB' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V]) == 1 for i in I
                self.single_terminal = self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V if v != 0) <= 1
                                                             for i in self.datapoints)
                for i in self.datapoints:
                    self.single_terminal[i].lazy = 3
                # terminal vertex of datapoint must be in reachable path for vertex and all children
                # s[i, v] + sum(s[i, u] for u in CHILD[v]) <= q[i, c] for i in I, for c in path[v], for v in V / {1}
                self.cumul_cut_constraint = self.model.addConstrs(
                    self.S[i, v] + quicksum(self.S[i, u] for u in self.tree.child[v])
                    <= self.Q[i, c] for i in self.datapoints
                    for v in self.tree.V if v != 0
                    for c in self.tree.path[v][1:])
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        for c in self.tree.path[v][1:]:
                            self.cumul_cut_constraint[i, v, c].lazy = 3
            elif 'ALL' in self.cb_type:
                # each datapoint has at most one terminal vertex
                # sum(s[i, v] for v in V) == 1 for i in I
                self.model.addConstrs(quicksum(self.S[i, v] for v in self.tree.V) <= 1 for i in self.datapoints)
                # terminal vertex of datapoint must be in reachable path for vertex and all children
                # s[i, v] + sum(s[i, u] for u in CHILD[v]) <= q[i, c] for i in I, for c in path[v], for v in V / {1}
                for i in self.datapoints:
                    for v in self.tree.V:
                        if v == 0: continue
                        self.model.addConstrs(self.S[i, v] + quicksum(self.S[i, u] for u in self.tree.child[v]) <=
                                              self.Q[i, c] for c in self.tree.path[v][1:])

        """ FlowOCT Model Connectivity Constraints (only used for Pareto Frontier Purposes) """
        if 'FOCT' in self.modeltype:
            # Flow vars
            self.Z = self.model.addVars(self.datapoints, self.tree.DG_prime.edges, vtype=GRB.CONTINUOUS, lb=0, name='Z')
            # Flow generator at node zero vars
            self.GEN = self.model.addVars(self.datapoints, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='G')

            # generate flow at root vertex, flow conservation at non-leaf vertices
            for v in self.tree.V:
                if v != 0:
                    self.model.addConstrs(quicksum(self.Z[i, u, v] for u in list(self.tree.DG.neighbors(v)) if u < v) ==
                                          quicksum(self.Z[i, v, u] for u in list(self.tree.DG.neighbors(v)) if v < u) + self.S[i, v]
                                          for i in self.datapoints)
                else:
                    self.model.addConstrs(quicksum(self.Z[i, v, u] for u in list(self.tree.DG.neighbors(v)) if v < u) +
                                          self.S[i, v] == self.GEN[i]
                                          for i in self.datapoints)

            # flow conservation at leaf vertices
            for n in self.tree.L:
                for (u, v) in list(self.tree.DG_prime.edges(n)):
                    self.model.addConstrs(self.Z[i, u, v] == self.S[i, v] for i in self.datapoints)

            # left right branching for vertices selected in s-t path for datapoint
            for v in self.tree.B:
                for (u, n) in list(self.tree.DG_prime.edges(v)):
                    if n % 2 == 1:
                        self.model.addConstrs(self.Z[i, v, n] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 0)
                                              for i in self.datapoints)
                    elif n % 2 == 0:
                        self.model.addConstrs(self.Z[i, v, n] <= quicksum(self.B[v, f] for f in self.features if self.data.at[i, f] == 1)
                                              for i in self.datapoints)

        # pass to model DV for callback purposes
        self.model._Q = self.Q
        self.model._S = self.S
        self.model._B = self.B
        self.model._W = self.W
        self.model._P = self.P
        self.model._modeltype = self.model

    ###########################################
    # Model Extras
    ###########################################
    def extras(self):
        # feature used once
        if any((match := elem).startswith('repeat_use') for elem in self.modelextras):
            self.repeat_use = int(re.sub("[^0-9]", "", match))
            print('Each feature used at most ' + str(self.repeat_use) + ' times')
            self.model.addConstrs(quicksum(self.B[n, f] for n in self.tree.V) <= self.repeat_use for f in self.features)

        # number of maximum branching nodes
        if any((match := elem).startswith('max_features') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print('No more than ' + str(self.max_features) + ' feature(s) used')
            self.model.addConstr(
                quicksum(self.B[v, f] for f in self.features for v in self.tree.B) <= self.max_features)

        # exact number of branching nodes
        if any((match := elem).startswith('num_features') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print(str(self.max_features)+' feature(s) used')
            self.model.addConstr(
                quicksum(self.B[v, f] for f in self.features for v in self.tree.B) == self.max_features)

        # regularization
        if any((match := elem).startswith('regularization') for elem in self.modelextras):
            self.regularization = int(re.sub("[^0-9]", "", match))
            print('Regularization value of '+str(self.regularization)+' applied at classification vertices')
            self.model.addConstrs(quicksum(self.S[i, v] for i in self.datapoints) >= self.regularization * self.P[v]
                                  for v in self.tree.V)

    ###########################################
    # Model Warm Start
    ###########################################
    def warm_start(self):
        print('Updating model with warm start values')
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

        for v in self.tree.V:
            if 'branch on feature' in self.wsv['tree'].DG_prime.nodes[v]:
                for f in self.features:
                    if f == self.wsv['tree'].DG_prime.nodes[v]['branch on feature']:
                        self.B[v, f].Start = 1.0
                    else:
                        self.B[v, f].Start = 0.0
                for k in self.classes: self.W[v, k].Start = 0.0
                self.P[v].Start = 0.0
            elif 'class' in self.wsv['tree'].DG_prime.nodes[v]:
                for k in self.classes:
                    if k == self.wsv['tree'].DG_prime.nodes[v]['class']:
                        self.W[v, k].Start = 1.0
                    else:
                        self.W[v, k].Start = 0.0
                for f in self.features: self.B[v, f].Start = 0.0
                self.P[v].Start = 1.0
            elif 'pruned' in self.wsv['tree'].DG_prime.nodes[v]:
                self.P[v].Start = 0.0
                for k in self.classes: self.W[v, k].Start = 0.0
                for f in self.features: self.B[v, f].Start = 0.0

        # Dataset Warm Start Values
        # For each datapoint
        #    Find terminal node and check if datapoint is correctly assigned at node
        #       If yes, s_i,v = 1
        #       If no, s_i,v = 0
        #       otherwise s_i,v = 0
        #   Activate correct source-terminal path nodes for Q
        #       If node in source-terminal path of datapoint, q_i,n: start = 1
        #       otherwise, q_i,v = 0
        if self.wsv['data']:
            for i in self.datapoints:
                for v in self.tree.V:
                    if v == self.wsv['data'][i][2][-1] and 'correct' in self.wsv['data'][i]:
                        self.S[i, v].Start = 1.0
                    elif v == self.wsv['data'][i][2][-1]:
                        self.S[i, v].Start = 0.0
                    else:
                        self.S[i, v].Start = 0.0
                    if v in self.wsv['data'][i][2]:
                        self.Q[i, v].Start = 1.0
                    else:
                        self.Q[i, v].Start = 0.0

    ###########################################
    # Model Optimization
    ###########################################
    def optimization(self):
        print(f'Optimizing model')
        # Solve model with callback if applicable
        if 'CUT' not in self.modeltype:
            self.model.optimize()
        elif 'CUT' in self.modeltype:
            if 'FRAC' in self.cb_type:
                # User cb.Cut FRAC S-Q cuts
                self.model.Params.PreCrush = 1
                if '1' in self.cb_type: self.model.optimize(SPEED_UP.frac1)
                if '2' in self.cb_type: self.model.optimize(SPEED_UP.frac2)
                if '3' in self.cb_type: self.model.optimize(SPEED_UP.frac3)
            if 'BOTH' in self.cb_type:
                # User cb.Lazy FRAC and INT S-Q cuts
                self.model.Params.LazyConstraints = 1
                self.model.Params.PreCrush = 1
                self.model.optimize(SPEED_UP.both)
            if 'INT' in self.cb_type:
                # User cb.Lazy INT S-Q cuts
                self.model.Params.LazyConstraints = 1
                if '1' in self.cb_type: self.model.optimize(SPEED_UP.int1)
                if '2' in self.cb_type: self.model.optimize(SPEED_UP.int2)
                if '3' in self.cb_type: self.model.optimize(SPEED_UP.int3)
            if 'GRB' in self.cb_type:
                self.model.optimize()
            if 'ALL' in self.cb_type:
                self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print('Optimal solution found in '+str(round(self.model.Runtime, 2))+'s. ('+str(time.strftime("%I:%M %p", time.localtime()))+')\n')
        else: print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')\n')

        # Uncomment to print tree assignments and training set source-terminal path
        # RESULTS.dv_results(self.model, self.tree, self.features, self.classes, self.datapoints)

        if self.model._numcb > 0:
            self.model._avgcuts = self.model._numcuts / self.model._numcb
        else:
            self.model._avgcuts = 0