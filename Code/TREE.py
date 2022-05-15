import networkx as nx
import graphviz


class TREE():

    def __init__(self, h):
        '''
        :param h: height of binary decision tree
        Binary Height h Tree Information including P_v and CHILD_v for each vertex
        Also pass initial information for plotting assigned tree
        '''
        self.height = h
        self.G = nx.generators.balanced_tree(r=2, h=h)
        self.DG = nx.DiGraph(self.G)  # bi-directed version of G
        hidden_edges = [(i, j) for (i, j) in self.DG.edges if i > j]
        self.DG_prime = nx.restricted_view(self.DG, [], hidden_edges)

        self.L = []
        self.B = []
        for node in self.G.nodes:
            if nx.degree(self.G, node) == 1:
                self.L.append(node)
            elif nx.degree(self.G, node) > 1:
                self.B.append(node)
            else:
                print("Error: tree must not have isolated vertices!")
        self.V = self.B + self.L

        self.path = nx.single_source_shortest_path(self.DG_prime, 0)
        self.child = {n: list(nx.descendants(self.DG_prime, n)) for n in self.DG_prime.nodes}
        self.RC = [n for n in self.DG_prime.nodes if n % 2 == 0 and n != 0]
        self.LC = [n for n in self.DG_prime.nodes if 0 != n % 2 == 1]

        self.depth = nx.shortest_path_length(self.DG_prime, 0)
        self.levels = list(range(h + 1))
        self.node_level = {level: [n for n in self.DG_prime.nodes if self.depth[n] == level] for level in self.levels}
        self.direct_ancestor = {n: self.path[n][-2] for n in self.DG_prime.nodes if n != 0}
        self.successor = {n: list(self.DG_prime.successors(n)) for n in self.DG_prime.nodes}

        self.color_map = []
        self.labels = {n: None for n in self.DG_prime.nodes}
        # self.pos = nx.nx_pydot.graphviz_layout(self.DG_prime, prog="dot", root=0)
