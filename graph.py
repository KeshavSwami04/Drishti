import networkx as nx

def build_graph(calls):
    G = nx.DiGraph()

    for c in calls:
        G.add_edge(c["caller"], c["callee"])

    return G