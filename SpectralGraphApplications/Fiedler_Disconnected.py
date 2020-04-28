import networkx as nx
import numpy as np
from utilities.GenerateGraph import plot
import matplotlib
matplotlib.use("TkAgg")

def fielder_of_disconnected_graph():
    G = nx.path_graph(6)
    print(nx.nodes(G))
    eigs = sorted(np.linalg.eigvals(nx.laplacian_matrix(G).toarray()))
    print(eigs)
    plot(G)
    G.remove_edge(2,3)
    eigs = sorted(np.linalg.eigvals(nx.laplacian_matrix(G).toarray()))
    print(eigs)
    plot(G)


def main():
    fielder_of_disconnected_graph()


if __name__ == "__main__":
    # execute only if run as a script
    main()