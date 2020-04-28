import networkx as nx
import numpy as np


def fielder_value():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    L = nx.laplacian_matrix(G).toarray()
    print("No edges")
    print(sorted(np.linalg.eigvals(L)))
    G.add_edge(1, 2)
    L = nx.laplacian_matrix(G).toarray()
    print("One Edge")
    print(sorted(np.linalg.eigvals(L)))
    G.add_edge(2, 3)
    L = nx.laplacian_matrix(G).toarray()
    print("Two Edge")
    print(sorted(np.around(np.linalg.eigvals(L))))
    print("one two skip a few ...")
    G = nx.complete_graph(5)
    L = nx.laplacian_matrix(G).toarray()
    print("Complete Graph")
    print(sorted(np.around(np.linalg.eigvals(L))))


def main():
    fielder_value()

if __name__ == "__main__":
    # execute only if run as a script
    main()