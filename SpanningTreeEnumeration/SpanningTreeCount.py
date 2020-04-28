import networkx as nx
import numpy as np
from utilities.GenerateGraph import plot
import matplotlib
matplotlib.use("TkAgg")


def count_uniquely_labelled_trees():
    G= nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5)])
    plot(G)
    L = nx.laplacian_matrix(G).toarray()
    eig_vals = sorted(np.linalg.eigvals(L))
    non_zero_eigen_values = []
    for eigen_value in eig_vals:
        if eigen_value > 0:
            non_zero_eigen_values.append(eigen_value)

    tao = np.divide(np.prod(non_zero_eigen_values),G.number_of_nodes() )
    print(tao)


def main():
    count_uniquely_labelled_trees()

if __name__ == "__main__":
    # execute only if run as a script
    main()