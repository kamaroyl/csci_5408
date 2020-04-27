import networkx as nx
import numpy as np


def estimate_graph_color(G):
    eigen_values = sorted(np.linalg.eigvals(nx.to_numpy_array(G)))
    eigen_length = len(eigen_values)
    mu_n = eigen_values[eigen_length - 1]
    mu_1 = 0
    for i in range(eigen_length):
        mu_1 = eigen_values[i]
        if mu_1 != 0:
            break
    if mu_1 != 0:
        return 1 - mu_n//mu_1
    else:
        print("couldn't find")
        return 0


def main():
    '''
    Not a tight bound;
    '''
    G = nx.petersen_graph()
    print("Petersen Graph CHI upper bound:")
    print(estimate_graph_color(G))
    G100 = nx.complete_graph(100)
    print("K_100 CHI upper bound:")
    print(estimate_graph_color(G100))


if __name__ == "__main__":
    # execute only if run as a script
    main()