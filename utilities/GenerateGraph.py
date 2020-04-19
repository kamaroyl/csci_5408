import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GraphException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


def join_graphs_with_connections(G, H, connectivity):
    g_vertex_count = len(G.nodes)
    h_vertex_count = len(H.nodes)
    number_of_vertices = g_vertex_count * h_vertex_count
    connections = np.zeros(number_of_vertices)
    number_of_cross_edges = np.floor(number_of_vertices * connectivity).astype(int)
    connections[:number_of_cross_edges] = 1
    print(g_vertex_count)
    A = nx.disjoint_union(G, H)
    ut = np.triu(nx.to_numpy_array(A), 1)
    print(ut)


def generate_random_graph(size, connectivity):
    if connectivity > 1 or connectivity < 0:
        raise GraphException("Connectivity must be in the interval [0,1]")

    # calculate the total number of entries in the matrix
    total_number_of_entries = size * size

    # calculate the number of edges due to the proportion
    number_of_edges = np.floor(total_number_of_entries * connectivity).astype(int)

    # construct a pool of edges and shuffle
    A = np.zeros(total_number_of_entries)
    A[:number_of_edges] = 1
    rg = np.random.default_rng()
    rg.shuffle(A)

    # reshape to square matrix
    A = A.reshape((size, size))
    # truncate to the upper triangle minus the diagonal
    ut = np.triu(A, 1)
    # transform to lower triangle
    lt = np.transpose(ut)
    # sum to create a symmetric matrix
    symmetric_graph = np.add(ut, lt)
    return symmetric_graph


def plot(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


def main():
    G = nx.from_numpy_matrix(generate_random_graph(10, .1))
    H = nx.from_numpy_matrix(generate_random_graph(11, .1))
    join_graphs_with_connections(G,H,.4)


if __name__ == "__main__":
       # execute only if run as a script
    main()
