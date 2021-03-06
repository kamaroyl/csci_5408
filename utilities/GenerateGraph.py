import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GraphException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message

'''
    G - Graph 1
    H - Graph 2
    connectivity - percent of 
'''
def join_graphs_with_connections(G, H, connectivity):
    g_vertex_count = len(G.nodes)
    h_vertex_count = len(H.nodes)
    number_of_vertices = g_vertex_count * h_vertex_count
    connections = np.zeros(number_of_vertices)
    number_of_cross_edges = np.floor(number_of_vertices * connectivity).astype(int)
    connections[:number_of_cross_edges] = 1
    rg = np.random.default_rng()
    rg.shuffle(connections)
    A = nx.disjoint_union(G, H)
    ut = np.triu(nx.to_numpy_array(A), 1)
    position = 0
    for i in range(g_vertex_count):
        for j in range(h_vertex_count):
            ut[i, (g_vertex_count + j)] = connections[position]
            position += 1
    lt = np.transpose(ut)
    symmetric_graph = np.add(ut, lt)
    return symmetric_graph


'''
input:
    n = 7  # row
    m = 7  # col
Output
    01  02  03  04
    05  06  07  08
    09  10  11  12
    13  14  15  16

    (1-2, 2-3, 3-4, 4-1, 5-6, 6-7, 7-8, 8-5
    Currently doesn't work with n != m; needs boundary work
'''
def build_grid(n, m):

    G = nx.Graph()

    node_count = n * m
    nodes = []
    for i in range(node_count):
        nodes.append((i + 1))
    G.add_nodes_from(nodes)
    for i in range(m):  # row
        for j in range(n):  # col
            x = (i*m + j + 2) % ((i+1)*n + 1)
            y = ((i+1)*m + j + 1) % node_count
            if x == 0:
                x = i*m + 1
            if y == 0:
                y = n*m
            G.add_edge(i*m + j + 1, x)
            G.add_edge(i*m + j + 1, y)
    return G


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


def plot_graph_random(G):
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


    '''
        merge sort method with comparison on the eigen value in a tuple of (eigen value, eigen vector)
    '''

def sort(eigen_value_tuples):
    eigen_value_tuples_length = len(eigen_value_tuples)
    if eigen_value_tuples_length > 1:
        middle = eigen_value_tuples_length // 2
        left = eigen_value_tuples[:middle]
        right = eigen_value_tuples[middle:]
        sort(left)
        sort(right)
        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i][0] < right[j][0]:
                eigen_value_tuples[k] = left[i]
                i += 1
            else:
                eigen_value_tuples[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            eigen_value_tuples[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            eigen_value_tuples[k] = right[j]
            j += 1
            k += 1


def main():
    G = nx.from_numpy_matrix(generate_random_graph(100, .5))
    H = nx.from_numpy_matrix(generate_random_graph(100, .5))
    J = nx.from_numpy_matrix(join_graphs_with_connections(G,H,.01))
    plot(J)


if __name__ == "__main__":
       # execute only if run as a script
    main()
