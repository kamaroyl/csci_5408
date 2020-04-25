import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as LA
import numpy as np
np.set_printoptions(threshold=np.inf)
import utilities as utils

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


def plot_partition(G, partition):
    pos = nx.spring_layout(G)
    print(partition[0])
    print(partition[1])
    A = G.subgraph(partition[0])
    B = G.subgraph(partition[1])
    nx.draw_networkx_nodes(G, pos, nodelist=partition[0], node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_nodes(G, pos, nodelist=partition[1], node_color='b', node_size=100, alpha=1)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


def partition_fiedler_0(fiedler_vector):
    partition = ([], [])
    fiedler_vector_length = len(fiedler_vector)

    for i in range(fiedler_vector_length):
        if fiedler_vector[i] < 0:
            partition[0].append(i)
        else:
            partition[1].append(i)

    return partition


def partition_fiedler(fiedler_vector):
    fiedler_vector_copy = fiedler_vector.copy()
    fiedler_vector_copy = sorted(fiedler_vector_copy)
    print("sorted fiedler vector")
    print(fiedler_vector_copy)
    fiedler_vector_length = len(fiedler_vector_copy)

    if fiedler_vector_length & 1 == 1:
        index = fiedler_vector_length >> 1
        median = fiedler_vector_copy[index]
    else:
        median = (fiedler_vector_copy[(fiedler_vector_length - 1)//2] + fiedler_vector_copy[fiedler_vector_length//2])/2

    partition = ([], [])
    print("Median " + str(median))
    for i in range(fiedler_vector_length):
        #print("fiedler: " + str(fiedler_vector[i]) + " median: " + str(median))
        #print("fiedler_vector[i] < median " + str(fiedler_vector[i] < median))
        if fiedler_vector[i] < median:
            partition[0].append(i)
        else:
            partition[1].append(i)

    return partition


def make_suspicious_graph():
    G = nx.from_numpy_matrix(utils.generate_random_graph(100, .75))
    H = nx.from_numpy_matrix(utils.generate_random_graph(100, .75))
    return nx.from_numpy_matrix(utils.join_graphs_with_connections(G, H, .01))


def main():
    J = make_suspicious_graph()

    utils.plot_graph_random(J)

    LG = nx.laplacian_matrix(J)
    eigen_values = LA.eig(LG.toarray())
    eigen_values = (np.around(eigen_values[0]), eigen_values[1])
    sorted_eigen_values = []

    for i in range(len(eigen_values[0])):
        sorted_eigen_values.append((abs(eigen_values[0][i]), eigen_values[1][i]))

    sort(sorted_eigen_values)
    internal_fiedler_vector = nx.fiedler_vector(J)
    print("Fiedler Value: " + str(sorted_eigen_values[1][0]))
    print("Fiedler Vector: " + str(sorted_eigen_values[1][1]))
    print("Internal Fiedler Vector: " + str())
    fiedler_partition = partition_fiedler(internal_fiedler_vector)
    print("J Nodes")
    print(J.nodes)
    print("Fiedler Partition")
    print(fiedler_partition)
    plot_partition(J, fiedler_partition)


if __name__ == "__main__":
    # execute only if run as a script
    main()
