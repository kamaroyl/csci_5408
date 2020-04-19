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


def main():
    G = nx.from_numpy_matrix(utils.generate_random_graph(10, .1))#nx.Graph()
    H = nx.from_numpy_matrix(utils.generate_random_graph(10, .1))

    print(nx.to_numpy_array(G))

    print(nx.to_numpy_array(H))
    J = nx.disjoint_union(G, H)

    print(nx.to_numpy_array(J))
    return
    utils.plot(J)
    #G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    #G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (6, 7), (7, 8)])
    LG = nx.laplacian_matrix(J)
    eigen_values = LA.eig(LG.toarray())
    eigen_values = (np.around(eigen_values[0]), eigen_values[1])
    print(eigen_values)
    sorted_eigen_values = []

    for i in range(len(eigen_values[0])):
        sorted_eigen_values.append((abs(eigen_values[0][i]), eigen_values[1][i]))

    sort(sorted_eigen_values)

    print("Fiedler Value: " + str(sorted_eigen_values[1][0]))
    print("Fiedler Vector: " + str(sorted_eigen_values[1][1]))


if __name__ == "__main__":
    # execute only if run as a script
    main()
