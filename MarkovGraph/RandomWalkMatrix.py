import networkx as nx
import numpy as np

'''
Random Walk Matrix:
W_ii = 1/2
If there exists an edge between i and j W_ij = 1/d(j) where d is the degree of the vertex i  
If there does not exist an edge between i and j W_ij = 0

Random Walk Covering
Sum from 2 to n of g_i * v_i/sqrt(lambda_i) where 
    g_i is a random variabe,
    lambda_i is the ith eigen value
    v_i is the ith eigen vector

Take the infinite norm times the number of nodes and it approximates
the cover time of G 
'''


def random_walk_matrix():
    matrix_size = 16

    # Graph
    G = nx.grid_2d_graph(4, 4)

    # Adjacency matrix
    M = nx.to_numpy_array(G)

    # Save aside the identity for this sized matrix
    Identity = np.identity(matrix_size)

    # Construct the Diagonal Weighted-Degree matrix
    degree_array = list(G.degree())
    degree_vector = np.ndarray((matrix_size, 1))
    for i in range(matrix_size):
        degree_vector[i] = degree_array[i][1]
    D = np.multiply(np.identity(16), degree_vector)

    # Calculate the inverse of the Diagonal Weighted-Degree matrix
    D_inv = np.linalg.inv(D)

    # Finally, the random walk graph!
    # This is one definition of the random walk
    W = np.matmul(M, D_inv)

    # This is the definition of the random walk matrix used by James R. Lee
    # Referred to in Daniel Spielman's book as the "Lazy Walk" matrix
    W = np.add(W, Identity)
    W = np.multiply(W, 0.5)

    # V is the starting position
    V = np.ndarray((matrix_size, 1))
    for i in range(matrix_size):
        V[i] = 0
    V[0] = 1
    X_Prev = np.copy(V)
    X_Current = V
    count = 0
    while count < 250:
        X_Prev = np.copy(X_Current)
        X_Current = np.matmul(W, X_Prev)

        count = count + 1
    print("Previous:")
    print(X_Prev)
    print("Current:")
    print(X_Current)


def main():
    random_walk_matrix()


if __name__ == "__main__":
       # execute only if run as a script
    main()