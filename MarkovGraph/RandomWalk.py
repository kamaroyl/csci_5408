import matplotlib.pyplot as plt
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
def main():
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

    print(W)

    # V is the starting position
    V = np.ndarray((matrix_size, 1))
    for i in range(matrix_size):
        V[i] = 0
    V[0] = 1
    X_1 = np.matmul(W, V)
    X_2 = np.matmul(W, X_1)
    print(X_1)
    print(X_2)
    #print(np.array_str(W, precision=2))
    #plot(G)


def plot(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
       # execute only if run as a script
    main()
