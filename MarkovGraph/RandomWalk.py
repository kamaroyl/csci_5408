import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


def approximate_cover(G):
    n = G.number_of_nodes()
    # Adjacency matrix
    M = nx.to_numpy_array(G)
    # Save aside the identity for this sized matrix
    I = np.identity(n)

    # Construct the Diagonal Weighted-Degree matrix
    degree_array = list(G.degree())
    degree_vector = np.ndarray((n, 1))
    for i in range(n):
        degree_vector[i] = degree_array[i][1]
    D = np.multiply(I, degree_vector)

    # Calculate the inverse of the Diagonal Weighted-Degree matrix
    D_inv = np.linalg.inv(D)

    # Finally, the random walk graph!
    # This is one definition of the random walk
    W = np.matmul(M, D_inv)
    L = I - W

    # 0 = lambda_1 <=lambda_2 <= lambda_3 <= ... <= lambda_n
    eigen_vals = np.delete(sorted(np.linalg.eigvals(L)), 0)
    reciperocal_sqrt_eigen_vals = np.reciprocal(np.sqrt(eigen_vals))

    # generate n gaussian i.i.d values N(0,1)
    g = np.random.normal(loc=0, scale=1, size=(n-1))
    print(np.multiply(n, np.square(np.linalg.norm(np.multiply(reciperocal_sqrt_eigen_vals, g), ord=np.inf))))

'''
 G = build_grid(5,5)
    print(nx.nodes(G))
    print(nx.edges(G))
    plot(G)
    random_walk(G, 1)'''
def random_walk(G, starting_node):
    nodes_left = list(G.nodes)
    current_location = starting_node
    count = 0
    while len(nodes_left) > 0:
        # print(current_location)
        next_locations = []
        G.number_of_nodes()
        for n in G.neighbors(current_location):
            next_locations.append(n)
        location_length = len(next_locations)
        for n in range(location_length):
            next_locations.append(current_location)
        if current_location in nodes_left:
            nodes_left.remove(current_location)
        current_location = random.choice(next_locations)
        count = count + 1
    print(count)



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
    while count < 1000:
        X_Prev = np.copy(X_Current)
        X_Current = np.matmul(W, X_Prev)
        print(X_Current)
        count = count + 1


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


def main():
    starting_node = 1
    G = build_grid(10, 10)
    approximate_cover(G)
    random_walk(G, starting_node)



def plot(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
       # execute only if run as a script
    main()
