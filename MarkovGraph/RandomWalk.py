import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from utilities.GenerateGraph import build_grid


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
