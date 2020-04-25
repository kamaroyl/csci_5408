"""
Draw a graph with matplotlib, color by degree.

You must have matplotlib for this to work.
"""
__author__ = """Aric Hagberg (hagberg@lanl.gov)"""
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np

from utilities import generate_random_graph, plot

def cube_plot():
    G=nx.cubical_graph()
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,
                           nodelist=[0,1,2,3],
                           node_color='r',
                           node_size=500,
                           alpha=0.8)
    nx.draw_networkx_nodes(G,pos,
                           nodelist=[4,5,6,7],
                           node_color='b',
                           node_size=500,
                           alpha=0.8)

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_edges(G,pos,
                           edgelist=[(0,1),(1,2),(2,3),(3,0)],
                           width=8,alpha=0.5,edge_color='r')
    nx.draw_networkx_edges(G,pos,
                           edgelist=[(4,5),(5,6),(6,7),(7,4)],
                           width=8,alpha=0.5,edge_color='b')


    # some math labels
    labels={}
    labels[0]=r'$a$'
    labels[1]=r'$b$'
    labels[2]=r'$c$'
    labels[3]=r'$d$'
    labels[4]=r'$\alpha$'
    labels[5]=r'$\beta$'
    labels[6]=r'$\gamma$'
    labels[7]=r'$\delta$'
    nx.draw_networkx_labels(G,pos,labels,font_size=16)

    plt.axis('off')
    plt.savefig("labels_and_colors.png") # save as png
    plt.show() # display


def plot_partition(G, partition):
    pos = nx.spring_layout(G)
    print(partition[0])
    print(partition[1])
    A = G.subgraph(partition[0])
    B = G.subgraph(partition[1])
    labels = labels = {}
    labels[0]=r'$0$'
    labels[1]=r'$1$'
    labels[2]=r'$2$'
    labels[3]=r'$3$'
    labels[4]=r'$4$'
    labels[5]=r'$5$'
    labels[6]=r'$6$'
    labels[7]=r'$7$'
    labels[8]=r'$8$'
    labels[9]=r'$9$'
    nx.draw_networkx_nodes(G, pos, nodelist=partition[0], node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_nodes(G, pos, nodelist=partition[1], node_color='b', node_size=100, alpha=1)
    nx.draw_networkx_labels(G, pos, labels, font_size=20)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


def adj_mat():
    A = nx.from_numpy_matrix(np.array([
                                        [0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                        [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]]))
    fiedler_vector = nx.fiedler_vector(A)
    LG = nx.laplacian_matrix(A)
    print(np.linalg.eigvals(LG.toarray()))
    print(fiedler_vector)
    partition = ([], [])
    for i in range(len(fiedler_vector)):
        if fiedler_vector[i] > 0:
            partition[1].append(i)
        else:
            partition[0].append(i)
    plot_partition(A, partition)


def connected_components():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10])
    G.add_edges_from([(1,10)])
    LG = nx.laplacian_matrix(G)
    eigen_values = np.linalg.eigvals(LG.toarray())
    print(eigen_values)


def wait_test():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4])
    G.add_edges_from([(1,2), (2,3), (3,4), (4,1)])
    plot(G)
    keyboardClick = False
    while keyboardClick != True:
        print("Waiting for input")
        keyboardClick = plt.waitforbuttonpress()
    plt.clf()
    G.add_edge(1,3)
    plot(G)

    print("Done")


def test_gaussian_norm():
    random_sample = np.random.normal(loc=0, scale=.9, size=(100))
    print(random_sample)


def main():
    test_gaussian_norm()


if __name__ == "__main__":
       # execute only if run as a script
    main()
