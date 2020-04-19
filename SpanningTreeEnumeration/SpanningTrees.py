import matplotlib.pyplot as plt
import networkx as nx


def main():
    G = nx.MultiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5)])
    plot(G)


def plot(G):
    fixed_positions = {1: (0.11330065,  0.99562374), 2: (0.68947051, -0.7463859), 3: (0.90566347, 0.41867599), 4: (-0.48183333, -0.86529587), 5: (-1.00,  0.19738203)}
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
                                                                           ),
                                    ),
                    )
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
