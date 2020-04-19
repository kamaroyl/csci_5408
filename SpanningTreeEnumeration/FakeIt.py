import matplotlib.pyplot as plt
import networkx as nx
COUNT = 0


# Recursive partitioning function; based off
# https://www.geeksforgeeks.org/print-subsets-given-size-set/
def PrintTree(arr, n, r, index, data, i):

    #if we have a full subset, "print"
    if index == r:
        G = nx.MultiGraph()
        G.add_nodes_from([1, 2, 3, 4, 5])
        G.add_edges_from(data)
        # Only actually plot if the graph is a tree
        if nx.is_tree(G):
            plot(G)
        return

    # If we're out of the bounds of the main array, return
    if i >= n:
        return

    # Fill up the sub array!
    data[index] = arr[i]
    PrintTree(arr, n, r, index + 1, data, i + 1)
    PrintTree(arr, n, r, index, data, i + 1)


def main():
    # List of possible edges - the edge missing is 3-4 (c-d)
    edgeList = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5)]
    subList = [(0,0), (0,0), (0,0), (0,0)]
    PrintTree(edgeList, 9, 4, 0, subList, 0)

# Cheers to atomh33ls' answer
# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
# Cheers to Joel's answer
# https://stackoverflow.com/questions/30035039/fix-position-of-subset-of-nodes-in-networkx-spring-graph
def plot(G):
    fixed_positions = {1: (0.11330065, 0.99562374), 2: (0.68947051, -0.7463859), 3: (0.90566347, 0.41867599),
                       4: (-0.48183333, -0.86529587), 5: (-1.00, 0.19738203)}
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)
    labels = labels = {}
    labels[1]=r'$a$'
    labels[2]=r'$b$'
    labels[3]=r'$c$'
    labels[4]=r'$d$'
    labels[5]=r'$e$'
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=100, alpha=1)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
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

    global COUNT
    plt.savefig("Graph" + str(COUNT) + ".png", format="PNG")
    plt.show(block=False)

    COUNT = COUNT + 1


if __name__ == "__main__":
    # execute only if run as a script
    main()