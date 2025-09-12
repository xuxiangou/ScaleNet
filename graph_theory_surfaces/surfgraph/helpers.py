import networkx as nx
from numpy.linalg import norm
from ase.data.colors import jmol_colors
import numpy as np

def grid_iterator(grid):
    """Yield all of the coordinates in a 3D grid as tuples

    Args:
        grid (tuple[int] or int): The grid dimension(s) to
                                  iterate over (x or (x, y, z))

    Yields:
        tuple: (x, y, z) coordinates
    """
    if isinstance(grid, int): # Expand to 3D grid
        grid = (grid, grid, grid)

    for x in range(-grid[0], grid[0]+1):
        for y in range(-grid[1], grid[1]+1):
            for z in range(-grid[2], grid[2]+1):
                yield (x, y, z)

def draw_atomic_graphs(graphs, labels=None, atoms=None):
    """Draws networkx graphs of atomic representations as created in this module.

    The defaults in this function should reproduce the style of graph as presented
    in the manuscript.  This function can serve as a basis for future extension if
    additional properties need to be visualized.

    Args:
        graphs (list[networkx.Graph]): Atomic graphs to show
        labels (list[list[str]] or None): A list of labels for each graph to draw on the nodes
        atoms (list[ase.Atoms] or None): A list of Atoms objects for each atomic graph for
                                         node coloring
    """
    import matplotlib.pyplot as plt

    if labels is None: # No labels are defined
        labels = [None] * len(graphs)

    for index, graph in enumerate(graphs):

        node_colors = [] # Unknown node colors at this point
        if labels[index] is None: # Label undefined, create it as the node's element
            labels[index] = {node:str(node).split("[")[0] for node in graph.nodes()}
        edge_styles = [] # Tracking this to encode bond type
        if atoms is None: # We don't have atoms data provided
            node_colors = 'red' # All nodes red
        else:
            # Using jmol colors based on atomic number, custom coloring could be added here
            node_colors = [jmol_colors[atoms[data['index']].number] for node, data in graph.nodes(data=True)]

        # Edge styling for encoding the bond type
        for e1, e2, data in graph.edges(data=True):
            if data.get("ads_only", True) == 0: # Ads-Ads bond
                edge_styles.append("dashed")
            elif data.get("dist", True) == 0: # Ads-Surface bond
                edge_styles.append("dotted")
            else: # Surface-Surface bond
                edge_styles.append("solid")

        # Basic formatting defaults
        plt.figure(index, figsize=(3.5, 3.5))
        plt.axis('off')

        # Spring layout tends to work best
        # TODO: Implement an atomic posititon layout?  Maybe someday
        pos=nx.spring_layout(graph, k=float(len(graph))**-0.6)
        nx.draw_networkx(graph, pos, node_color=node_colors, edge_color="black", edgecolors="black",
                         labels=labels[index], style=edge_styles, node_size=450, linewidths=2, width=2,
                         font_weight='bold')

    # Show all of the graphs created
    plt.show()


def normalize(vector):
    return vector / norm(vector) if norm(vector) != 0 else vector * 0

def offset_position(atoms, neighbor, offset):
   return atoms[neighbor].position + np.dot(offset, atoms.get_cell())
