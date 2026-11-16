
import networkx as nx
import matplotlib.pyplot as plt

# Create RNN graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(["X1", "X2", "X3", "H1", "H2", "H3", "Y"])

# Add connections
G.add_edges_from([
    ("X1", "H1"),
    ("H1", "H2"),
    ("H2", "H3"),
    ("X2", "H2"),
    ("X3", "H3"),
    ("H3", "Y")
])

# Draw graph
nx.draw(G, with_labels=True,
        node_color='lightblue',
        node_size=2000,
        font_size=12,
        arrows=True)

plt.title("Unfolded RNN Architecture")
plt.show()
