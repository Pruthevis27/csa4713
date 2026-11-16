
import networkx as nx
import matplotlib.pyplot as plt

# Create Bayesian Network
G = nx.DiGraph()

# Add nodes
G.add_edges_from([
    ("Rain", "Wet Grass"),
    ("Sprinkler", "Wet Grass"),
    ("Rain", "Cloudy")
])

# Draw network
nx.draw(G, with_labels=True,
        node_color='lightblue',
        node_size=2500,
        font_size=10,
        arrows=True)

plt.title("Bayesian Network")
plt.show()
