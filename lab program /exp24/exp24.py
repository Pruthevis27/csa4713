
import networkx as nx
import matplotlib.pyplot as plt

# Create graphical model
G = nx.DiGraph()

G.add_edges_from([
    ("Rain", "Wet Grass"),
    ("Sprinkler", "Wet Grass")
])

# Generate random samples
for i in range(5):
    rain = "Yes" if i % 2 == 0 else "No"
    sprinkler = "Yes" if i % 3 == 0 else "No"

    wet = "Yes" if rain == "Yes" or sprinkler == "Yes" else "No"

    print(rain, sprinkler, wet)

# Draw graph
nx.draw(G, with_labels=True,
        node_color='lightblue',
        node_size=2500,
        arrows=True)

plt.title("Graphical Model")
plt.show()
