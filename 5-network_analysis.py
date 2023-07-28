import networkx as nx
import pandas as pd
import community
from community import community_louvain

features_df = pd.read_csv('extracted_features.csv', index_col='Unnamed: 0')

# Create an empty undirected graph
G = nx.Graph()

# Add nodes to the graph with node attributes from the features_df DataFrame
for index, row in features_df.iterrows():
    node_id = index
    node_attributes = row.drop('label').to_dict()
    G.add_node(node_id, **node_attributes)

# Graph Analysis
# Calculate node degree (number of edges connected to each node)
node_degrees = dict(G.degree())

# Detect communities using Louvain method
partition = community_louvain.best_partition(G)

# Assign community IDs as node attributes
nx.set_node_attributes(G, partition, name='community')

# Visualization
# Draw the graph with node colors representing communities
pos = nx.spring_layout(G)  # Layout algorithm to position nodes
node_colors = [partition[node] for node in G.nodes()]
plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_color=node_colors, cmap='viridis', with_labels=False, node_size=50)
plt.title('Graph Network with Communities (Louvain)')
plt.show()