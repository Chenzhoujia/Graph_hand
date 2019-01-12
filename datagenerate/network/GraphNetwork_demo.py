import graph_nets as gn
import sonnet as snt
# Provide your own functions to generate graph-structured data.
input_graphs = get_graphs()
# Create the graph network.
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))
# Pass the input graphs to the graph network, and return the output graphs.
output_graphs = graph_net_module(input_graphs)