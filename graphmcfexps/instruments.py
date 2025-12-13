from collections import defaultdict

import numpy as np
import networkx as nx

# глобальное capacity мультиребра для нашей задачи MCF всегда 1, вес любого demand тоже 1
C_MAX = 1.0

# вспомогательный класс для demands для входа в наш алгоритм решения MCF
class Demand:
    def __init__(self, source: int, sink: int, capacity: float):
        self.source = source
        self.sink = sink
        self.capacity = capacity

# вспомогательные функции для MCFP (Maximum Concurrent Flow Problem)

def get_incidence_matrix_for_mcfp(graph: nx.DiGraph) -> np.ndarray:
    """
    Создаем incidence matrix
    параметры: nx.DiGraph
    return: incidence_matrix
    """
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True)
    incidence_matrix = incidence_matrix.toarray()
    return incidence_matrix

def get_capacities_for_mcfp(graph: nx.DiGraph) -> np.ndarray:
    """
    Достаем capacities
    параметры: nx.DiGraph, граф с capacities (key: weights) на ребрах
    return: np.ndarray
    """
    edges_with_weights = [(edge, data['weight']) for edge, data in graph.edges.items()]
    edges_with_weights_dict = {key: value for key, value in edges_with_weights}
    return np.array(list(edges_with_weights_dict.values()), dtype=np.float64)

# вспомогательные функции для целочисленного MCF (Multi Commodity Flow)
# комментарии на английском - оставил оригинал

# Custom function to find the shortest path with edge keys
def shortest_path_with_edge_keys(G, source, target, edge_costs):
    paths = nx.shortest_path(G, source, target, weight=lambda u, v, key: edge_costs[(u, v, key)])
    edges_with_keys = []
    for u, v in zip(paths[:-1], paths[1:]):
        # Get the edge key with the minimum cost for the (u, v) pair
        min_key = min(G[u][v], key=lambda key: edge_costs[(u, v, key)])
        edges_with_keys.append((u, v, min_key))
    return edges_with_keys


# Function to copy the graph and filter edges based on the demand capacity
def copy_and_filter_graph(flow_graph, demand_capacity):
    filtered_graph = nx.MultiDiGraph()
    for u, v, key, data in flow_graph.edges(data=True, keys=True):
        if data['capacity'] >= demand_capacity:
            filtered_graph.add_edge(u, v, key=key, **data)
    return filtered_graph


# Function to group demands by their source and sink, save demand indices, and create a mapping from i to source-target pair
def group_demands_and_create_mapping(demands, unsatisfied_demands: set):
    grouped_demands = []
    demand_indices_by_group = defaultdict(list)
    i_to_source_target = {}

    # Group demands by source-target pairs and store indices
    demand_dict = defaultdict(lambda: {"capacity": 0, "indices": []})

    for index, demand in enumerate(demands):
        if index not in unsatisfied_demands:
            continue
        key = (demand.source, demand.sink)

        demand_dict[key]["capacity"] += demand.capacity
        demand_dict[key]["indices"].append(index)

    # Create grouped demands and mappings
    for i, ((source, sink), info) in enumerate(demand_dict.items()):
        grouped_demands.append(Demand(source, sink, info["capacity"]))
        demand_indices_by_group[i] = info["indices"]
        i_to_source_target[i] = (source, sink)

    return grouped_demands, demand_indices_by_group, i_to_source_target


# Function to calculate D(l) = C_max * sum(l(e))
def D(graph, C_max):
    return C_max * sum(nx.get_edge_attributes(graph, 'l').values())


# Function to initialize l(e) = gamma / C_max

# Initialize l(e) as a graph attribute
def initialize_l(G, C_max, eps):
    m = len(G.edges(keys=True))
    gamma = (m / (1 - eps)) ** (-1 / eps)

    #gamma = 0.01
    l_values = {e: gamma / C_max for e in G.edges(keys=True)}
    nx.set_edge_attributes(G, l_values, "l")


# Function to find the shortest path with edge costs l(e)
# Find shortest path and return edges with their keys
def shortest_path_with_l(G, source, sink):
    # Get the shortest path as a list of nodes
    node_path = nx.shortest_path(G, source, sink, weight='l')

    edges_with_keys = []

    # Iterate through the node pairs in the path
    for u, v in zip(node_path[:-1], node_path[1:]):
        # Get the edge key with the minimum l(e) for the (u, v) pair
        min_key = min(G[u][v], key=lambda key: G[u][v][key]['l'])
        edges_with_keys.append((u, v, min_key))  # Append (u, v, key)

    return edges_with_keys


# Update l(e) -> l(e) * (1 + eps) for all edges in the path
def update_l_on_path(G, path, eps):
    for u, v, key in path:
        G[u][v][key]['l'] *= (1 + eps)


# Main flow procedure
def multi_commodity_flow(G, grouped_demands, eps=0.1):
    # Initialize l(e)
    initialize_l(G, C_MAX, eps)

    # Initialize flow structures (separate for each commodity flow)
    flow = {i: defaultdict(float) for i in range(len(grouped_demands))}
    iter_max = G.number_of_edges()
    iter_num = 0
    while D(G, C_MAX) < 1 and iter_num < iter_max:
        iter_num += 1
        for i, demand in enumerate(grouped_demands):
            source, sink = demand.source, demand.sink
            d_i = demand.capacity

            # While D(l) < 1 and there is remaining demand to route
            while D(G, C_MAX) < 1 and d_i > 0:
                # Find the shortest path based on current l(e)
                path = shortest_path_with_l(G, source, sink)

                # Set u_flow = min(C_max, d_i)
                u_flow = min(C_MAX, d_i)

                # Augment the flow along the path and reduce remaining demand
                for u, v, key in path:
                    flow[i][(u, v, key)] += u_flow

                d_i -= u_flow  # Reduce remaining demand

                # Update l(e) -> l(e) * (1 + eps) for all edges in the path
                update_l_on_path(G, path, eps)

    return flow


# Function to scale the flow to make it feasible
def scale_flows(flow, G):
    # Find the maximum sum of flows on any edge
    max_over_capacity = 1
    for u, v, key in G.edges(keys=True):
        G[u][v][key]['capacity'] = C_MAX
        total_flow = sum(f.get((u, v, key), 0) for f in flow.values())
        max_over_capacity = max(max_over_capacity, total_flow / C_MAX)

    # If the maximum over-capacity ratio is more than 1, scale the flow
    if max_over_capacity > 1:
        for f in flow.values():
            for edge in f:
                f[edge] /= max_over_capacity


def subtract_flow_from_graph(flow_graph, path, demand_capacity):
    path_with_keys = []
    for u, v in zip(path[:-1], path[1:]):
        # Find the edge in the original flow graph with sufficient capacity
        for key in flow_graph[u][v]:
            if flow_graph[u][v][key]['capacity'] >= demand_capacity:
                flow_graph[u][v][key]['capacity'] -= demand_capacity
                path_with_keys.append((u, v, key))  # Save the (u, v, key) format
                if flow_graph[u][v][key]['capacity'] <= 0:
                    flow_graph.remove_edge(u, v, key=key)
                break  # We found a valid key, so we can stop here
    return path_with_keys


# Function to subdivide flows by paths for each source-target pair
def subdivide_flows_by_paths(flow, demand_indices_by_group, ungrouped_demands, i_to_source_target):
    satisfied_demands = []  # To store indices of satisfied demands
    flow_paths = {}  # To store paths by demand index

    # Create flow graphs for each source-target pair
    flow_graphs = {}
    for i, f in flow.items():
        flow_graph = nx.MultiDiGraph()
        for (u, v, key), flow_value in f.items():
            if flow_value > 0:  # Only add edges where flow > 0
                flow_graph.add_edge(u, v, key=key, capacity=flow_value)
        flow_graphs[i] = flow_graph

    # Process each grouped demand by its source-target pair
    for i, demand_indices in demand_indices_by_group.items():
        source, sink = i_to_source_target[i]

        # Find the corresponding flow graph for this source-target pair
        flow_graph = flow_graphs[i]

        # Sort demands by capacity in descending order
        #sorted_demand_indices = sorted(demand_indices, key=lambda idx: ungrouped_demands[idx].capacity, reverse=True)
        sorted_demand_indices = demand_indices

        # Process each demand in descending order of capacity
        for demand_index in sorted_demand_indices:
            demand = ungrouped_demands[demand_index]

            # Copy and filter the graph for edges that can handle the demand capacity
            filtered_graph = copy_and_filter_graph(flow_graph, demand.capacity)
            if not source in filtered_graph:
                continue
            if not sink in filtered_graph:
                continue
            try:
                # Find the shortest path in the filtered graph
                path = nx.shortest_path(filtered_graph, source, sink)

                # Subtract the demand capacity from the original flow graph and get the (u, v, key) path
                path_with_keys = subtract_flow_from_graph(flow_graph, path, demand.capacity)

                # Save the satisfied demand and its path
                satisfied_demands.append(demand_index)
                flow_paths[demand_index] = path_with_keys  # Store path with (u, v, key) format

            except nx.NetworkXNoPath:
                # If no path found, we skip this demand
                continue

    return flow_paths, satisfied_demands


# Function to subtract flow from capacities in a graph copy
def subtract_flow_from_capacity(G, flow_paths, ungrouped_demands):
    graph_copy = G.copy()

    # Subtract flow paths from graph copy
    for demand_index, path_with_keys in flow_paths.items():
        demand = ungrouped_demands[demand_index]
        for u, v, key in path_with_keys:
            graph_copy[u][v][key]['capacity'] -= demand.capacity
            if graph_copy[u][v][key]['capacity'] <= 0:
                graph_copy.remove_edge(u, v, key=key)

    return graph_copy


# Function to fulfill remaining demands in the leftover graph
def fulfill_remaining_demands(graph_copy, ungrouped_demands, demand_indices_by_group, i_to_source_target,
                              left_to_satisfy: set):
    remaining_paths = {}
    satisfied_demands = []

    # Process each grouped demand by its source-target pair
    for i, demand_indices in demand_indices_by_group.items():
        source, sink = i_to_source_target[i]

        # Sort demands by capacity in descending order
        #sorted_demand_indices = sorted(demand_indices, key=lambda idx: ungrouped_demands[idx].capacity, reverse=True)
        sorted_demand_indices = demand_indices

        # Process each demand in descending order of capacity
        for demand_index in sorted_demand_indices:
            demand = ungrouped_demands[demand_index]
            if demand_index not in left_to_satisfy:
                continue
            # Copy and filter the graph for edges that can handle the demand capacity
            filtered_graph = copy_and_filter_graph(graph_copy, demand.capacity)
            if not source in filtered_graph:
                continue
            if not sink in filtered_graph:
                continue
            try:
                # Find the shortest path in the filtered graph
                path = nx.shortest_path(filtered_graph, source, sink)

                # Subtract the demand capacity from the graph copy and get the (u, v, key) path
                path_with_keys = subtract_flow_from_graph(graph_copy, path, demand.capacity)

                # Save the satisfied demand and its path
                satisfied_demands.append(demand_index)
                remaining_paths[demand_index] = path_with_keys  # Store path with (u, v, key) format

            except nx.NetworkXNoPath:
                # If no path found, we skip this demand
                continue

    return remaining_paths, satisfied_demands
