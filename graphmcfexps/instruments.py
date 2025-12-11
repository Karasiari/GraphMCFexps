import numpy as np
import networkx as nx

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
