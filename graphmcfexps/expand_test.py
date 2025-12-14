import random
import networkx as nx

from .core import *

# вспомогательные функции для теста по расширению сети

def get_available_types() -> list[str]:
  return ["initial", "alpha", "random", "min_cut", "min_Lalpha_cut", "betweenness_unweighted"]

def get_edges_by_alpha(graph: GraphMCFexps, k: int, pref: str):
  pref_edges = graph.generate_cut(type=pref)
  if len(pref_edges) < k:
    raise ValueError(f'Количество новых ресурсов слишком велико для теста "alpha": {k} расширений vs {len(pref_edges)} ребер разреза')

  edges_with_alphas = []
  for source, target in pref_edges:
    source_target_multiedges = [(source, target, key) for key in graph.multigraph.get_edge_data(source, target).keys()]
    for source, target, key in source_target_multiedges:
      graph.change_multiedge(source, target, "delete", key)
    alpha = graph.calculate_alpha()
    edges_with_alphas.append((source, target, alpha))
    graph.restore_graph()

  edges_with_alphas.sort(key=lambda x: x[2], reverse=True)
  return edges_with_alphas[:k]

def sequences_to_edges(edge_sequence: list[(int, int)], capacity_sequence: list[float], graph: GraphMCFexps) -> None:
  for (source, target), capacity in zip(edge_sequence, capacity_sequence):
    graph.change_multiedge(source, target, type="insert", capacity=capacity)

# основная функция для теста по расширению сети

def expand_network_test(additional_capacities: list[float], graph: GraphMCFexps, type: str, alpha_type: str = None) -> float:
  additional_capacities.sort(reverse=True)
  number_to_add = len(additional_capacities)

  if type == "alpha":
    if alpha_type is None:
      raise ValueError(f'Для type="alpha" необходимо передать значение alpha_type для предварительной выборки ребер из разреза')

    edges_with_alphas = get_edges_by_alpha(graph, number_to_add, pref=alpha_type)

    source_target_sequence = [(source, target) for source, target, alpha in edges_with_alphas]
    sequences_to_edges(source_target_sequence, additional_capacities, graph)
    gamma_for_test = graph.solve_mcfp()
    graph.restore_graph()
  
  elif type == "min_cut":
    source_target_sequence = graph.generate_cut(type="min")
    if len(source_target_sequence) < number_to_add:
      raise ValueError(f'Количество новых ресурсов слишком велико для теста "min_cut": {number_to_add} расширений vs {len(source_target_sequence)} ребер разреза')

    random.shuffle(source_target_sequence)
    sequences_to_edges(source_target_sequence[:number_to_add], additional_capacities, graph)
    gamma_for_test = graph.solve_mcfp()
    graph.restore_graph()
  
  elif type == "min_Lalpha_cut":
    source_target_sequence = graph.generate_cut(type="min_Lalpha")
    if len(source_target_sequence) < number_to_add:
      raise ValueError(f'Количество новых ресурсов слишком велико для теста "min_Lalpha_cut": {number_to_add} расширений vs {len(source_target_sequence)} ребер разреза')
    
    random.shuffle(source_target_sequence)
    sequences_to_edges(source_target_sequence[:number_to_add], additional_capacities, graph)
    gamma_for_test = graph.solve_mcfp()
    graph.restore_graph()
  
  elif type == "random":
    source_target_sequence = [(source, target) for source, target in graph.graph.edges()]
    if len(source_target_sequence) < number_to_add:
      raise ValueError(f'Количество новых ресурсов слишком велико для теста "random": {number_to_add} расширений vs {len(source_target_sequence)} ребер всего')

    random.shuffle(source_target_sequence)
    sequences_to_edges(source_target_sequence[:number_to_add], additional_capacities, graph)
    gamma_for_test = graph.solve_mcfp()
    graph.restore_graph()
  
  elif type == "betweenness_unweighted":
    edges_with_betweenness_unweighted = [(source, target, value) for (source, target), value in nx.edge_betweenness_centrality(graph.graph, weight=None).items()]
    edges_with_betweenness_unweighted.sort(key=lambda x: x[2], reverse=True)
    source_target_sequence = [(source, target) for source, target, value in edges_with_betweenness_unweighted]
    if len(source_target_sequence) < number_to_add:
      raise ValueError(f'Количество новых ресурсов слишком велико для теста "betweenness_unweighted": {number_to_add} расширений vs {len(source_target_sequence)} ребер всего')

    sequences_to_edges(source_target_sequence[:number_to_add], additional_capacities, graph)
    gamma_for_test = graph.solve_mcfp()
    graph.restore_graph()

  elif type == "initial":
    gamma_for_test = graph.solve_mcfp()
  
  else:
    raise ValueError(f'Нет подходящего вида type')

  return gamma_for_test
