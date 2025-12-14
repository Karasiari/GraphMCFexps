from __future__ import annotations
import copy
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigsh

import cvxpy as cp

from .instruments import *

class GraphMCFexps:
    def __init__(self, multigraph: nx.MultiGraph, demands_multidigraph: nx.MultiDiGraph) -> None:
        # инициализация
        self.multigraph = multigraph
        self.demands_multidigraph = demands_multidigraph

        self.graph = self._aggregate_graph(multigraph, "capacity")
        self.demands_graph = self._aggregate_graph(demands_multidigraph, "weight")
        self.demands_laplacian = self._get_laplacian(self.demands_graph)
        
        # поскольку граф будет меняться в экспериментах по расширению - храним исходный вариант
        self.mutltigraph_initial = self.multigraph.copy()
        self.graph_initial = self.graph.copy()

        # последние вычисленные alpha и "усредненная" L_alpha
        self.alpha: Optional[float] = None
        self.L_alpha: Optional[np.ndarray] = None

        # кэши для расчёта alpha / cut
        self.laplacian: Optional[np.ndarray] = self._get_laplacian(self.graph)
        self.graph_pinv_sqrt: Optional[np.ndarray] = None

        # последние посчитанные разрезы self.graph
        self.mincut: Optional[np.ndarray] = None
        self.cut_alpha: Optional[np.ndarray] = None

        # последнее рассчитанное gamma для MCFP
        self.gamma: Optional[float] = None

        # максимальная capacity мультиребра self.multigraph для MCF
        self.C_max = max([data["capacity"] for _, _, data in self.multigraph.edges(data=True)])

        # флаг - решилось ли последнее MCF
        self.mcf_solved: Optional[bool] = None

    # ---------- базовая подготовка ----------
    def _aggregate_graph(self, multigraph, value: str) -> nx.Graph:
        """
        Агрегируем мультиграф в неориентированный граф, где для каждого ребра
        будет сумма value для одного source-target.
        """
        G = nx.Graph()
        G.add_nodes_from(range(multigraph.number_of_nodes()))

        for u, v, data in multigraph.edges(data=True):
            weight = data[value]

            if G.has_edge(u, v):
                G[u][v]['weight'] += weight
            else:
                G.add_edge(u, v, weight=weight)

        return G

    def _get_laplacian(self, graph) -> np.ndarray:
        mat = nx.laplacian_matrix(graph)
        laplacian = mat.astype(float).toarray()
        return laplacian

    def _get_graph_pinv_sqrt(self) -> np.ndarray:
        nodelist = list(self.graph.nodes())
        Lg = nx.laplacian_matrix(self.graph, nodelist=nodelist, weight="weight").astype(float).toarray()
        self.laplacian = Lg
        Lg_pinv = np.linalg.pinv(Lg)
        self.graph_pinv_sqrt = fractional_matrix_power(Lg_pinv, 0.5)
        return self.graph_pinv_sqrt

    # ---------- визуализация немультиграфов ----------
    def visualise(self, version="initial", title="Граф смежности", node_size=300, font_size=10) -> None:
        if version == "current":
          graph = self.graph
        else:
          graph = self.graph_initial
        pos = nx.spring_layout(graph, seed=42)
        plt.figure(figsize=(9, 7))
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#4C79DA", alpha=0.9)
        nx.draw_networkx_edges(graph, pos, edge_color="#888", alpha=0.8)
        edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
        plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()

    def visualise_with_demands(self, version="initial", node_size: int = 110, font_size: int = 9, figsize=(14, 6),
                               demand_edge_width_range=(1.5, 6.0), node_color="dimgray", base_edge_color="gray",
                               demand_edge_cmap="viridis", edge_alpha=0.9, colorbar_label="Вес запроса") -> None:
        """
        Визуализирует граф и граф demand.
        """
        if not isinstance(self.demands_graph, nx.Graph):
            raise AttributeError("self.demands_graph не задан")
        if version == "current":
          graph = self.graph
        else:
          graph = self.graph_initial
        DG = self.demands_graph
        pos = nx.spring_layout(graph, seed=42)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

        # слева — базовый граф
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axL)
        nx.draw_networkx_edges(graph, pos, edge_color=base_edge_color, width=1.5, alpha=edge_alpha, ax=axL)
        labels = {(u, v): f"{d.get('weight', 0):.0f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=font_size, ax=axL)
        axL.set_title("Граф смежности"); axL.axis("off")

        # справа — demands
        nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axR)
        edgelist = list(DG.edges(data=True))
        uv = [(u, v) for u, v, _ in edgelist]
        W = np.array([float(d.get("weight", 1.0)) for _, _, d in edgelist]) if edgelist else np.array([])
        # ширины
        if W.size:
            w_min, w_max = float(W.min()), float(W.max())
            lo, hi = demand_edge_width_range
            widths = [0.5 * (lo + hi)] * len(W) if np.isclose(w_min, w_max) else list(lo + (W - w_min) * (hi - lo) / (w_max - w_min))
        else:
            widths = []
        # цвета
        cmap = mpl.cm.get_cmap(demand_edge_cmap)
        vmin, vmax = (float(W.min()), float(W.max())) if W.size else (0.0, 1.0)
        nx.draw_networkx_edges(DG, pos, edgelist=uv, width=widths, edge_color=W, edge_cmap=cmap,
                               edge_vmin=vmin, edge_vmax=vmax, alpha=edge_alpha, ax=axR)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

        cbar = plt.colorbar(sm, ax=axR); cbar.set_label(colorbar_label, fontsize=font_size)
        axR.set_title("Граф запросов"); axR.axis("off")
        plt.tight_layout(); plt.show()

    # ---------- alpha ----------
    def calculate_alpha(self) -> float:
        if self.demands_graph is None:
            raise AttributeError("demands_graph не задан")
        Ld = self.demands_laplacian
        Lg_inv_sqrt = self._get_graph_pinv_sqrt()
        L_alpha = Lg_inv_sqrt @ Ld @ Lg_inv_sqrt
        self.L_alpha = L_alpha
        eig, _ = eigsh(L_alpha, k=1, which="LA")
        lam_max = float(eig[0]) if eig.size else 0.0
        tr = float(np.trace(L_alpha))
        self.alpha = lam_max / tr if tr != 0.0 else float("inf")
        return lam_max / tr if tr != 0.0 else float("inf")

    # ---------- cut ----------
    def _compute_least_nonzero_vector(self, L: np.ndarray) -> np.ndarray:
        # находим все собственные значения и векторы
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # находим индекс минимального ненулевого собственного значения
        # (первое значение в спектре для связного графа — 0)
        eps = 1e-12  # порог для сравнения с нулём
        nonzero_indices = np.where(eigenvalues > eps)[0]

        idx = nonzero_indices[0]  # минимальное ненулевое собственное значение

        # собственный вектор, соответствующий минимальному ненулевому собственному значению
        least_nonzero_vector = eigenvectors[:, idx]

        return least_nonzero_vector

    def generate_cut(self, type: str = "min") -> list:
        """
        Генерирует разбиение графа на два кластера.
        - type="min": Разбиение по минимальному ненулевому вектору спектра self.laplacian
        - type="min_Lalpha": Разбиение по минимальному ненулевому вектору спектра self.L_alpha

        Аргументы:
            type (str): Тип разбиения, может быть "min" или "min_Lalpha".

        Возвращает:
            list: Список рёбер self.graph (source, target), где вершины принадлежат разным кластерам.
        """
        if type not in {"min", "min_Lalpha"}:
            raise ValueError("type должен быть 'min' или 'min_Lalpha'")

        if self.demands_graph is None:
              raise AttributeError("demands_graph не задан")

        if type == "min":
            # для min-cut используем первый ненулевой собственный вектор лапласиана смежности
            self.laplacian = self._get_laplacian(self.graph)
            v = self._compute_least_nonzero_vector(self.laplacian)
        if type == "min_Lalpha":
            # для min_Lalpha используем первый ненулевой собственный вектор L_alpha
            self.calculate_alpha()
            v = self._compute_least_nonzero_vector(self.L_alpha)

        # используем медиану вектора для разбиения
        med = float(np.median(v))
        # создаём разметку вершин (0 или 1)
        cluster_labels = (v <= med).astype(int)

        # список рёбер между кластерами (где метки разные)
        edges_in_cut = []

        for u, v in self.graph.edges():
          if cluster_labels[u] != cluster_labels[v]:
            edges_in_cut.append((u, v))
            
        if type == "min":
          self.mincut = edges_in_cut
        elif type == "min_Lalpha":
          self.cut_alpha = edges_in_cut

        return edges_in_cut

    # ---------- изменения self.multigraph ----------
    def change_multiedge(self, source: int, target: int, type: str, key: int = None, capacity: float = None) -> None:
        """
        Удаление или добавление мультиребра мультиграфа смежности self.multigraph
        type: "delete" или "insert"
        key: значение ключа удаляемого мультиребра (только для delete)
        capacity: значение capacity нового мультиребра (только для insert)
        """

        if type == "delete":
          if key is None:
            raise ValueError("Для delete необходимо указать key удаляемого мультиребра")

          edge_data = self.multigraph.get_edge_data(source, target, key=key)
          if edge_data:
              capacity_to_decrease = edge_data["capacity"]
              self.multigraph.remove_edge(source, target, key=key)
              current_capacity = self.graph.get_edge_data(source, target)["weight"]
              new_capacity = current_capacity - capacity_to_decrease
              if new_capacity > 0:
                  self.graph[source][target]["weight"] = float(new_capacity)
              else:
                  self.graph.remove_edge(source, target)
          else:
              print(f"Удаляемое мультиребро ({source}, {target}, {key}) не найдено")

        elif type == "insert":
            if capacity is None:
                raise ValueError("Для insert необходимо указать параметр capacity")
            elif capacity <= 0:
                raise ValueError("Параметр capacity должен быть положительным")

            self.multigraph.add_edge(source, target, capacity=capacity)

            if self.graph.get_edge_data(source, target):
                current_capacity = self.graph.get_edge_data(source, target)["weight"]
                new_capacity = current_capacity + capacity
                self.graph[source][target]["weight"] = float(new_capacity)
            else:
                self.graph.add_edge(source, target, weight=float(capacity))
        
        else:
          raise ValueError('type должен быть "delete" или "insert"')

    def restore_graph(self) -> None:
        """
        Восстановление self.multigraph из self.multigraph_initial
        """

        self.multigraph = self.multigraph_initial.copy()
        self.graph = self.graph_initial.copy()

    # ---------- решения основных задач на графе ----------
    
    # MCFP (gamma)
    def solve_mcfp(self, **solver_kwargs) -> float:
        """
        Решение задачи максимального пропускного потока на графе self.graph + self.demands_graph с использованием CVXPY.
        solver_kwargs: параметры для solver.solve(), такие как методы решения и точность.
        return: gamma
        """
        # копируем граф и преобразуем его в ориентированный
        graph = self.graph.copy()
        graph = nx.DiGraph(graph)

        # копируем лапласиан запросов
        demands_laplacian = self.demands_laplacian.copy()

        # получаем incidence matrix и capacities рёбер
        incidence_mat = get_incidence_matrix_for_mcfp(graph)
        bandwidth = get_capacities_for_mcfp(graph)

        # определяем переменные потока и гамму
        flow = cp.Variable((len(graph.edges), len(graph.nodes)))
        gamma = cp.Variable()

        # определяем задачу
        prob = cp.Problem(
            cp.Maximize(gamma),
            [
                cp.sum(flow, axis=1) <= bandwidth,
                incidence_mat @ flow == -gamma * demands_laplacian.T,
                flow >= 0,
                gamma >= 0,
            ]
        )

        # решаем задачу
        prob.solve(**solver_kwargs)

        if prob.status != "optimal":
            gamma = None
            
        gamma = gamma.value if gamma is not None else None
        self.gamma = gamma
        
        return gamma

    # MCF (проложенные запросы, индексы проложенных запросов, флаг - проложились ли все запросы)
    def solve_mcf(self, eps=0.1):
        # Step 0: Get right representation for demands
        demands = []
        index, unsatisfied_subset = 0, set()
        for source, sink, key, data in self.demands_multidigraph.edges(keys=True, data=True):
            capacity = data.get("weight", 0.0)
            demands.append(Demand(source, sink, capacity))
            unsatisfied_subset.add(index)
            index += 1
        
        # Step 1: Group demands and create the mapping from i to source-target pairs
        grouped_demands, demand_indices_by_group, i_to_source_target = group_demands_and_create_mapping(demands,
                                                                                                        unsatisfied_subset)
        G = nx.MultiDiGraph(self.multigraph)
        G_copy = G.copy()
        
        # Step 2: Run the multicommodity flow procedure to generate the flow and l(e) values
        flow = multi_commodity_flow(G_copy, grouped_demands, self.C_max, eps)

        # Step 3: Scale the flow to make it feasible (ensures flows respect edge capacities)
        scale_flows(flow, G_copy, self.C_max)

        # Step 4: Subdivide flows by paths for ungrouped demands
        flow_paths, satisfied_demands = subdivide_flows_by_paths(flow, demand_indices_by_group, demands,
                                                                 i_to_source_target)

        # Step 5: Subtract the satisfied demands from the graph capacity
        graph_copy = subtract_flow_from_capacity(G_copy, flow_paths, demands)

        satisfied_demands_set = set(satisfied_demands)
        left_to_satisfy = unsatisfied_subset - satisfied_demands_set
        
        # Step 6: Try to fulfill remaining demands in the leftover graph
        remaining_paths, remaining_satisfied_demands = fulfill_remaining_demands(graph_copy, demands,
                                                                                 demand_indices_by_group,
                                                                                 i_to_source_target, left_to_satisfy)

        # Combine the satisfied demands
        satisfied_demands += remaining_satisfied_demands
        flow_paths.update(remaining_paths)
        solved = unsatisfied_subset == set(satisfied_demands)
        self.mcf_solved = solved

        return flow_paths, satisfied_demands, solved
