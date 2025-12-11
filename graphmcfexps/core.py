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
    def __init__(self, adjacency_matrix: np.ndarray, demands_matrix: np.ndarray) -> None:
        # инициализация
        self.adjacency_matrix = np.array(adjacency_matrix, dtype=float)
        self.demands_matrix = np.array(demands_matrix, dtype=float)
        self._validate_matrices()
        self.graph = self._create_networkx_graph(self.adjacency_matrix)
        self.n = self.graph.number_of_nodes()
        self.demands_graph = self._create_networkx_graph(self.demands_matrix)
        self.demands_laplacian = self._get_laplacian(self.demands_graph)
        # поскольку граф будет меняться в экспериментах по расширению - храним исходный вариант
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

    # ---------- базовая подготовка ----------
    def _validate_matrices(self) -> None:
        A = self.adjacency_matrix
        D = self.demands_matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Матрица смежности должна быть квадратной")
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("Матрица корреспонденций должна быть квадратной")
        if not np.allclose(A, A.T):
            raise ValueError("Матрица смежности должна быть симметричной (неориентированный граф)")
        if (A < 0).any():
            raise ValueError("Capacity рёбер должно быть неотрицательным")
        if A.shape[0] != D.shape[0]:
          raise ValueError("Матрица смежности и матрица корреспонденций разных размеров")

    def _create_networkx_graph(self, matrix) -> nx.Graph:
        A = matrix
        n = A.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                w = A[i, j]
                if w:
                    G.add_edge(i, j, weight=float(w))
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

    # ---------- визуализация ----------
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

    # ---------- изменения self.graph ----------
    def remove_edge(self, source: int, target: int, type: str, weight: float = None) -> None:
        """
        Удаление или уменьшение веса ребра графа смежности self.graph
        type: "delete" или "reduce"
        weight: насколько уменьшить вес (только для reduce)
        """

        # текущий вес ребра в матрице смежности
        current_weight = self.adjacency_matrix[source, target]

        # ребра нет 
        if current_weight == 0:
          return

        # удаление ребра
        if type == "delete":
          self.adjacency_matrix[source, target] = 0.0
          self.adjacency_matrix[target, source] = 0.0
          
          self.graph.remove_edge(source, target)

        # уменьшение веса
        elif type == "reduce":
          if weight is None:
            raise ValueError("Для reduce необходимо указать параметр weight")

          new_weight = current_weight - weight

          if new_weight <= 0:
            # по факту delete
            self.adjacency_matrix[source, target] = 0.0
            self.adjacency_matrix[target, source] = 0.0

            self.graph.remove_edge(source, target)
          else:
            self.adjacency_matrix[source, target] = new_weight
            self.adjacency_matrix[target, source] = new_weight

            self.graph[source][target]["weight"] = float(new_weight)

        else:
          raise ValueError('type должен быть "delete" или "reduce"')

    def restore_graph(self) -> None:
        """
        Восстановление self.graph из self.graph_initial
        """

        self.graph = self.graph_initial.copy()

        # восстанавливаем self.adjacency_matrix
        n = self.n
        A = np.zeros((n, n), dtype=float)

        for u, v, data in self.graph.edges(data=True):
          w = float(data.get("weight", 1.0))
          A[u, v] = w
          A[v, u] = w

        self.adjacency_matrix = A

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
