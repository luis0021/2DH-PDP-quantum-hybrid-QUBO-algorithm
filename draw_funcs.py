# pip install networkx
import re
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ----------------- Utilidades -----------------

def _all_vars(linear, quadratic):
    """Conjunto de variables presentes en lineales y cuadráticos."""
    vars_lin = set(linear.keys())
    vars_quad = set(itertools.chain.from_iterable(quadratic.keys()))
    return sorted(vars_lin | vars_quad, key=lambda s: tuple(int(x) for x in re.findall(r'\d+', s)))

def _index_map(vars_list):
    return {v: i for i, v in enumerate(vars_list)}

def build_adj_matrix(quadratic, vars_list=None):
    """Matriz de adyacencia (solo términos bilineales)."""
    if vars_list is None:
        vars_list = _all_vars({}, quadratic)
    idx = _index_map(vars_list)
    n = len(vars_list)
    A = np.zeros((n, n), dtype=float)
    for (u, v), w in quadratic.items():
        i, j = idx[u], idx[v]
        if i == j:
            # si llega algo diagonal por error, lo ponemos ahí
            A[i, j] += w
        else:
            A[i, j] += w
            A[j, i] += w
    return A, vars_list

def build_adj_matrix_normalized(quadratic, vars_list=None):
    """Matriz de adyacencia (solo términos bilineales)."""
    if vars_list is None:
        vars_list = _all_vars({}, quadratic)
    idx = _index_map(vars_list)
    n = len(vars_list)
    A = np.zeros((n, n), dtype=float)
    for (u, v), w in quadratic.items():
        i, j = idx[u], idx[v]
        if i == j:
            # si llega algo diagonal por error, lo ponemos ahí
            A[i, j] += w
        else:
            A[i, j] += w
            A[j, i] += w
    max_val = np.max(np.abs(A))
    if max_val > 0:
        A_matrix_normalized = A / max_val
    else:
        A_matrix_normalized = A
    return A_matrix_normalized, vars_list


# ----------------- Gráficos -----------------

## 1) Indica qué variables tienen más influencia positiva/negativa
def plot_linear_bar(linear, top_k=None, sort_by_abs=True, title="Lineales (QUBO)"):
    """Bar chart de los coeficientes lineales."""
    items = list(linear.items())
    if sort_by_abs:
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    else:
        items.sort(key=lambda kv: kv[0])
    if top_k is not None:
        items = items[:top_k]

    labels = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(max(6, len(items)*0.35), 4))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Peso")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_quadratic_bar(quadratic, top_k=None, sort_by_abs=True, title="Cuadraticos (QUBO)"):
    """Bar chart de los coeficientes cuadraticos."""
    items = list(quadratic.items())
    if sort_by_abs:
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    else:
        items.sort(key=lambda kv: kv[0])
    if top_k is not None:
        items = items[:top_k]

    labels = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(max(6, len(items)*0.35), 4))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Peso")
    plt.title(title)
    plt.tight_layout()
    plt.show()

## 2) Muestra bloques de variables fuertemente acopladas; abs_weights=True para ver magnitudes; threshold para ocultar acoplamientos débiles
def plot_quadratic_heatmap(quadratic, vars_list=None, reorder_by_strength=True,
                           abs_weights=False, threshold=None, title="Cuadráticos (heatmap)"):
    """
    Heatmap de los términos cuadráticos (bilineales). Opciones:
      - abs_weights=True para ver magnitudes |w|.
      - threshold: ignora |w| < threshold (los pone a 0).
      - reorder_by_strength: reordena según grado ponderado para resaltar estructura.
    """
    A, vars_list = build_adj_matrix(quadratic, vars_list)
    W = np.abs(A) if abs_weights else A.copy()
    if threshold is not None:
        W[np.abs(W) < threshold] = 0.0

    # reordenar por 'fuerza' (grados ponderados) para resaltar bloques
    order = np.arange(len(vars_list))
    if reorder_by_strength:
        strength = W.sum(axis=0)
        order = np.argsort(-strength)  # descendente
        W = W[order][:, order]
        vars_list = [vars_list[i] for i in order]

    plt.figure(figsize=(max(6, len(vars_list)*0.35), max(5, len(vars_list)*0.35)))
    im = plt.imshow(W, aspect='auto', cmap='Greys_r')
    plt.xticks(np.arange(len(vars_list)), vars_list, rotation=90)
    plt.yticks(np.arange(len(vars_list)), vars_list)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title + (" (|w|)" if abs_weights else ""))
    plt.tight_layout()
    plt.show()

def plot_quadratic_heatmap_normalized(quadratic, vars_list=None, reorder_by_strength=True,
                           abs_weights=False, threshold=None, title="Cuadráticos (heatmap)"):
    """
    Heatmap de los términos cuadráticos (bilineales). Opciones:
      - abs_weights=True para ver magnitudes |w|.
      - threshold: ignora |w| < threshold (los pone a 0).
      - reorder_by_strength: reordena según grado ponderado para resaltar estructura.
    """
    A, vars_list = build_adj_matrix_normalized(quadratic, vars_list)
    W = np.abs(A) if abs_weights else A.copy()
    if threshold is not None:
        W[np.abs(W) < threshold] = 0.0

    # reordenar por 'fuerza' (grados ponderados) para resaltar bloques
    order = np.arange(len(vars_list))
    if reorder_by_strength:
        strength = W.sum(axis=0)
        order = np.argsort(-strength)  # descendente
        W = W[order][:, order]
        vars_list = [vars_list[i] for i in order]

    plt.figure(figsize=(max(6, len(vars_list)*0.35), max(5, len(vars_list)*0.35)))
    im = plt.imshow(W, aspect='auto')
    plt.xticks(np.arange(len(vars_list)), vars_list, rotation=90)
    plt.yticks(np.arange(len(vars_list)), vars_list)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title + (" (|w|)" if abs_weights else ""))
    plt.tight_layout()
    plt.show()

## 3) si tu QUBO está “desbalanceado” (unos pocos pesos gigantes frente a muchos pequeños)
def plot_linear_hist(linear, bins=30, title="Distribución de lineales"):
    """Histograma de los coeficientes lineales."""
    vals = np.array(list(linear.values()), dtype=float)
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.xlabel("Peso")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_quadratic_hist(quadratic, bins=30, abs_weights=True, title="Distribución de cuadráticos"):
    """Histograma de los coeficientes cuadráticos (una sola vez por par)."""
    vals = np.array(list(quadratic.values()), dtype=float)
    if abs_weights:
        vals = np.abs(vals)
        title = title + " (|w|)"
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.xlabel("Peso")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.show()

## 4) Grafo de acoplamientos del QUBO
def plot_qubo_graph(quadratic, linear=None, vars_list=None, threshold=0.0, title="Grafo QUBO"):
    """
    Grafo: nodos = variables, aristas ponderadas por pesos bilineales.
    - threshold: ignora aristas con |w| < threshold.
    - tamaño de nodo proporcional a |lineal| (si se pasa).
    """
    if not HAS_NX:
        raise RuntimeError(
            "networkx no está instalado. Ejecuta: pip install networkx"
        )
    if vars_list is None:
        vars_list = _all_vars(linear or {}, quadratic)

    G = nx.Graph()
    G.add_nodes_from(vars_list)

    for (u, v), w in quadratic.items():
        if abs(w) >= threshold:
            G.add_edge(u, v, weight=w)

    # tamaño de nodos
    if linear:
        # normaliza |lineal| para un tamaño agradable
        lv = np.array([abs(linear.get(v, 0.0)) for v in vars_list], dtype=float)
        if lv.max() > 0:
            sizes = 300 * (0.2 + 0.8 * (lv / lv.max()))  # 60% de rango útil
        else:
            sizes = 300 * np.ones(len(vars_list))
    else:
        sizes = 300 * np.ones(len(vars_list))

    # layout
    pos = nx.spring_layout(G, seed=42, k=None)  # fuerza-resorte

    plt.figure(figsize=(7, 6))
    # aristas: grosor por |peso|
    widths = [max(1.0, 3.0 * abs(d['weight']) / (1e-9 + max(abs(w) for _,_,d in G.edges(data=True) for w in [d['weight']])))
              for _,_,d in G.edges(data=True)] if G.number_of_edges() else []

    nx.draw_networkx_nodes(G, pos, node_size=sizes)
    nx.draw_networkx_labels(G, pos, font_size=8)
    if widths:
        nx.draw_networkx_edges(G, pos, width=widths)
    plt.title(title + (f" (thr={threshold})" if threshold else ""))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
