import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend no interactivo: nunca abre ventanas ni bloquea
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os


def create_distances_dict(coords_dict):
    """
    Crea un diccionario de distancias a partir de un diccionario de coordenadas.
    """
    coordenadas = []
    for value in coords_dict.values():
        coordenadas.append(list(value))

    distancias_condensadas = pdist(coordenadas)
    matriz_distancias = squareform(distancias_condensadas)

    dist_matrix = np.array(matriz_distancias)
    nombres_locaciones = list(coords_dict.keys())
    num_locaciones = len(nombres_locaciones)

    if dist_matrix.shape[0] != num_locaciones or dist_matrix.shape[1] != num_locaciones:
        raise ValueError(
            "La dimension de la matriz de distancias no coincide con el numero de localizaciones."
        )

    diccionario_distancias_resultado = {}
    for i in range(num_locaciones):
        for j in range(num_locaciones):
            nombre1 = nombres_locaciones[i]
            nombre2 = nombres_locaciones[j]
            distancia = dist_matrix[i, j]
            diccionario_distancias_resultado[(nombre1, nombre2)] = distancia

    return diccionario_distancias_resultado


def draw_deliveries(nombres, x, y, VISUALIZATION_COORDS=None, route=None, title=None,
                    omega1=0, omega2=0, save_image=False, out_dir=None, show=False):
    """Dibuja la ruta de entregas y opcionalmente guarda la figura.

    Parametros
    ----------
    show : bool
        Si True, llama a plt.show(). Con el backend Agg (no interactivo) esto
        no hace nada, pero se mantiene la firma por compatibilidad con notebooks.
        En CLI siempre debe ser False (default).
    save_image : bool
        Si True, guarda la figura en out_dir/imgs/<title>.png.
        El guardado ocurre ANTES del show() para evitar que este borre la figura
        de memoria en backends interactivos.
    """
    fig, ax = plt.subplots()

    # Dibujar todos los nodos
    ax.scatter(x, y)
    for i, nombre in enumerate(nombres):
        ax.text(x[i] - 4, y[i] + 1, nombre, fontsize=9)

    if route and len(route) > 0 and VISUALIZATION_COORDS and len(VISUALIZATION_COORDS) > 0:
        route_x = []
        route_y = []
        for loc in route:
            coords = VISUALIZATION_COORDS[loc]
            rx, ry = coords
            route_x.append(rx)
            route_y.append(ry)

        ax.plot(route_x, route_y, color='blue', linestyle='-',
                linewidth=2, marker='o', markersize=8, zorder=1)

    if title is None:
        title = "Ruta de Entregas"
    ax.set_title(title)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    # Guardar ANTES de show(): show() puede borrar la figura en backends interactivos
    fig_name = title + '.png'
    route_fig_path = None
    if save_image and out_dir is not None:
        img_dir = os.path.join(out_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        route_fig_path = os.path.join(img_dir, fig_name)
        fig.savefig(route_fig_path, bbox_inches='tight')
        print("  [draw] Imagen guardada: {}".format(route_fig_path))

    if show:
        plt.show()

    plt.close(fig)  # libera memoria siempre, evita acumulacion entre rutas
    return fig_name, route_fig_path
