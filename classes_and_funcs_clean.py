from datetime import datetime
import json
import math
import time
import os

import dimod
import numpy as np
import heapq
from dateutil import parser
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridBQMSampler, DWaveCliqueSampler
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler
from qci_client import QciClient
import dwave.inspector
from dwave.embedding.chain_strength import uniform_torque_compensation

# --- Constantes Globales ---
DEPOT_LOCATION = "depot"
DRIVER_WORKING_DAY = 480  # minutos
sorted_var_names_global = []


# --- Helpers ---
def _require_env(var_name):
    """Devuelve el valor de la variable de entorno o lanza un error claro.

    Usado para los tokens de servicios cloud (D-Wave, QCI). Se invoca solo en el
    momento de llamar al solver remoto, de modo que el resto del codigo (en
    particular el solver clasico de Simulated Annealing) sigue funcionando sin
    necesidad de configurar ningun token.
    """
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"La variable de entorno {var_name} no esta definida. "
            f"Configurala en tu shell o en un archivo .env antes de usar este solver. "
            f"Consulta .env.example para mas detalles."
        )
    return value


# --- Estructuras de Datos ---
class Vehicle:
    def __init__(self, id, type, capacity_weight, capacity_dimension, cost_factor=0):
        self.id = id
        self.type = type
        self.capacity_weight = capacity_weight
        self.capacity_dimension = capacity_dimension
        self.cost_factor = cost_factor
        self.current_load_weight = 0
        self.current_load_dimension = 0
        self.current_location = DEPOT_LOCATION
        self.time_on_current_route = 0
        self.current_route_segments = []
        self.final_route_segments = []
        self.final_route_total_time = 0
        self.has_completed_a_full_route = False

    def __repr__(self):
        return (f"Vehicle(id={self.id}, type='{self.type}', loc='{self.current_location}', "
                f"load_w={self.current_load_weight}/{self.capacity_weight}, done={self.has_completed_a_full_route})")


class Delivery:
    def __init__(self, id, location, weight, dimension, is_tp=False, deadline=float('inf')):
        self.id = id
        self.location = location
        self.weight = weight
        self.dimension = dimension
        self.is_tp = is_tp
        self.deadline = deadline
        self.status = "pending"

    def __repr__(self):
        return f"Delivery(id={self.id}, loc='{self.location}', tp={self.is_tp}, deadline={self.deadline})"

    def __lt__(self, other):
        if self.is_tp and other.is_tp:
            return self.deadline < other.deadline
        elif self.is_tp:
            return True
        elif other.is_tp:
            return False
        return self.id < other.id


# --- Gestion de Distancias ---
DISTANCES = {}


def get_travel_time(loc1, loc2):
    if (loc1, loc2) in DISTANCES:
        return DISTANCES[(loc1, loc2)]
    elif (loc2, loc1) in DISTANCES:
        return DISTANCES[(loc2, loc1)]
    elif loc1 == loc2:
        return 0
    print(f"CRITICAL WARNING: Distance not defined between '{loc1}' and '{loc2}'.")
    return float('inf')


# --- Inicializacion y Setup (S1) ---
def initialize_system(vehicles_data, deliveries_data, distances):
    print("--- INICIALIZACION DEL SISTEMA ---")
    global DISTANCES
    DISTANCES = distances

    all_vehicles = [Vehicle(**data) for data in vehicles_data]
    all_deliveries = [Delivery(**data) for data in deliveries_data]
    # Orden de despacho: vehiculos propios primero, dentro de cada grupo por capacidad desc.
    all_vehicles.sort(key=lambda v: (v.type == "rental", -v.capacity_weight))

    tp_deliveries_heap = [d for d in all_deliveries if d.is_tp and d.status == "pending"]
    heapq.heapify(tp_deliveries_heap)
    regular_deliveries = [d for d in all_deliveries if not d.is_tp and d.status == "pending"]

    return all_vehicles, tp_deliveries_heap, regular_deliveries, all_deliveries


# ---------------------------------------------------------------------------
# K-MEDOIDS CAPACITADO
# ---------------------------------------------------------------------------

def _euclidean_dist_coords(c1, c2):
    """Distancia euclidea entre dos tuplas (x, y)."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def compute_capacitated_kmedoids(pending_deliveries, K, cap_per_cluster,
                                  visualization_coords, max_iter=15,
                                  cap_per_cluster_d=float('inf'),
                                  mode="spatial", capacity_alpha=1.5):
    """
    K-medoids (PAM) con selector de modo + recorte post-PAM.

    Parametros
    ----------
    pending_deliveries   : lista de objetos Delivery (solo regulares pendientes)
    K                    : numero de clusters (calculado externamente como K_flota)
    cap_per_cluster      : float, capacidad maxima de PESO por cluster = target_cap
                           del vehiculo activo. Tras PAM, clusters que superen esta
                           cota se recortan eliminando los nodos mas lejanos del
                           medoide. Los nodos recortados quedan pendientes para
                           la siguiente iteracion (los absorbera un vehiculo con
                           mayor o igual capacidad).
    visualization_coords : dict {loc_name: (x, y)}
    max_iter             : iteraciones maximas del bucle PAM
    cap_per_cluster_d    : float, capacidad maxima de DIMENSION por cluster.
                           Por defecto inf (sin restriccion de dimension).
    mode                 : str, "spatial" (PAM geografico puro, comportamiento original)
                           o "capacitated" (PAM con distancia penalizada por carga:
                           dist * (1 + capacity_alpha * load_ratio)). El recorte
                           post-PAM se aplica en ambos modos.
    capacity_alpha       : float, factor de penalizacion de carga en modo "capacitated".
                           Valores tipicos: 1.0 (suave) a 3.0 (agresivo). Default 1.5.

    Retorna
    -------
    Tupla (clusters, deferred_nodes): clusters es una lista de listas de Delivery
    (no vacios, peso <= cap_per_cluster y dimension <= cap_per_cluster_d, ordenados
    por peso total desc); deferred_nodes son los que no pudieron ser ubicados.
    """
    if not pending_deliveries:
        return []

    n = len(pending_deliveries)
    K = min(K, n)

    depot_coords = visualization_coords.get(DEPOT_LOCATION, (0.0, 0.0))

    def coords(d):
        return visualization_coords.get(d.location, (0.0, 0.0))

    def dist(d1, d2):
        return _euclidean_dist_coords(coords(d1), coords(d2))

    def dist_from_depot(d):
        return _euclidean_dist_coords(coords(d), depot_coords)

    # --- Inicializacion Farthest-First ---
    remaining = list(range(n))
    medoid_indices = []

    # Primer medoide: mas lejano del deposito
    first = max(remaining, key=lambda i: dist_from_depot(pending_deliveries[i]))
    medoid_indices.append(first)
    remaining.remove(first)

    # Siguientes medoides: maxima distancia minima a los medoides ya elegidos
    for _ in range(K - 1):
        if not remaining:
            break
        nxt = max(
            remaining,
            key=lambda i: min(dist(pending_deliveries[i], pending_deliveries[m])
                              for m in medoid_indices)
        )
        medoid_indices.append(nxt)
        remaining.remove(nxt)

    # --- Bucle PAM ---
    # modo "spatial":      asignacion geografica pura (Voronoi). Comportamiento original.
    # modo "capacitated":  distancia penalizada por carga del cluster en la iteracion
    #                      anterior: dist * (1 + capacity_alpha * load_w / cap_per_cluster).
    #                      Desvía nodos hacia clusters con menos carga, produciendo
    #                      particiones mas equilibradas en peso sin abandonar la
    #                      coherencia geografica. El recorte post-PAM sigue aplicandose
    #                      en ambos modos como salvaguarda de factibilidad.
    cluster_assignments = [-1] * n

    if mode == "capacitated":
        print(f"  [K-medoids] Modo capacitated (alpha={capacity_alpha:.2f})")
    else:
        print(f"  [K-medoids] Modo spatial (geografico puro)")

    for iteration in range(max_iter):
        # Calcular carga actual de cada cluster (usada en modo capacitated)
        if mode == "capacitated":
            cluster_loads_w = [0.0] * K
            for i in range(n):
                k = cluster_assignments[i]
                if k >= 0:
                    cluster_loads_w[k] += pending_deliveries[i].weight

        # 1. Asignacion: geografica pura o penalizada por carga
        def _assignment_dist(node_idx, cluster_k):
            d = dist(pending_deliveries[node_idx],
                     pending_deliveries[medoid_indices[cluster_k]])
            if mode == "capacitated" and cap_per_cluster > 0:
                load_ratio = cluster_loads_w[cluster_k] / cap_per_cluster
                d = d * (1.0 + capacity_alpha * load_ratio)
            return d

        new_assignments = [
            min(range(K), key=lambda k, i=i: _assignment_dist(i, k))
            for i in range(n)
        ]

        # 2. Actualizar medoides (minimizar suma de distancias internas)
        new_medoid_indices = list(medoid_indices)
        changed = False
        for k in range(K):
            members = [i for i in range(n) if new_assignments[i] == k]
            if not members:
                continue
            best = min(
                members,
                key=lambda i: sum(dist(pending_deliveries[i], pending_deliveries[j])
                                  for j in members)
            )
            if best != medoid_indices[k]:
                new_medoid_indices[k] = best
                changed = True

        cluster_assignments = new_assignments
        medoid_indices = new_medoid_indices
        if not changed:
            print(f"  [K-medoids] Convergencia en iteracion {iteration + 1}/{max_iter}")
            break

    # --- Recorte post-PAM: eliminar nodos perifericos hasta cumplir cap_per_cluster Y cap_per_cluster_d ---
    # Los nodos recortados (cluster_assignments[i] = -1) quedan pendientes y se
    # re-clusterizaran en la siguiente iteracion con el vehiculo que corresponda.
    for k in range(K):
        members = [i for i in range(n) if cluster_assignments[i] == k]
        tw = sum(pending_deliveries[i].weight for i in members)
        td = sum(pending_deliveries[i].dimension for i in members)
        if tw <= cap_per_cluster and td <= cap_per_cluster_d:
            continue
        # Ordenar por distancia al deposito ASCENDENTE: eliminar primero los mas
        # cercanos al deposito. Asi el cluster retiene los nodos lejanos (caros de
        # servir en solitario) y difiere los baratos (cercanos al depot, faciles
        # de incluir en cualquier ruta futura). Logica Clarke-Wright: los mayores
        # ahorros vienen de agrupar nodos lejanos entre si.
        members.sort(
            key=lambda i: dist_from_depot(pending_deliveries[i]),
            reverse=False
        )
        # Eliminar nodos hasta que el cluster quepa en cap_per_cluster Y cap_per_cluster_d
        for i in members:
            if tw <= cap_per_cluster and td <= cap_per_cluster_d:
                break
            tw -= pending_deliveries[i].weight
            td -= pending_deliveries[i].dimension
            cluster_assignments[i] = -1  # nodo diferido a la siguiente iteracion

    # Construir clusters finales y lista de nodos diferidos
    clusters = [[] for _ in range(K)]
    deferred_nodes = []
    for i in range(n):
        k = cluster_assignments[i]
        if k >= 0:
            clusters[k].append(pending_deliveries[i])
        else:
            deferred_nodes.append(pending_deliveries[i])

    if deferred_nodes:
        dw = sum(d.weight for d in deferred_nodes)
        print(f"  [K-medoids] {len(deferred_nodes)} nodos diferidos por recorte (peso={dw:.1f})")

    # --- REDISTRIBUCION GLOBAL DE NODOS DIFERIDOS ---
    # El recorte post-PAM no sabe qué cluster se seleccionará después.
    # Antes de devolver los clusters, intentamos reubicar cada nodo diferido
    # en cualquier cluster que tenga hueco suficiente (peso Y dimension),
    # sin importar cuál sea el seleccionado en setup_routing_problem.
    # Criterio greedy: insertar primero los nodos más ligeros para maximizar
    # el número de nodos reabsorbidos; dentro de los candidatos, elegir el
    # cluster con mayor capacidad libre en peso (más holgura).
    if deferred_nodes:
        still_deferred = []
        for d in sorted(deferred_nodes, key=lambda x: x.weight, reverse=False):
            best_k, best_slack = None, -1
            for k, cluster in enumerate(clusters):
                if not cluster:
                    continue
                cw = sum(c.weight for c in cluster)
                cd = sum(c.dimension for c in cluster)
                slack_w = cap_per_cluster - cw
                slack_d = cap_per_cluster_d - cd
                if d.weight <= slack_w and d.dimension <= slack_d and slack_w > best_slack:
                    best_slack = slack_w
                    best_k = k
            if best_k is not None:
                clusters[best_k].append(d)
            else:
                still_deferred.append(d)

        absorbed = len(deferred_nodes) - len(still_deferred)
        if absorbed > 0:
            abs_locs = [d.location for d in deferred_nodes if d not in still_deferred]
            print(f"  [K-medoids] Redistribucion global: {absorbed} nodo(s) reabsorbido(s)"
                  f" en otros clusters -> {abs_locs}")
        deferred_nodes = still_deferred
        if deferred_nodes:
            dw2 = sum(d.weight for d in deferred_nodes)
            print(f"  [K-medoids] {len(deferred_nodes)} nodo(s) sin cluster tras redistribucion"
                  f" (peso={dw2:.1f})")

    non_empty = [c for c in clusters if c]
    non_empty.sort(key=lambda c: sum(d.weight for d in c), reverse=True)

    print(f"  [K-medoids] {len(non_empty)} clusters sobre {n} entregas regulares pendientes (K={K})")
    for idx, cl in enumerate(non_empty):
        locs = [d.location for d in cl]
        tw = sum(d.weight for d in cl)
        td = sum(d.dimension for d in cl)
        print(f"    Cluster {idx + 1}: {len(cl)} entregas, peso={tw:.1f}, dim={td:.1f}, nodos={locs}")

    # Retorna tupla: (clusters_factibles, nodos_diferidos)
    # Los nodos diferidos se usan en setup_routing_problem para el paso de relleno.
    return non_empty, deferred_nodes



# ---------------------------------------------------------------------------

def setup_routing_problem(available_vehicles, tp_deliveries_heap, regular_deliveries,
                          all_deliveries_map, use_kmedoids=False, visualization_coords=None,
                          kmedoids_mode="spatial"):
    print("\n--- S1: CONFIGURACION DEL PROBLEMA DE ENRUTAMIENTO (SRP) ---")

    active_vehicle = None
    for v in available_vehicles:
        if v.current_location != DEPOT_LOCATION and not v.has_completed_a_full_route:
            active_vehicle = v
            break
    if not active_vehicle:
        for v in available_vehicles:
            if not v.has_completed_a_full_route:
                active_vehicle = v
                break

    if not active_vehicle:
        print("No hay vehiculos disponibles.")
        return None, None, None, None

    print(f"Vehiculo seleccionado: {active_vehicle.id} en '{active_vehicle.current_location}'")
    srp_origin_loc = active_vehicle.current_location
    current_time = active_vehicle.time_on_current_route
    srp_type, srp_dest_obj = None, None

    if active_vehicle.current_location != DEPOT_LOCATION:
        potential_next_tp = None
        sorted_candidates = sorted(list(tp_deliveries_heap), key=lambda d: d.deadline)

        for tp in sorted_candidates:
            if tp.status != "pending": continue
            t_to_tp = get_travel_time(srp_origin_loc, tp.location)
            t_to_depot = get_travel_time(tp.location, DEPOT_LOCATION)
            cap_ok = (active_vehicle.current_load_weight + tp.weight <= active_vehicle.capacity_weight)
            time_ok = (
                        current_time + t_to_tp <= tp.deadline and current_time + t_to_tp + t_to_depot <= DRIVER_WORKING_DAY)

            if cap_ok and time_ok:
                potential_next_tp = tp
                break

        if potential_next_tp:
            srp_type = "TP-TP"
            srp_dest_obj = potential_next_tp
        else:
            srp_type = "TP-Depot"
    else:
        valid_tps = [d for d in tp_deliveries_heap if d.status == "pending"]
        if valid_tps:
            srp_type = "Depot-TP"
            srp_dest_obj = valid_tps[0]
        else:
            srp_type = "Regular"

    print(f"  Tipo SRP: {srp_type} -> Destino: {srp_dest_obj.id if srp_dest_obj else 'Depot'}")
    srp_usable_w = active_vehicle.capacity_weight - active_vehicle.current_load_weight
    srp_usable_d = active_vehicle.capacity_dimension - active_vehicle.current_load_dimension

    if srp_type == "TP-TP":
        srp_max_dur = srp_dest_obj.deadline - current_time
    elif srp_type == "Depot-TP":
        srp_max_dur = srp_dest_obj.deadline
    else:
        srp_max_dur = DRIVER_WORKING_DAY - current_time

    qubo_nodes_map = {0: srp_origin_loc}
    inv_qubo_map = {srp_origin_loc: 0}
    node_objects = {}
    idx_counter = 1

    if srp_origin_loc != DEPOT_LOCATION:
        node_objects[0] = all_deliveries_map.get(srp_origin_loc)

    srp_dest_loc = srp_dest_obj.location if srp_dest_obj else DEPOT_LOCATION
    srp_dest_qubo_idx = None

    if srp_dest_loc in inv_qubo_map:
        srp_dest_qubo_idx = inv_qubo_map[srp_dest_loc]
    else:
        qubo_nodes_map[idx_counter] = srp_dest_loc
        inv_qubo_map[srp_dest_loc] = idx_counter
        srp_dest_qubo_idx = idx_counter
        if srp_dest_obj: node_objects[idx_counter] = srp_dest_obj
        idx_counter += 1

    candidates = [d for d in list(tp_deliveries_heap) + regular_deliveries if d.status == "pending"]
    MAX_NODES_PER_SRP = 10
    print("MAX_NODES_PER_SRP=", MAX_NODES_PER_SRP)

    # Filtrado geometrico (eliptico): se descarta todo candidato cuyo desvio
    # origen -> nodo -> destino no quepa en la duracion maxima del SRP.
    filtered_candidates = []
    for d in candidates:
        if d.location in inv_qubo_map: continue
        t_to_node = get_travel_time(srp_origin_loc, d.location)
        t_node_to_dest = get_travel_time(d.location, srp_dest_loc)

        if (t_to_node + t_node_to_dest) <= srp_max_dur:
            filtered_candidates.append(d)
    # --- Seleccion y ordenacion de candidatos ---
    if use_kmedoids and srp_type == "Regular" and visualization_coords is not None:
        # K-MEDOIDS CAPACITADO: clustering global sobre todas las entregas regulares pendientes
        pending_regular_all = [d for d in regular_deliveries if d.status == "pending"]

        km_success = False
        if pending_regular_all:
            # Capacidad disponible del vehiculo activo: limite por cluster.
            target_cap = active_vehicle.capacity_weight - active_vehicle.current_load_weight
            target_cap_d = active_vehicle.capacity_dimension - active_vehicle.current_load_dimension
            total_w = sum(d.weight for d in pending_regular_all)

            # K_flota: minimo de vehiculos en orden de despacho (owned primero)
            # cuya suma de capacidades cubre el peso total pendiente.
            # Refleja la flota real y evita crear mas clusters de los necesarios.
            caps_av = [v.capacity_weight - v.current_load_weight
                       for v in available_vehicles
                       if not v.has_completed_a_full_route]
            cumulative_cap = 0
            K_flota = len(caps_av)
            for k_idx, cap in enumerate(caps_av, 1):
                cumulative_cap += cap
                if cumulative_cap >= total_w:
                    K_flota = k_idx
                    break

            # K = K_flota: con PAM geografico + recorte post-PAM, el exceso de
            # peso se difiere a iteraciones futuras (vehiculos con mayor capacidad).
            # No hace falta K_peso: el recorte garantiza que todos los clusters
            # sean factibles para el vehiculo activo sin crear clusters de mas.
            K = K_flota

            clusters, deferred_nodes = compute_capacitated_kmedoids(
                pending_regular_all, K, target_cap, visualization_coords,
                cap_per_cluster_d=target_cap_d, mode=kmedoids_mode, capacity_alpha=1)

            if clusters:
                # Seleccion por puntuacion combinada:
                #   score = 0.2 * utilizacion_de_peso + 0.8 * densidad_relativa
                feasible = [c for c in clusters if sum(d.weight for d in c) <= target_cap
                            and sum(d.dimension for d in c) <= target_cap_d]
                if not feasible:
                    feasible = clusters
                max_nodes = max(len(c) for c in feasible) if feasible else 1
                best_cluster = max(
                    feasible,
                    key=lambda c: (
                        0.2 * (sum(d.weight for d in c) / target_cap if target_cap > 0 else 0.0) +
                        0.8 * (len(c) / max_nodes if max_nodes > 0 else 0.0)
                    )
                )

                # --- RELLENO POST-SELECCION ---
                # El recorte post-PAM fue conservador (no sabia que vehiculo elegiria
                # este cluster). Ahora que sabemos la capacidad real disponible,
                # reincorporamos nodos diferidos que sean geograficamente proximos
                # al cluster seleccionado y quepan en la capacidad restante.
                # Criterio geografico: centroide del cluster seleccionado.
                if deferred_nodes:
                    cluster_weight = sum(d.weight for d in best_cluster)
                    cluster_dim = sum(d.dimension for d in best_cluster)
                    remaining_cap = target_cap - cluster_weight
                    remaining_cap_d = target_cap_d - cluster_dim
                    if remaining_cap > 0 and remaining_cap_d > 0:
                        # Centroide del cluster seleccionado
                        cx = sum(visualization_coords.get(d.location, (0.0, 0.0))[0]
                                 for d in best_cluster) / len(best_cluster)
                        cy = sum(visualization_coords.get(d.location, (0.0, 0.0))[1]
                                 for d in best_cluster) / len(best_cluster)
                        # Ordenar diferidos por distancia al centroide (mas cercanos primero)
                        deferred_sorted = sorted(
                            deferred_nodes,
                            key=lambda d: _euclidean_dist_coords(
                                visualization_coords.get(d.location, (0.0, 0.0)), (cx, cy))
                        )
                        filled = []
                        for d in deferred_sorted:
                            if d.weight <= remaining_cap and d.dimension <= remaining_cap_d:
                                best_cluster = list(best_cluster) + [d]
                                remaining_cap -= d.weight
                                remaining_cap_d -= d.dimension
                                filled.append(d.location)
                        if filled:
                            print(f"  [K-medoids] Relleno: {len(filled)} nodos reincorporados "
                                  f"-> {filled}")

                tw_sel = sum(d.weight for d in best_cluster)
                td_sel = sum(d.dimension for d in best_cluster)
                util_pct = 100.0 * tw_sel / target_cap if target_cap > 0 else 0.0
                util_pct_d = 100.0 * td_sel / target_cap_d if target_cap_d > 0 else 0.0
                print(f"  [K-medoids] Cluster asignado a {active_vehicle.id}: "
                      f"{len(best_cluster)} entregas, peso={tw_sel:.1f}/{target_cap:.1f} "
                      f"({util_pct:.1f}%), dim={td_sel:.1f}/{target_cap_d:.1f} "
                      f"({util_pct_d:.1f}%)")

                best_ids = {d.id for d in best_cluster}

                # Mantener solo candidatos del cluster que pasaron el filtro geometrico
                km_candidates = [d for d in filtered_candidates if d.id in best_ids]

                if km_candidates:
                    # Ordenar por peso descendente: meter lo mas pesado primero
                    filtered_candidates = sorted(km_candidates,
                                                 key=lambda d: d.weight, reverse=True)
                    km_success = True
                else:
                    print("  [K-medoids] Cluster seleccionado sin candidatos factibles "
                          "en tiempo; usando sectorizacion por semilla.")

        if not km_success:
            # Fallback: sectorizacion por semilla (comportamiento original)
            if len(filtered_candidates) > MAX_NODES_PER_SRP:
                seed_node = max(filtered_candidates,
                                key=lambda d: get_travel_time(DEPOT_LOCATION, d.location))
                filtered_candidates.sort(
                    key=lambda d: get_travel_time(seed_node.location, d.location))

    elif srp_type == "Regular" and len(filtered_candidates) > MAX_NODES_PER_SRP:
        # SECTORIZACION POR SEMILLA (modo original sin K-medoids)
        # El nodo mas lejano del deposito actua como semilla del cluster.
        seed_node = max(filtered_candidates,
                        key=lambda d: get_travel_time(DEPOT_LOCATION, d.location))
        filtered_candidates.sort(
            key=lambda d: get_travel_time(seed_node.location, d.location))

    else:
        # Para rutas TP-Depot, Depot-TP o TP-TP, y Regular con pocos candidatos:
        # ordenar por urgencia / camino mas directo al destino.
        filtered_candidates.sort(
            key=lambda d: get_travel_time(srp_origin_loc, d.location)
                          + get_travel_time(d.location, srp_dest_loc))
    # ---------------------------------------------------------------------------------------
    total_nodes_added_weight = 0
    nodes_added = 0
    for d in filtered_candidates:
        if nodes_added >= MAX_NODES_PER_SRP: break
        # Se compara contra srp_usable_w (capacidad RESTANTE del vehiculo), no contra
        # capacity_weight (total). Si el vehiculo ya lleva carga, capacity_weight
        # sobreestimaria el hueco disponible y se incluirian candidatos que no caben.
        total_weight_capacity_ratio  = (total_nodes_added_weight + d.weight) / srp_usable_w
        if total_weight_capacity_ratio > 1: break
        if d.location not in inv_qubo_map:
            qubo_nodes_map[idx_counter] = d.location
            inv_qubo_map[d.location] = idx_counter
            node_objects[idx_counter] = d
            idx_counter += 1
            nodes_added += 1
            total_nodes_added_weight = total_nodes_added_weight + d.weight
            print("candidate node:", d.location)

    # print("filtered_candidates: " + '\n'.join(str(p) for p in filtered_candidates) )

    num_qubo_nodes = len(qubo_nodes_map)
    srp_problem_def = {
        "type": srp_type, "origin_loc": srp_origin_loc, "dest_loc": srp_dest_loc,
        "origin_qubo_idx": 0, "dest_qubo_idx": srp_dest_qubo_idx, "max_dur": srp_max_dur,
        "max_w": srp_usable_w, "max_d": srp_usable_d, "num_nodes": num_qubo_nodes,
        "max_pos": num_qubo_nodes, "map_idx_to_id": qubo_nodes_map, "node_objects": node_objects
    }

    print(f"  SRP Definido: {num_qubo_nodes} nodos, Max Dur: {srp_max_dur:.1f}")
    return active_vehicle, srp_problem_def, tp_deliveries_heap, regular_deliveries


# --- Construccion del QUBO ---
def get_qubo_var(node, pos):
    return f"q_{node}_{pos}"


def build_srp_qubo(srp, omega1, omega2, penalties=None):
    ql, qq, offset = {}, {}, 0.0

    def add_l(v, c):
        ql[v] = ql.get(v, 0) + c

    def add_q(v1, v2, c):
        if v1 == v2:
            add_l(v1, c)
        else:
            pair = tuple(sorted((v1, v2)))
            qq[pair] = qq.get(pair, 0) + c

    N, P, map_idx = srp["num_nodes"], srp["max_pos"], srp["map_idx_to_id"]

    max_dist = 0
    for i in range(N):
        for j in range(N):
            d = get_travel_time(map_idx[i], map_idx[j])
            if d != float('inf') and d > max_dist: max_dist = d

    ref_omega = max(omega1, omega2)
    pen = penalties if penalties else {}
    # Factor base parametrizable por el autotuner (4.0 por defecto)
    base_penalty = ref_omega * pen.get("base_penalty_factor", 4.0)
    print("base_penalty: " + str(base_penalty))
    P_orig = base_penalty * pen.get("p_origin_penalty_w", 1.0)
    P_once = base_penalty * pen.get("p_delivery_once_w", 1.0)
    P_pos = base_penalty * pen.get("p_position_once_w", 1.0)
    P_dest = base_penalty * pen.get("p_dest_inclusion_w", 1.0)
    P_cons = base_penalty * pen.get("p_consecutiveness_w", 1.0)
    P_cap = base_penalty * pen.get("p_capacity_w", 1.0)

    # --- OBJETIVOS ---
    for p in range(P - 1):
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dist = get_travel_time(map_idx[i], map_idx[j])
                if dist == float('inf') or max_dist == 0: continue
                dist_norm = dist / max_dist
                add_q(get_qubo_var(i, p), get_qubo_var(j, p + 1), omega1 * dist_norm)

    dest_idx = srp["dest_qubo_idx"]
    if dest_idx is not None and dest_idx != srp["origin_qubo_idx"]:
        for p in range(P):
            add_l(get_qubo_var(dest_idx, p), omega2 * (-(p + 1.0)))

    # --- Recompensas lineales por tipo de ruta ---
    # Incentivan incluir entregas adicionales en el SRP. El factor de cada tipo
    # de ruta es parametrizable por el autotuner (reward_*). En rutas Regular la
    # recompensa decrece con el numero de nodos N para no saturar el QUBO.
    base_reward = 0.0
    if srp["type"] == "Regular":
        base_reward = (10.0 / (5.0 + N)) * pen.get("reward_regular", 1.5)
    elif srp["type"] == "TP-Depot":
        base_reward = omega2 * pen.get("reward_tp_depot", 1.0)
    elif srp["type"] in ["Depot-TP", "TP-TP"]:
        base_reward = omega2 * pen.get("reward_depot_tp", 0.75)

    if base_reward > 0:
        for i in srp["node_objects"]:
            if i == srp["origin_qubo_idx"]: continue
            if dest_idx is not None and i == dest_idx: continue
            for p in range(1, P):
                add_l(get_qubo_var(i, p), -base_reward)

    print("Rewards: ")
    print("base_reward: " + str(base_reward))

    # --- RESTRICCIONES (QUBO) ---
    orig_idx = srp["origin_qubo_idx"]

    # R1: Origen del SRP en pos 0 del QUBO
    add_l(get_qubo_var(orig_idx, 0), -P_orig)
    offset += P_orig
    for p in range(1, P): add_l(get_qubo_var(orig_idx, p), P_orig)
    for i in range(N):
        if i != orig_idx: add_l(get_qubo_var(i, 0), P_orig)

    # R2 & R3: Unicidad de Nodo y Posicion
    for i in range(N):
        for p1 in range(P):
            for p2 in range(p1 + 1, P): add_q(get_qubo_var(i, p1), get_qubo_var(i, p2), P_once)
    for p in range(1, P):
        for i1 in range(N):
            for i2 in range(i1 + 1, N): add_q(get_qubo_var(i1, p), get_qubo_var(i2, p), P_pos)

    # R4: Inclusion del destino
    if dest_idx is not None and dest_idx != orig_idx:
        dest_vars = [get_qubo_var(dest_idx, p) for p in range(1, P)]
        for v in dest_vars: add_l(v, -P_dest)
        for i in range(len(dest_vars)):
            for j in range(i + 1, len(dest_vars)): add_q(dest_vars[i], dest_vars[j], 2 * P_dest)
        offset += P_dest

    # R5: Consecutividad
    for p in range(P - 1):
        vars_p = [get_qubo_var(i, p) for i in range(N)]
        vars_next = [get_qubo_var(j, p + 1) for j in range(N)]
        for v_next in vars_next: add_l(v_next, P_cons)
        for vp in vars_p:
            for vn in vars_next: add_q(vp, vn, -P_cons)

    # R6: Weight Capacity Constraint (Slack variables)
    max_w = srp["max_w"]
    safe_cap_w = max_w if max_w > 1 else 1.0
    num_slack_w = 4
    if max_w > 0 and pen.get("use_capacity_qubo", True):
        weight_terms = []
        for i in range(N):
            d_obj = srp["node_objects"].get(i)
            if d_obj:
                w_norm = d_obj.weight / safe_cap_w
                for p in range(P): weight_terms.append((get_qubo_var(i, p), w_norm))

        current_slack_sum = 0.0
        for k in range(num_slack_w):
            coeff = 1.0 / (2 ** (k + 1))
            weight_terms.append((f"s_w_{k}", coeff))
            current_slack_sum += coeff

        residual = 1.0 - current_slack_sum
        if residual > 0.000001: weight_terms.append((f"s_w_res", residual))

        limit = 1.0
        for var, coeff in weight_terms:
            add_l(var, P_cap * (coeff ** 2 - 2 * coeff * limit))
        for i in range(len(weight_terms)):
            for j in range(i + 1, len(weight_terms)):
                v1, c1 = weight_terms[i]
                v2, c2 = weight_terms[j]
                add_q(v1, v2, 2 * P_cap * c1 * c2)
        offset += P_cap * (limit ** 2)

    dwave_qubo = {}
    for v, c in ql.items(): dwave_qubo[(v, v)] = dwave_qubo.get((v, v), 0) + c
    for pair, c in qq.items(): dwave_qubo[pair] = dwave_qubo.get(pair, 0) + c

    penalties_report = {"P_origin": P_orig, "P_once": P_once, "P_pos": P_pos, "P_dest": P_dest, "P_cons": P_cons,
                        "P_cap": P_cap}
    return ql, qq, offset, dwave_qubo, penalties_report


# --- Decodificacion ---
def decode_solution_sample(sample, srp_problem_def):
    map_idx = srp_problem_def["map_idx_to_id"]
    active_assignments = []

    if isinstance(sample, list):
        for i, val in enumerate(sample):
            if val == 1:
                var_name = sorted_var_names_global[i]
                if var_name.startswith('q_'):
                    parts = var_name.split('_')
                    active_assignments.append((int(parts[2]), int(parts[1])))
    else:
        for var_name, val in sample.items():
            if val == 1 and var_name.startswith('q_'):
                parts = var_name.split('_')
                active_assignments.append((int(parts[2]), int(parts[1])))

    active_assignments.sort()
    decoded_path, served_deliveries, validation_errors = [], [], []
    total_time, total_w, total_d = 0, 0, 0
    is_valid = True

    if not active_assignments or active_assignments[0][0] != 0:
        return {"is_valid": False, "errors": ["Fallo estructural: No empieza en 0"], "path_str": "Ruta vacia"}

    # Check estructural: las restricciones one-hot (un nodo por posicion y una
    # posicion por nodo) son penalizaciones blandas en el QUBO, asi que el
    # solver puede violarlas. Si varios nodos comparten posicion la ruta esta
    # mal definida y se rechaza como fallo estructural.
    positions = [p for p, _ in active_assignments]
    nodes = [n for _, n in active_assignments]
    if len(positions) != len(set(positions)) or len(nodes) != len(set(nodes)):
        return {"is_valid": False,
                "errors": ["Fallo estructural: posiciones o nodos duplicados (one-hot violado)"],
                "path_str": " -> ".join(str(map_idx.get(n, "?")) for _, n in active_assignments)}

    prev_loc = srp_problem_def["origin_loc"]
    # step_idx (y no `pos`) garantiza que solo se omite el tramo del primer
    # nodo (el origen), aunque hubiera posiciones repetidas.
    for step_idx, (pos, node_idx) in enumerate(active_assignments):
        curr_loc = map_idx.get(node_idx)
        if not curr_loc: continue

        if step_idx > 0:
            dt = get_travel_time(prev_loc, curr_loc)
            if dt == float('inf'):
                validation_errors.append(f"Salto imposible: {prev_loc} -> {curr_loc}")
                is_valid, dt = False, 0
            total_time += dt

        decoded_path.append(curr_loc)
        d_obj = srp_problem_def["node_objects"].get(node_idx)
        if d_obj and d_obj.status == "pending":
            served_deliveries.append(d_obj)
            total_w += d_obj.weight
            total_d += d_obj.dimension
        prev_loc = curr_loc

    dest_loc = srp_problem_def["dest_loc"]
    if prev_loc != dest_loc:
        dt = get_travel_time(prev_loc, dest_loc)
        if dt == float('inf'):
            validation_errors.append(f"Retorno imposible: {prev_loc} -> {dest_loc}")
            is_valid, dt = False, 0
        total_time += dt
        decoded_path.append(dest_loc)

    if total_time > srp_problem_def["max_dur"]:
        validation_errors.append(f"Tiempo excedido: {total_time:.1f} > {srp_problem_def['max_dur']:.1f}")
        is_valid = False

    # R1 Capacidad de peso: la penalizacion P_cap del QUBO es blanda, asi que
    # este chequeo actua como barrera dura contra soluciones que la violen.
    if total_w > srp_problem_def["max_w"]:
        validation_errors.append(f"Capacidad peso excedida: {total_w:.1f} > {srp_problem_def['max_w']:.1f}")
        is_valid = False

    # R1 Capacidad de dimension: no esta modelada en el QUBO, pero el decoder
    # la comprueba para descartar soluciones fisicamente inviables.
    if total_d > srp_problem_def["max_d"]:
        validation_errors.append(f"Capacidad dimension excedida: {total_d:.1f} > {srp_problem_def['max_d']:.1f}")
        is_valid = False

    if srp_problem_def["dest_qubo_idx"] is not None:
        target_obj = srp_problem_def["node_objects"].get(srp_problem_def["dest_qubo_idx"])
        arrived_at_target = (decoded_path[-1] == target_obj.location) if target_obj else False
        if target_obj and target_obj not in served_deliveries and not arrived_at_target:
            validation_errors.append(f"Fallo Objetivo: TP destino ({target_obj.id}) no visitado/servido.")
            is_valid = False

    return {
        "final_loc": dest_loc, "served": served_deliveries, "duration": total_time,
        "load_w": total_w, "load_d": total_d, "path_str": " -> ".join(decoded_path),
        "is_valid": is_valid, "errors": validation_errors, "assignments": active_assignments
    }


# --- Solvers ---
def get_dynamic_sa_params(srp_problem, base_reads=100, base_sweeps=1000):
    n_vars = srp_problem.get("num_nodes", 0) * srp_problem.get("max_pos", 0)
    if n_vars == 0: return base_reads, base_sweeps
    dynamic_sweeps = min(int(base_sweeps + (n_vars * 50)), 20000)
    dynamic_reads = min(int(base_reads + (n_vars * 2)), 1000)
    return dynamic_reads, dynamic_sweeps


def solve_srp_with_qubo_and_decode(vehicle, srp_problem, num_reads=None, num_sweeps=None, omega1=1.0, omega2=1.0,
                                   penalties=None, seed=None, **kwargs):
    print("\n--- S2: SOLVER SIMULATED ANNEALING ---")
    ql, qq, offset, dwave_qubo, _ = build_srp_qubo(srp_problem, omega1, omega2, penalties)
    if not dwave_qubo: return None, None

    sampler = SimulatedAnnealingSampler()
    if seed is not None:
        response = sampler.sample_qubo(dwave_qubo, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    else:
        response = sampler.sample_qubo(dwave_qubo, num_reads=num_reads, num_sweeps=num_sweeps)
    result = decode_solution_sample(response.first.sample, srp_problem)

    if result:
        result["ql"] = ql
        result["qq"] = qq
        if result["is_valid"]:
            print(f"  [OK] Ruta Valida: {result['path_str']} (T={result['duration']:.1f})")
            return result, None
        else:
            print(f"  [X] Ruta INVALIDA: {result['path_str']} - {result['errors']}")
            return None, result
    return None, None


def solve_srp_with_dwave_qubo_and_decode_dinamic(vehicle, srp_problem, num_reads=5, omega1=1.0, omega2=1.0,
                                                 penalties=None, DWaveModel="QUBO", **kwargs):
    print("\n--- S2: SOLVER D-WAVE QUANTUM ANNEALER ---")
    ratio = 1.0
    base_penalty_dwave = omega1 * ratio
    dwave_penalties = penalties if penalties else {
        "p_origin_penalty_w": base_penalty_dwave * 1.2, "p_delivery_once_w": base_penalty_dwave,
        "p_position_once_w": base_penalty_dwave, "p_dest_inclusion_w": base_penalty_dwave,
        "p_consecutiveness_w": base_penalty_dwave * 0.6, "p_capacity_w": base_penalty_dwave * 1.5
    }
    # True: la restriccion de capacidad (R6, variables de slack) se incluye en el QUBO
    dwave_penalties["use_capacity_qubo"] = True

    ql, qq, offset, dwave_qubo, _ = build_srp_qubo(srp_problem, omega1, omega2, penalties=dwave_penalties)
    if not dwave_qubo: return None, None

    try:
        MY_TOKEN = _require_env("DWAVE_TOKEN")
        if DWaveModel == "QUBO":
            max_coeff = max([abs(c) for c in dwave_qubo.values()])
            chain_strength = uniform_torque_compensation
            print(f"    chain_strength={chain_strength}")
            sampler = EmbeddingComposite(DWaveSampler(token=MY_TOKEN, solver={'name': 'Advantage2_system1'}))

            response = sampler.sample_qubo(dwave_qubo, num_reads=num_reads, annealing_time=1000,
                                           chain_strength=chain_strength, auto_scale=True, label="Q4RPD - SRP Solve - EmbComp")
            valor_calculado = response.info['embedding_context']['chain_strength']
            print(f"chain_strength: {valor_calculado}")
            print(f"Chain break fraction: {response.record.chain_break_fraction.mean()}")
            emb = response.info.get('embedding_context', {}).get('embedding', {})
            if emb:
                lens = [len(ch) for ch in emb.values()]
                print(
                    f"  vars_logicas={len(emb)}  qubits_fisicos={sum(lens)}  cadena_max={max(lens)}  media={sum(lens) / len(lens):.1f}")
            print(f"Response variables: {len(response.variables)}")
        elif DWaveModel == "BQM":
            bqm = dimod.BQM.from_qubo(dwave_qubo)
            response = LeapHybridBQMSampler(token=MY_TOKEN).sample(bqm)
        else:
            raise ValueError("Unsupported DWave Model")

        # D-Wave devuelve varias muestras; se contabiliza su validez y despues
        # se toma la primera valida en orden de energia ascendente.
        print(f"  D-Wave devolvio {len(response)} muestras. Buscando valida...")

        total_samples = num_reads
        correctly_decoded = 0
        wrongly_decoded = 0
        not_decoded = 0
        for sample, energy in response.data(['sample', 'energy']):
            decoded = decode_solution_sample(sample, srp_problem)
            if decoded:
                if decoded["is_valid"]:
                    correctly_decoded += 1
                else:
                    wrongly_decoded += 1
            else:
                not_decoded += 1

        print(f"Total samples: {total_samples}")
        print(f"correctly_decoded: {correctly_decoded}")
        print(f"wrongly_decoded: {wrongly_decoded}")
        print(f"not_decoded: {not_decoded}")

        print_counter = 0
        dwave_success = False
        for sample, energy in response.data(['sample', 'energy']):
            print_counter += 1
            decoded = decode_solution_sample(sample, srp_problem)

            if decoded:
                if decoded["is_valid"]:
                    print(
                        f"  [OK] Ruta D-Wave Valida (E={energy:.2f}): {decoded['path_str']} (T={decoded['duration']:.1f})")
                    result = decoded
                    result["ql"] = ql
                    result["qq"] = qq
                    dwave_success = True
                    break
                elif print_counter < 20:
                    # Log de las primeras muestras invalidas para diagnostico
                    print(f"    [Debug] Muestra invalida: {decoded['errors']}")

        if not dwave_success:
            print("  [!] Ninguna muestra de D-Wave fue valida.")

    except Exception as e:
        print(f"  Error D-Wave: {e}")
        dwave_success = False

    # --- FALLBACK SYSTEM ---
    if not dwave_success:
        print(f"\n    FALLBACK ACTIVADO (D-Wave fallo)")
        print("      Conmutando a Simulated Annealing local.")
        return solve_srp_with_qubo_and_decode(
            vehicle, srp_problem,
            num_reads=500,
            num_sweeps=1000,
            omega1=omega1,
            omega2=omega2,
            penalties=penalties
        )

    return result, None

def solve_srp_with_qci_qubo_and_decode_dinamic(vehicle, srp_problem, num_samples=1, omega1=1.0, omega2=1.0,
                                               penalties=None, **kwargs):
    print("\n--- S2: SOLVER QCI DIRAC-1 ---")
    ql, qq, _, _, _ = build_srp_qubo(srp_problem, omega1, omega2, penalties=penalties)

    global sorted_var_names_global
    all_vars = set(ql.keys())
    for (u, v) in qq.keys(): all_vars.update([u, v])
    sorted_var_names_global = sorted(list(all_vars))
    var_map = {name: i for i, name in enumerate(sorted_var_names_global)}
    n_vars = len(sorted_var_names_global)

    Q_matrix = np.zeros((n_vars, n_vars))
    for u, c in ql.items(): Q_matrix[var_map[u], var_map[u]] = c
    for (u, v), c in qq.items():
        i, j = var_map[u], var_map[v]
        Q_matrix[i, j] = c / 2.0;
        Q_matrix[j, i] = c / 2.0

    max_val = np.max(np.abs(Q_matrix))
    Q_matrix_normalized = Q_matrix / max_val if max_val > 0 else Q_matrix

    token = _require_env("QCI_TOKEN")
    try:
        client = QciClient(api_token=token, url="https://api.qci-prod.com")
        file_resp = client.upload_file(file={'file_config': {'qubo': {"data": Q_matrix_normalized}}})
        job_body = client.build_job_body(job_type="sample-qubo", qubo_file_id=file_resp["file_id"],
                                         job_params={"device_type": "dirac-1", "num_samples": num_samples})
        job_resp = client.process_job(job_body=job_body)

        if job_resp and "results" in job_resp and job_resp["results"]:
            for idx, sol in enumerate(job_resp["results"].get("solutions", [])):
                decoded = decode_solution_sample(sol, srp_problem)
                if decoded and decoded["is_valid"]:
                    print(f"  [OK] Ruta QCI Valida: {decoded['path_str']}")
                    decoded["ql"], decoded["qq"] = ql, qq
                    return decoded, None
    except Exception as e:
        print(f"  Error QCI: {e}")

    print(f"\n   FALLBACK SA")
    return solve_srp_with_qubo_and_decode(vehicle, srp_problem, num_reads=1000, num_sweeps=5000, omega1=omega1,
                                          omega2=omega2, penalties=penalties)


def solve_srp_with_qci_dirac3(vehicle, srp_problem, num_samples=1, omega1=1.0, omega2=1.0, penalties=None, **kwargs):
    print("\n--- S2: SOLVER QCI DIRAC-3 (HAMILTONIAN) ---")
    penalty_val = max(omega1, 10.0) * 6.0
    qci_penalties = {
        "p_origin_penalty_w": penalty_val, "p_delivery_once_w": penalty_val * 1.2,
        "p_position_once_w": penalty_val, "p_dest_inclusion_w": penalty_val,
        "p_consecutiveness_w": penalty_val * 0.5, "p_cap": penalty_val * 2
    }

    ql, qq, _, _, _ = build_srp_qubo(srp_problem, max(omega1, 10.0),
                                     omega2 if srp_problem["type"] not in ["Depot-TP", "TP-TP"] else 0.0,
                                     penalties=qci_penalties)

    global sorted_var_names_global
    all_vars = set(ql.keys())
    for (u, v) in qq.keys(): all_vars.update([u, v])
    sorted_var_names_global = sorted(list(all_vars))
    var_map = {name: i for i, name in enumerate(sorted_var_names_global)}
    n_vars = len(sorted_var_names_global)

    Q_matrix = np.zeros((n_vars, n_vars))
    for u, c in ql.items(): Q_matrix[var_map[u], var_map[u]] = c
    for (u, v), c in qq.items():
        i, j = var_map[u], var_map[v]
        Q_matrix[i, j] = c / 2.0;
        Q_matrix[j, i] = c / 2.0

    max_val = np.max(np.abs(Q_matrix))
    Q_matrix_norm = Q_matrix / max_val if max_val > 0 else Q_matrix

    token = _require_env("QCI_TOKEN")
    try:
        client = QciClient(api_token=token, url="https://api.qci-prod.com")
        file_resp = client.upload_file(file={'file_config': {'hamiltonian': {'data': Q_matrix_norm}}})
        job_body = client.build_job_body(job_type="sample-hamiltonian", hamiltonian_file_id=file_resp["file_id"],
                                         job_params={"device_type": "dirac-3", "num_samples": num_samples})
        job_resp = client.process_job(job_body=job_body)

        if job_resp and "results" in job_resp and job_resp["results"]:
            for sol in job_resp["results"].get("solutions", []):
                decoded = decode_solution_sample(sol, srp_problem)
                if decoded and decoded["is_valid"]:
                    print(f"  [OK] Ruta Dirac-3 Valida: {decoded['path_str']}")
                    decoded["ql"], decoded["qq"] = ql, qq
                    return decoded
    except Exception as e:
        print(f"  Error Dirac-3: {e}")

    print(f"\n   FALLBACK SA")
    return solve_srp_with_qubo_and_decode(vehicle, srp_problem, num_reads=200, num_sweeps=25000, omega1=omega1,
                                          omega2=omega2, penalties=penalties)


# --- Actualizacion y Cierre (S3) ---
def store_and_update_problem_improved(vehicle, result, tp_heap, reg_list, all_map):
    if not result: return False

    print("\n--- S3: ACTUALIZACION ---")
    vehicle.current_route_segments.append(result)
    vehicle.current_location = result["final_loc"]
    vehicle.time_on_current_route += result["duration"]
    vehicle.current_load_weight += result["load_w"]

    count = 0
    served_ids = set()

    for d in result["served"]:
        real_d = all_map.get(d.id)
        if real_d:
            real_d.status = "delivered"
            served_ids.add(real_d.id)
            count += 1

    if count > 0:
        new_tp_list = [d for d in tp_heap if d.id not in served_ids]
        tp_heap[:] = new_tp_list
        heapq.heapify(tp_heap)
        new_reg_list = [d for d in reg_list if d.id not in served_ids]
        reg_list[:] = new_reg_list

    print(f"  Entregas servidas: {count}")

    if vehicle.current_location == DEPOT_LOCATION:
        print(f"  Vehiculo {vehicle.id}: ruta completa.")
        vehicle.has_completed_a_full_route = True
        vehicle.final_route_total_time = vehicle.time_on_current_route
        vehicle.final_route_segments = " | ".join(
            [seg["path_str"] for seg in vehicle.current_route_segments])
        vehicle.current_route_segments = []

    return count > 0