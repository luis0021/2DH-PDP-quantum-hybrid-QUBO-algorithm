import time
import classes_and_funcs_clean as cafc
import sys
sys.path.append("../../")
from aux_funcs import draw_deliveries, create_distances_dict
import draw_funcs

def run_problem(problem_data, QCIUse=False, DWaveUse=False, DWaveModel="", solver_type="",
                dynamic_omega=False, generate_graphics=False, seed=None, use_kmedoids=False,
                kmedoids_mode="spatial", out_dir=None):
    VEHICLES_DATA = problem_data["VEHICLES_DATA"]
    DELIVERIES_DATA = problem_data["DELIVERIES_DATA"]
    DISTANCES = problem_data["DISTANCES"]
    VISUALIZATION_COORDS = problem_data["VISUALIZATION_COORDS"]

    # Inicializacion
    available_vehicles, tp_deliveries_heap, regular_deliveries, all_deliveries_list = cafc.initialize_system(
        VEHICLES_DATA, DELIVERIES_DATA, DISTANCES)
    all_deliveries_map = {d.id: d for d in all_deliveries_list}
    iteration_count = 0
    max_iterations = 8   # Suficiente para cubrir instancias con 2 TPs + entregas adicionales
    completed_full_routes_log = []

    omega1 = problem_data["omega1"]
    omega2 = problem_data["omega2"]
    p_origin_penalty_w = problem_data["p_origin_penalty_w"]
    p_delivery_once_w = problem_data["p_delivery_once_w"]
    p_position_once_w = problem_data["p_position_once_w"]
    p_dest_inclusion_w = problem_data["p_dest_inclusion_w"]
    p_consecutiveness_w = problem_data["p_consecutiveness_w"]
    p_capacity_w = problem_data["p_capacity_w"]
    num_reads_dwave = problem_data["num_reads_dwave"]
    num_sweeps_dwave = problem_data["num_sweeps_dwave"]

    while any(d.status == "pending" for d in all_deliveries_list) and iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n--- ITERACION Q4RPD {iteration_count} ---")
        # S1: Routing Problem Setup
        active_vehicle, srp_problem_def, _, _ = cafc.setup_routing_problem(
            available_vehicles, tp_deliveries_heap, regular_deliveries, all_deliveries_map,
            use_kmedoids=use_kmedoids, visualization_coords=VISUALIZATION_COORDS,
            kmedoids_mode=kmedoids_mode)

        # Check valid SRP definition
        if not active_vehicle or not srp_problem_def:
            if not any(d.status == "pending" for d in all_deliveries_list):
                break
            print("No se pudo configurar un SRP valido en esta iteracion.")
            if all(v.has_completed_a_full_route for v in available_vehicles if not v.current_route_segments):
                print("Todos los vehiculos elegibles han completado sus rutas pero quedan entregas.")
                break
            if srp_problem_def and srp_problem_def["num_qubo_nodes"] <= 1:
                print(f"SRP trivial para {active_vehicle.id if active_vehicle else 'N/A'}, saltando.")
                if active_vehicle:
                    active_vehicle.has_completed_a_full_route = True
                continue
            time.sleep(0.1)
            continue

        # --- AJUSTE DINAMICO DE PARAMETROS SEGUN TIPO DE RUTA ---
        route_type = srp_problem_def["type"]
        dyn_omega1 = omega1
        dyn_omega2 = omega2
        if dynamic_omega == True:
            if route_type in ["Depot-TP", "TP-TP"]:
                dyn_omega1 = omega1
                dyn_omega2 = omega2
                print(f"  [Config] Modo URGENCIA ({route_type}): w1={dyn_omega1}, w2={dyn_omega2}")
            elif route_type == "TP-Depot":
                dyn_omega1 = omega1
                dyn_omega2 = omega1
                print(f"  [Config] Modo RECOLECCION ({route_type}): w1={dyn_omega1}, w2={dyn_omega2}")
            elif route_type == "Regular":
                dyn_omega1 = omega1
                dyn_omega2 = 0.5 * omega1
                print(f"  [Config] Modo RECOLECCION ({route_type}): w1={dyn_omega1}, w2={dyn_omega2}")

        # S2: Problem resolution
        penalties_dict = {
            "p_origin_penalty_w": p_origin_penalty_w,
            "p_delivery_once_w": p_delivery_once_w,
            "p_position_once_w": p_position_once_w,
            "p_dest_inclusion_w": p_dest_inclusion_w,
            "p_consecutiveness_w": p_consecutiveness_w,
            "p_capacity_w": p_capacity_w,
        }
        # Reintentos con ajuste de parametros (max_retries=1 desactiva el reintento)
        max_retries = 1
        solved_srp_result = None
        wrong_result = None

        dyn_reads, dyn_sweeps = cafc.get_dynamic_sa_params(srp_problem_def)
        current_num_reads = num_reads_dwave if num_reads_dwave is not None else dyn_reads
        current_num_sweeps = num_sweeps_dwave if num_sweeps_dwave is not None else dyn_sweeps
        current_omega1 = dyn_omega1
        current_omega2 = dyn_omega2

        for attempt in range(max_retries):
            print(f"  Intento {attempt + 1}/{max_retries} (Sweeps={current_num_sweeps}, Reads={current_num_reads}, w1={current_omega1}, w2={current_omega2})")
            if QCIUse == True:
                if solver_type == "Dirac-1":
                    solved_srp_result, wrong_result = cafc.solve_srp_with_qci_qubo_and_decode_dinamic(
                        active_vehicle, srp_problem_def, num_samples=5,
                        omega1=current_omega1, omega2=current_omega2, penalties=penalties_dict)
                elif solver_type == "Dirac-3":
                    solved_srp_result = cafc.solve_srp_with_qci_dirac3(
                        active_vehicle, srp_problem_def, num_samples=1,
                        omega1=current_omega1, omega2=current_omega2)
                else:
                    print("ERROR: solver_type must be either Dirac-1 or Dirac-3")
                    return None
            elif DWaveUse == True:
                solved_srp_result, wrong_result = cafc.solve_srp_with_dwave_qubo_and_decode_dinamic(
                    active_vehicle, srp_problem_def, num_reads=500,
                    omega1=current_omega1, omega2=current_omega2,
                    penalties=penalties_dict, DWaveModel=DWaveModel)
            else:
                iter_seed = (seed + iteration_count) if seed is not None else None
                solved_srp_result, wrong_result = cafc.solve_srp_with_qubo_and_decode(
                    active_vehicle, srp_problem_def,
                    num_reads=current_num_reads, num_sweeps=current_num_sweeps,
                    omega1=current_omega1, omega2=current_omega2,
                    iteration=iteration_count, penalties=penalties_dict, seed=iter_seed)

            if solved_srp_result is not None:
                break

            current_omega2 = max(0.1, current_omega2 * 0.5)
            current_omega1 = min(20.0, current_omega1 * 1.5)
            current_num_sweeps += 5000
            current_num_reads += 500
            print("    -> Fallo. Reintentando con parametros ajustados...")

        # S3: Solution storage and problem update
        progress_made = cafc.store_and_update_problem_improved(
            active_vehicle, solved_srp_result, tp_deliveries_heap,
            regular_deliveries, all_deliveries_map)

        if active_vehicle and active_vehicle.has_completed_a_full_route and not active_vehicle.current_route_segments:
            completed_full_routes_log.append(
                ({"vehicle_id": active_vehicle.id},
                 {"route": active_vehicle.final_route_segments},
                 {"total_duration": active_vehicle.final_route_total_time},
                 {"vehicle": active_vehicle}))

        all_vehicles_done_this_round = all(
            v.has_completed_a_full_route for v in available_vehicles if not v.current_route_segments)
        if all_vehicles_done_this_round and any(d.status == "pending" for d in all_deliveries_list):
            print("Todos los vehiculos disponibles han completado una ruta, pero aun hay entregas pendientes.")
            for d in all_deliveries_list:
                if d.status == "pending":
                    print(d)
            break

        pending_deliveries_ids = [d.id for d in all_deliveries_list if d.status == 'pending']
        print(f"Entregas pendientes: {len(pending_deliveries_ids)} - "
              f"{pending_deliveries_ids if len(pending_deliveries_ids) < 5 else pending_deliveries_ids[:5] + ['...']}")

    print("\n===== FIN DE LA SIMULACION Q4RPD =====")
    all_deliveries_done = False
    max_iterations_reached = False
    if not any(d.status == "pending" for d in all_deliveries_list):
        print("Todas las entregas completadas!")
        all_deliveries_done = True
    elif iteration_count >= max_iterations:
        print("Se alcanzo el limite maximo de iteraciones.")
        max_iterations_reached = True
    else:
        print("Simulacion terminada por otra condicion.")

    print("\nLog de Rutas Completadas:")
    [print(
        f"  Ruta {i + 1}: Vehiculo {log_entry[0]['vehicle_id']} | ruta  {log_entry[1]['route']} | "
        f"total duration = {log_entry[2]['total_duration']} | "
        f"capacity used = {log_entry[3]['vehicle'].current_load_weight}/{log_entry[3]['vehicle'].capacity_weight}")
     for i, log_entry in enumerate(completed_full_routes_log)]

    # --- VISUALIZACION ---
    nombres = []
    x = []
    y = []
    for key, value in VISUALIZATION_COORDS.items():
        nombres.append(key)
        x.append(value[0])
        y.append(value[1])

    total_route_duration = 0
    routes = []

    if generate_graphics:
        print("\nGenerando graficos...")
    for i, log_entry in enumerate(completed_full_routes_log):
        title = "Ruta_" + str(i + 1)
        raw_route = log_entry[1]['route']

        if not isinstance(raw_route, str):
            raw_route = list(raw_route)[0]

        route_clean_str = raw_route.replace(" | ", " -> ")
        route = route_clean_str.split(" -> ")

        clean_route = [route[0]]
        for loc in route[1:]:
            if loc != clean_route[-1]:
                clean_route.append(loc)
        route = clean_route

        if generate_graphics:
            try:
                # show=False: evita que plt.show() bloquee el proceso en CLI.
                # save_image se activa solo si se paso out_dir desde main.py.
                draw_deliveries(nombres, x, y, VISUALIZATION_COORDS=VISUALIZATION_COORDS,
                                route=route, title=title,
                                omega1=omega1, omega2=omega2,
                                save_image=(out_dir is not None), out_dir=out_dir,
                                show=False)
            except KeyError as e:
                print(f"Error dibujando ruta {i + 1}: Localizacion desconocida {e}")
                continue

        total_route_duration = total_route_duration + log_entry[2]['total_duration']
        routes.append(route)

    # Conteo de entregas completadas (para scoring suave en el autotuner).
    n_delivered = sum(1 for d in all_deliveries_list if d.status == "delivered")
    n_total = len(all_deliveries_list)
    print(f"\nEntregas completadas: {n_delivered}/{n_total}")
    print("\nTotal cost (duration/distance): ", total_route_duration)
    return {
        "total_route_duration": total_route_duration,
        "routes": routes,
        "all_deliveries_done": all_deliveries_done,
        "max_iterations_reached": max_iterations_reached,
        "n_delivered": n_delivered,
        "n_total": n_total,
    }
