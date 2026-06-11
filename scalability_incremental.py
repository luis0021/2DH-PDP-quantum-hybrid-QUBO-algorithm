import time
import pandas as pd
import matplotlib.pyplot as plt
# Nota: problem_runner_timed es una variante de problem_runner con medicion de
# tiempos de solver; debe re-derivarse para reproducir este estudio.
import problem_runner_timed as pr
import sys
sys.path.append("../../")
import test_data
import copy

def run_incremental_test(problem_key="D10_P0", min_size=5, step=1):
    """
    Ejecuta un análisis incremental sobre un ÚNICO problema, aumentando
    el número de entregas paso a paso para ver la evolución interna.
    """
    
    # 1. Cargar datos base
    if problem_key not in test_data.VEHICLES_DATA_DICT:
        print(f"Error: {problem_key} no encontrado en test_data.")
        return

    FULL_VEHICLES_DATA = test_data.VEHICLES_DATA_DICT[problem_key]
    FULL_DELIVERIES_DATA = test_data.DELIVERIES_DATA_DICT[problem_key]
    DISTANCES = test_data.DISTANCES_DICT[problem_key]
    VISUALIZATION_COORDS = test_data.VISUALIZATION_COORDS_DICT[problem_key]

    max_size = len(FULL_DELIVERIES_DATA)
    results = []

    print(f"\nStarting Incremental Analysis for {problem_key}...")
    print(f"Range: {min_size} to {max_size} deliveries (step={step})")
    print(f"{'Size':<5} | {'Solver Time (s)':<15} | {'Total Cost':<10} | {'Status':<10}")

    QCIUse=False
    DWaveUse=True
    DWaveModel = "QUBO" # QUBO, BQM, CQM
    # 2. Bucle incremental
    for n in range(min_size, max_size + 1, step):

        # Recortar entregas (Slicing)
        # Nota: Asumimos que el orden en test_data es relevante. 
        # Si hay TPs, suelen estar al principio en test_data, así que se incluirán.
        current_deliveries = FULL_DELIVERIES_DATA[:n]
        
        # Configuración del problema
        omega1 = 1.0
        omega2 = 0.15 * omega1
        
        problem_data = {
            "VEHICLES_DATA": copy.deepcopy(FULL_VEHICLES_DATA), # Copia para no alterar originales
            "DELIVERIES_DATA": copy.deepcopy(current_deliveries),
            "DISTANCES": DISTANCES,
            "VISUALIZATION_COORDS": VISUALIZATION_COORDS,
            "omega1": omega1,
            "omega2": omega2,
            "p_origin_penalty_w": 1.0,
            "p_delivery_once_w": 4.0,
            "p_position_once_w": 4.0,
            "p_dest_inclusion_w": 0.50,
            "p_consecutiveness_w": 0.50,
            "p_capacity_w": 1.0,
            "num_reads_dwave": None,
            "num_sweeps_dwave": None,
        }
        
        solver_time = 0.0
        resolution_time = 0.0
        total_cost = 0.0
        status = "Failed"
        
        try:
            # Ejecutar solver (sin gráficos para velocidad)
            res = pr.run_problem(problem_data, QCIUse=QCIUse, DWaveUse=DWaveUse, DWaveModel=DWaveModel, solver_type="Dirac-1", dynamic_omega=False, generate_graphics=False)
            
            if res:
                total_cost = res["total_route_duration"]
                solver_time = res["total_solver_time"]
                resolution_time = res["total_solution_time"]

                if res["all_deliveries_done"]:
                    status = "Solved"
                elif res["max_iterations_reached"]:
                    status = "MaxIter"
                else:
                    status = "Partial"
            
        except Exception as e:
            status = f"Err: {str(e)[:10]}"
            print(f"Exception at size {n}: {e}")

        print(f"{n:<5} | {solver_time:<15.4f} | {resolution_time:<15.4f} | {total_cost:<10.2f} | {status:<10}")

        results.append({
            "Problem": problem_key,
            "Size": n,
            "SolverTime": solver_time,
            "ResolutionTime": resolution_time,
            "Cost": total_cost,
            "Success": status == "Solved"
        })

    # 3. Guardar y Graficar
    try:
        df = pd.DataFrame(results)
        csv_name = f"incremental_results_{problem_key}.csv"
        df.to_csv(csv_name, index=False)
        print(f"\nResults saved to {csv_name}")
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Gráfica 1: Tiempo solver vs Tamaño

        if QCIUse == True:
            plt.subplot(3, 1, 1)
        else:
            plt.subplot(2, 1, 1)

        plt.plot(df["Size"], df["SolverTime"], marker='o', linestyle='-', color='blue', label='Solver Time')
        plt.title(f"{problem_key} - Solver Execution Time")
        plt.ylabel("Total Solver Time (s)")
        plt.grid(True)
        plt.legend()

        if QCIUse == True:
            # Gráfica 2: Tiempo resolution vs Tamaño
            plt.subplot(3, 1, 2)
            plt.plot(df["Size"], df["ResolutionTime"], marker='o', linestyle='-', color='blue', label='Solver Time')
            plt.title(f"{problem_key} - Solver Execution Time")
            plt.ylabel("Total Resolution Time (s)")
            plt.grid(True)
            plt.legend()

        # Gráfica 3: Coste vs Tamaño
        if QCIUse == True:
            plt.subplot(3, 1, 3)
        else:
            plt.subplot(2, 1, 2)

        df_success = df[df["Success"] == True]
        if not df_success.empty:
            plt.plot(df_success["Size"], df_success["Cost"], marker='s', linestyle='-', color='green', label='Total Cost')

        plt.title(f"{problem_key} - Solution Cost")
        plt.xlabel("Number of Deliveries (N)")
        plt.ylabel("Total Distance (km)")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plot_name = f"incremental_plot_{problem_key}.png"
        plt.savefig(plot_name)
        print(f"Plot saved to {plot_name}")
        
    except Exception as e:
        print(f"Error generating report/plots: {e}")

if __name__ == "__main__":
    # Puedes cambiar el problema aquí (ej: D29_P0 para ver el caso más grande)
    run_incremental_test(problem_key="D10_P0", min_size=4, step=1)
