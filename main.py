"""
Entrypoint CLI reproducible para los experimentos del TFM Q4RPD.

Ejemplo de uso:

    # Simulated Annealing local con semilla 42
    python main.py --problem D14_P1 --solver SA --seed 42

    # Repetir 10 runs cambiando la semilla (0..9)
    python main.py --problem D14_P1 --solver SA --runs 10

    # D-Wave Leap Hybrid (requiere DWAVE_TOKEN en el entorno)
    python main.py --problem D14_P1 --solver DWave-Hybrid

    # D-Wave Direct QPU
    python main.py --problem D14_P1 --solver DWave-QPU

    # QCI Dirac-1 (requiere QCI_TOKEN en el entorno)
    python main.py --problem D14_P1 --solver Dirac-1

    # Usar best_params de un estudio Optuna (sobreescribe todos los pesos QUBO)
    python main.py --problem D14_P1 --solver SA --runs 5 \\
        --params-file autotune-results/q4rpd_tpe/best_params.json

Los resultados se guardan en
    results/{problem}/{solver}/run_seed{seed}/output.txt
de modo que cada ejecución es identificable por su semilla y trazable.

Parámetros del autotuner que se aplican con --params-file
---------------------------------------------------------
Los campos del best_params.json se mapean así:

  omega2_over_omega1     -> omega2 = omega1 * valor   (sobreescribe --omega2-factor)
  p_origin_penalty_w     -> problem_data directamente
  p_delivery_once_w      -> problem_data directamente
  p_position_once_w      -> problem_data directamente
  p_dest_inclusion_w     -> problem_data directamente
  p_consecutiveness_w    -> problem_data directamente
  p_capacity_w           -> problem_data directamente
  base_penalty_factor    -> monkey-patch de build_srp_qubo (era 20.0 fijo)
  reward_regular         -> monkey-patch de build_srp_qubo
  reward_tp_depot        -> monkey-patch de build_srp_qubo
  reward_depot_tp        -> monkey-patch de build_srp_qubo

Si base_penalty_factor o algún reward_* están en el JSON, se activa
automáticamente el monkey-patch de autotune_penalties.patched_qubo_builder.
Sin --params-file (o si esos campos no están en el JSON) se usa build_srp_qubo
sin modificar.
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import problem_runner as pr
import test_data

# Monkey-patch del autotuner: solo se importa si se necesita
try:
    from autotune_penalties import patched_qubo_builder as _patched_qubo_builder
    _AUTOTUNE_AVAILABLE = True
except ImportError:
    _AUTOTUNE_AVAILABLE = False


# ------------------------------------------------------------------
# Constantes y configuración por defecto
# ------------------------------------------------------------------

# Pesos por defecto idénticos a los de problems-test.ipynb.
DEFAULT_QUBO_WEIGHTS = {
    "omega1": 1.0,
    "p_origin_penalty_w": 1.0,
    "p_delivery_once_w": 4.0,
    "p_position_once_w": 4.0,
    "p_dest_inclusion_w": 0.5,
    "p_consecutiveness_w": 0.5,
    "p_capacity_w": 2.0,
    "num_reads_dwave": None,
    "num_sweeps_dwave": None,
}

# Mapeo de --solver a las banderas internas que espera run_problem.
SOLVER_CONFIGS = {
    "SA":           {"QCIUse": False, "DWaveUse": False, "DWaveModel": "",     "solver_type": ""},
    "DWave-Hybrid": {"QCIUse": False, "DWaveUse": True,  "DWaveModel": "BQM",  "solver_type": ""},
    "DWave-QPU":    {"QCIUse": False, "DWaveUse": True,  "DWaveModel": "QUBO", "solver_type": ""},
    "Dirac-1":      {"QCIUse": True,  "DWaveUse": False, "DWaveModel": "",     "solver_type": "Dirac-1"},
    "Dirac-3":      {"QCIUse": True,  "DWaveUse": False, "DWaveModel": "",     "solver_type": "Dirac-3"},
}

# Nombre de carpeta usado en results/{problem}/{folder}/
SOLVER_FOLDERS = {
    "SA":           "SimAnn",
    "DWave-Hybrid": "DWave-Hybrid",
    "DWave-QPU":    "DWave-QPU",
    "Dirac-1":      "QCI-Dirac-1",
    "Dirac-3":      "QCI-Dirac-3",
}

VALID_PROBLEMS = sorted(test_data.VEHICLES_DATA_DICT.keys())


# ------------------------------------------------------------------
# Carga de instancia
# ------------------------------------------------------------------

def load_best_params(params_file: str) -> dict:
    """Carga un best_params.json generado por autotune_penalties.py.

    Acepta tanto el payload completo (con clave "best_params") como un dict
    plano de parámetros.  Devuelve siempre el dict plano de parámetros.
    """
    with open(params_file, encoding="utf-8") as f:
        payload = json.load(f)
    # El autotuner envuelve los params en {"best_params": {...}, ...}
    return payload.get("best_params", payload)


def load_problem_data(problem_key, omega2_factor=0.15, extra_weights: dict = None):
    """Construye el diccionario problem_data para una instancia del benchmark.

    Si se pasa ``extra_weights`` (procedente de un best_params.json), los
    pesos estándar del QUBO se sobreescriben con los valores del autotuner.
    Los campos ``base_penalty_factor`` y ``reward_*`` NO se aplican aquí
    (requieren monkey-patch; se gestionan en ``run_single``).
    """
    if problem_key not in test_data.VEHICLES_DATA_DICT:
        raise ValueError(
            f"Instancia '{problem_key}' no encontrada. "
            f"Opciones válidas: {VALID_PROBLEMS}"
        )

    # Claves de penalización que viajan en problem_data
    _PEN_KEYS = (
        "p_origin_penalty_w", "p_delivery_once_w", "p_position_once_w",
        "p_dest_inclusion_w", "p_consecutiveness_w", "p_capacity_w",
    )

    omega1 = DEFAULT_QUBO_WEIGHTS["omega1"]

    # omega2_factor puede venir del autotuner (campo omega2_over_omega1)
    if extra_weights and "omega2_over_omega1" in extra_weights:
        omega2_factor = extra_weights["omega2_over_omega1"]

    problem_data = {
        "omega1": omega1,
        "omega2": omega2_factor * omega1,
        "p_origin_penalty_w":   DEFAULT_QUBO_WEIGHTS["p_origin_penalty_w"],
        "p_delivery_once_w":    DEFAULT_QUBO_WEIGHTS["p_delivery_once_w"],
        "p_position_once_w":    DEFAULT_QUBO_WEIGHTS["p_position_once_w"],
        "p_dest_inclusion_w":   DEFAULT_QUBO_WEIGHTS["p_dest_inclusion_w"],
        "p_consecutiveness_w":  DEFAULT_QUBO_WEIGHTS["p_consecutiveness_w"],
        "p_capacity_w":         DEFAULT_QUBO_WEIGHTS["p_capacity_w"],
        "num_reads_dwave":      DEFAULT_QUBO_WEIGHTS["num_reads_dwave"],
        "num_sweeps_dwave":     DEFAULT_QUBO_WEIGHTS["num_sweeps_dwave"],
        "VEHICLES_DATA":        test_data.VEHICLES_DATA_DICT[problem_key],
        "DELIVERIES_DATA":      test_data.DELIVERIES_DATA_DICT[problem_key],
        "DISTANCES":            test_data.DISTANCES_DICT[problem_key],
        "VISUALIZATION_COORDS": test_data.VISUALIZATION_COORDS_DICT[problem_key],
    }

    # Sobreescribir pesos de penalización con los del autotuner
    if extra_weights:
        for k in _PEN_KEYS:
            if k in extra_weights:
                problem_data[k] = extra_weights[k]

    return problem_data


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def set_global_seeds(seed):
    """Fija TODAS las semillas globales para reproducibilidad."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class _Tee:
    """Duplica escritura a dos streams (consola + archivo)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout(path):
    """Context manager: duplica stdout a un fichero log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open("w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, fp)
    try:
        yield
    finally:
        sys.stdout = original
        fp.close()


def print_header(args, problem_key, solver, seed, omega2_factor, best_params=None):
    print("=" * 72)
    print(f"Q4RPD - run reproducible")
    print(f"  Fecha:           {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Instancia:       {problem_key}")
    print(f"  Solver:          {solver}")
    print(f"  Semilla:         {seed}")
    print(f"  omega2_factor:   {omega2_factor}")
    print(f"  dynamic_omega:   {args.dynamic_omega}")
    print(f"  generate_graphics: {args.generate_graphics}")
    if best_params:
        print(f"  params_file:     {args.params_file}")
        print(f"  [best_params aplicados]")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
    print("=" * 72)


# ------------------------------------------------------------------
# Ejecución de un run individual
# ------------------------------------------------------------------

def run_single(problem_key, solver, seed, args, output_root):
    if solver not in SOLVER_CONFIGS:
        raise ValueError(f"Solver desconocido: {solver}. "
                         f"Opciones: {list(SOLVER_CONFIGS)}")

    config = SOLVER_CONFIGS[solver]
    folder = SOLVER_FOLDERS[solver]

    # Timestamp en el nombre del directorio → varias ejecuciones con la misma
    # semilla no se pisan entre sí.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuned_suffix = "_tuned" if args.params_file else ""
    run_label = f"run_seed{seed}{tuned_suffix}_{ts}"
    run_dir = Path(output_root) / problem_key / folder / run_label
    log_path = run_dir / "output.txt"
    summary_path = run_dir / "summary.json"

    # Semillas globales antes de cualquier llamada estocástica.
    set_global_seeds(seed)

    # Cargar best_params si se indicó --params-file
    best_params = None
    if args.params_file:
        best_params = load_best_params(args.params_file)

    problem_data = load_problem_data(
        problem_key,
        omega2_factor=args.omega2_factor,
        extra_weights=best_params,
    )

    # Determinar el omega2_factor efectivo (puede haber sido sobreescrito por best_params)
    effective_omega2_factor = problem_data["omega2"] / problem_data["omega1"]

    # Decidir si necesitamos monkey-patch (solo si hay base_penalty_factor o reward_*)
    _PATCH_KEYS = {"base_penalty_factor", "reward_regular", "reward_tp_depot", "reward_depot_tp"}
    needs_patch = bool(best_params and (_PATCH_KEYS & best_params.keys()))

    if needs_patch and not _AUTOTUNE_AVAILABLE:
        raise ImportError(
            "El best_params.json incluye base_penalty_factor / reward_* pero "
            "no se pudo importar autotune_penalties.py. "
            "Asegúrate de que el fichero está en el mismo directorio."
        )

    def _do_run():
        with tee_stdout(log_path):
            print_header(args, problem_key, solver, seed, effective_omega2_factor, best_params)
            return pr.run_problem(
                problem_data,
                QCIUse=config["QCIUse"],
                DWaveUse=config["DWaveUse"],
                DWaveModel=config["DWaveModel"],
                solver_type=config["solver_type"],
                dynamic_omega=args.dynamic_omega,
                generate_graphics=args.generate_graphics,
                seed=seed,
                use_kmedoids=False,
                kmedoids_mode="spatial",  # "spatial" | "capacitated"
                out_dir=str(run_dir) if args.generate_graphics else None,
            )

    if needs_patch:
        with _patched_qubo_builder(best_params):
            result = _do_run()
    else:
        result = _do_run()

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": problem_key,
        "solver": solver,
        "seed": seed,
        "omega2_factor": effective_omega2_factor,
        "dynamic_omega": args.dynamic_omega,
        "params_file": args.params_file,
        "best_params_applied": best_params,
        "result": {
            "total_route_duration": result.get("total_route_duration") if result else None,
            "all_deliveries_done":  result.get("all_deliveries_done")  if result else None,
            "max_iterations_reached": result.get("max_iterations_reached") if result else None,
            "n_routes": len(result.get("routes", [])) if result else 0,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[run_single] Guardado log:     {log_path}")
    print(f"[run_single] Guardado resumen: {summary_path}")
    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Entrypoint CLI reproducible para experimentos Q4RPD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--problem", required=True, choices=VALID_PROBLEMS,
                   help="Instancia del benchmark.")
    p.add_argument("--solver", required=True, choices=list(SOLVER_CONFIGS),
                   help="Solver a utilizar. 'SA' es local; los demás requieren "
                        "DWAVE_TOKEN o QCI_TOKEN en el entorno.")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla del solver. Sólo afecta a SA.")
    p.add_argument("--runs", type=int, default=1,
                   help="Número de ejecuciones (con semillas seed, seed+1, ...).")
    p.add_argument("--omega2-factor", type=float, default=0.15, dest="omega2_factor",
                   help="Factor multiplicativo para omega2 = factor * omega1.")
    p.add_argument("--dynamic-omega", action="store_true", dest="dynamic_omega",
                   help="Activa el ajuste dinámico de omega1/omega2 por tipo de SRP.")
    p.add_argument("--generate-graphics", action="store_true", dest="generate_graphics",
                   help="Genera figuras de las rutas tras la ejecución.")
    p.add_argument("--output-root", default=None,
                   help="Carpeta raíz donde guardar los resultados. Por defecto, "
                        "results/ junto a este fichero.")
    p.add_argument("--params-file", default=None, metavar="PATH",
                   help="Ruta a un best_params.json generado por autotune_penalties.py. "
                        "Sobreescribe omega2_factor y todos los pesos del QUBO, "
                        "incluyendo base_penalty_factor y los reward_* (estos últimos "
                        "se aplican vía monkey-patch de build_srp_qubo). "
                        "Ejemplo: autotune-results/q4rpd_tpe/best_params.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_root is None:
        output_root = Path(__file__).resolve().parent / "results"
    else:
        output_root = Path(args.output_root)

    summaries = []
    for i in range(args.runs):
        run_seed = args.seed + i
        print(f"\n========== RUN {i + 1}/{args.runs} (seed={run_seed}) ==========")
        try:
            summary = run_single(args.problem, args.solver, run_seed, args, output_root)
        except Exception as e:
            print(f"[main] ERROR en run con seed={run_seed}: {type(e).__name__}: {e}")
            raise
        summaries.append(summary)
        print(f"  -> coste total: {summary['result']['total_route_duration']}")

    if args.runs > 1:
        agg_path = output_root / args.problem / SOLVER_FOLDERS[args.solver] / f"aggregate_seeds{args.seed}-{args.seed + args.runs - 1}.json"
        costs = [s["result"]["total_route_duration"] for s in summaries
                 if s["result"]["total_route_duration"] is not None]
        agg = {
            "problem": args.problem,
            "solver":  args.solver,
            "n_runs":  len(summaries),
            "n_successful": len(costs),
            "seeds":   [s["seed"] for s in summaries],
            "costs":   [s["result"]["total_route_duration"] for s in summaries],
            "mean":    (float(np.mean(costs)) if costs else None),
            "std":     (float(np.std(costs, ddof=1)) if len(costs) > 1 else None),
            "min":     (float(np.min(costs)) if costs else None),
            "max":     (float(np.max(costs)) if costs else None),
            "median":  (float(np.median(costs)) if costs else None),
        }
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print("\n========== RESUMEN AGREGADO ==========")
        print(json.dumps(agg, indent=2))
        print(f"\n[main] Guardado agregado: {agg_path}")


if __name__ == "__main__":
    main()
