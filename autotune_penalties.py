"""autotune_penalties.py — Auto-tuner Optuna (TPE bayesiano) para Q4RPD-QUBO.

Optimiza los pesos del QUBO de `build_srp_qubo` y los rewards por tipo de
ruta usando **Simulated Annealing** como validador (sin gastar tokens
cuánticos). Inspirado en autotune_vrp_solver.py de CosminCuruliuc/QuantumVRP
pero adaptado al esquema iterativo Q4RPD (S1→S2→S3) de este TFM.

Parámetros que se afinan
------------------------
- ``base_penalty_factor``  : k en  ``base_penalty = max(omega1, omega2) · k``
- ``omega2_over_omega1``   : ratio omega2/omega1 (omega1 se fija a 1.0)
- ``p_origin_penalty_w``   : peso relativo de la restricción "origen en pos 0"
- ``p_delivery_once_w``    : peso de unicidad de nodo
- ``p_position_once_w``    : peso de unicidad de posición
- ``p_dest_inclusion_w``   : peso de inclusión del destino
- ``p_consecutiveness_w``  : peso de consecutividad
- ``p_capacity_w``         : peso de la capacidad de peso (R6)
- ``reward_regular``       : factor c_R  en  ``(10/(5+N)) · c_R`` (Regular)
- ``reward_tp_depot``      : factor c_TD en  ``omega2 · c_TD``    (TP-Depot)
- ``reward_depot_tp``      : factor c_TT en  ``omega2 · c_TT``    (Depot-TP, TP-TP)

Métrica (lexicográfica, minimizar)
----------------------------------
- Si **validity_rate < 1.0** ⇒ score = ``(1 - validity_rate) · 1e9 + avg_cost``
- Si **validity_rate = 1.0** ⇒ score = ``avg_cost``

Donde:
- ``validity_rate``  = fracción de (instancia × seed) que terminan con
  ``all_deliveries_done == True``.
- ``avg_cost``       = media del ``total_route_duration`` sobre los runs válidos
  (los inválidos no se promedian; se penalizan con el primer término).

Aceleraciones
-------------
1. **Tiers barato→caro + pruning Optuna.** Las instancias se ordenan por
   tamaño y se agrupan en `n_tiers` bloques. Tras cada tier se reporta un
   score parcial (`trial.report`) y se consulta `trial.should_prune()`: los
   trials con pesos malos se abortan tras los bloques baratos y nunca llegan
   a tocar las instancias caras (`D21_*`, `D29_*`).
2. **Perfil SA rápido (`--fast`).** Reduce el techo de `get_dynamic_sa_params`
   (sweeps/reads) durante el tuneo. El tuneo sólo necesita *discriminar*
   pesos buenos de malos, no producir la solución de calidad final. Tras el
   study se valida el `best_params` con el SA completo.
3. **Paralelismo limpio por procesos (`--workers`).** El bucle interno
   (instancia × seed) de cada tier se reparte en un `ProcessPoolExecutor`.
   Cada proceso aplica su propio monkey-patch de `build_srp_qubo`, así que
   NO hay estado global compartido entre trials (a diferencia de
   `study.optimize(n_jobs>1)`, que con el monkey-patch sería inseguro).

Modo de uso
-----------
    # Smoke test (10 trials, sólo D6_P0 y D10_P0, 1 semilla, SA rápido)
    python autotune_penalties.py --smoke

    # Full study (200 trials, todo el benchmark, 3 semillas, 8 procesos, SA rápido)
    python autotune_penalties.py --trials 200 --seeds 3 --instances all \
        --workers 8 --fast --study-name q4rpd_full

Salidas (carpeta ``autotune-results/<study_name>/``)
----------------------------------------------------
- ``trials.csv``       : un trial por fila (incl. podados), params + métrica
- ``best_params.json`` : mejor combinación encontrada
- ``optuna_study.db``  : SQLite con el study (para reanudar/continuar)
- ``importance.csv``   : importancia relativa de cada hiperparámetro
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optuna (dependencia explícita del auto-tuner; instalar con `pip install optuna`)
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Modulos del proyecto (mismos imports que problem_runner)
import classes_and_funcs_clean as cafc
import problem_runner as pr
import test_data

# Re-exportamos las funciones originales para poder hacer monkey-patch limpio
_original_build_srp_qubo = cafc.build_srp_qubo
_original_get_dynamic_sa_params = cafc.get_dynamic_sa_params


# ---------------------------------------------------------------------------
# 1) Builder tunable: réplica fiel de cafc.build_srp_qubo, parametrizada
# ---------------------------------------------------------------------------

def make_tunable_build_srp_qubo(params: Dict):
    """Devuelve un build_srp_qubo que usa los hiperparámetros del trial.

    Mantenemos la misma firma ``(srp, omega1, omega2, penalties=None)`` para
    poder reemplazar el original sin tocar `solve_srp_with_*`.
    """

    base_penalty_factor = params["base_penalty_factor"]
    reward_regular = params["reward_regular"]
    reward_tp_depot = params["reward_tp_depot"]
    reward_depot_tp = params["reward_depot_tp"]

    def build_srp_qubo_tuned(srp, omega1, omega2, penalties=None):
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
                d = cafc.get_travel_time(map_idx[i], map_idx[j])
                if d != float('inf') and d > max_dist:
                    max_dist = d

        ref_omega = max(omega1, omega2)
        base_penalty = ref_omega * base_penalty_factor  # <-- TUNEABLE

        pen = penalties if penalties else {}
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
                    if i == j:
                        continue
                    dist = cafc.get_travel_time(map_idx[i], map_idx[j])
                    if dist == float('inf') or max_dist == 0:
                        continue
                    dist_norm = dist / max_dist
                    add_q(cafc.get_qubo_var(i, p),
                          cafc.get_qubo_var(j, p + 1),
                          omega1 * dist_norm)

        dest_idx = srp["dest_qubo_idx"]
        if dest_idx is not None and dest_idx != srp["origin_qubo_idx"]:
            for p in range(P):
                add_l(cafc.get_qubo_var(dest_idx, p),
                      omega2 * (-(p + 1.0)))

        # --- REWARDS DINÁMICOS POR TIPO DE RUTA (TUNEABLES) ---
        base_reward = 0.0
        if srp["type"] == "Regular":
            base_reward = (10.0 / (5.0 + N)) * reward_regular
        elif srp["type"] == "TP-Depot":
            base_reward = omega2 * reward_tp_depot
        elif srp["type"] in ("Depot-TP", "TP-TP"):
            base_reward = omega2 * reward_depot_tp

        if base_reward > 0:
            for i in srp["node_objects"]:
                if i == srp["origin_qubo_idx"]:
                    continue
                if dest_idx is not None and i == dest_idx:
                    continue
                dynamic_reward = base_reward  # decay desactivado, como en el original
                if dynamic_reward > 0:
                    for p in range(1, P):
                        add_l(cafc.get_qubo_var(i, p), -dynamic_reward)

        # --- RESTRICCIONES ---
        orig_idx = srp["origin_qubo_idx"]

        # R1: Origen en posición 0
        add_l(cafc.get_qubo_var(orig_idx, 0), -P_orig)
        offset += P_orig
        for p in range(1, P):
            add_l(cafc.get_qubo_var(orig_idx, p), P_orig)
        for i in range(N):
            if i != orig_idx:
                add_l(cafc.get_qubo_var(i, 0), P_orig)

        # R2 & R3: Unicidad de Nodo y Posición
        for i in range(N):
            for p1 in range(P):
                for p2 in range(p1 + 1, P):
                    add_q(cafc.get_qubo_var(i, p1),
                          cafc.get_qubo_var(i, p2),
                          P_once)
        for p in range(1, P):
            for i1 in range(N):
                for i2 in range(i1 + 1, N):
                    add_q(cafc.get_qubo_var(i1, p),
                          cafc.get_qubo_var(i2, p),
                          P_pos)

        # R4: Inclusión del destino
        if dest_idx is not None and dest_idx != orig_idx:
            dest_vars = [cafc.get_qubo_var(dest_idx, p) for p in range(1, P)]
            for v in dest_vars:
                add_l(v, -P_dest)
            for i in range(len(dest_vars)):
                for j in range(i + 1, len(dest_vars)):
                    add_q(dest_vars[i], dest_vars[j], 2 * P_dest)
            offset += P_dest

        # R5: Consecutividad
        for p in range(P - 1):
            vars_p = [cafc.get_qubo_var(i, p) for i in range(N)]
            vars_next = [cafc.get_qubo_var(j, p + 1) for j in range(N)]
            for v_next in vars_next:
                add_l(v_next, P_cons)
            for vp in vars_p:
                for vn in vars_next:
                    add_q(vp, vn, -P_cons)

        # R6: Capacidad de Peso (slacks binarias)
        max_w = srp["max_w"]
        safe_cap_w = max_w if max_w > 1 else 1.0
        num_slack_w = 4
        if max_w > 0:
            weight_terms = []
            for i in range(N):
                d_obj = srp["node_objects"].get(i)
                if d_obj:
                    w_norm = d_obj.weight / safe_cap_w
                    for p in range(P):
                        weight_terms.append((cafc.get_qubo_var(i, p), w_norm))

            current_slack_sum = 0.0
            for k in range(num_slack_w):
                coeff = 1.0 / (2 ** (k + 1))
                weight_terms.append((f"s_w_{k}", coeff))
                current_slack_sum += coeff

            residual = 1.0 - current_slack_sum
            if residual > 1e-6:
                weight_terms.append((f"s_w_res", residual))

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
        for v, c in ql.items():
            dwave_qubo[(v, v)] = dwave_qubo.get((v, v), 0) + c
        for pair, c in qq.items():
            dwave_qubo[pair] = dwave_qubo.get(pair, 0) + c

        penalties_report = {"P_origin": P_orig, "P_once": P_once,
                            "P_pos": P_pos, "P_dest": P_dest,
                            "P_cons": P_cons, "P_cap": P_cap}
        return ql, qq, offset, dwave_qubo, penalties_report

    return build_srp_qubo_tuned


@contextlib.contextmanager
def patched_qubo_builder(params: Dict):
    """Context manager que sustituye temporalmente cafc.build_srp_qubo."""
    cafc.build_srp_qubo = make_tunable_build_srp_qubo(params)
    try:
        yield
    finally:
        cafc.build_srp_qubo = _original_build_srp_qubo


# ---------------------------------------------------------------------------
# 1.bis) Perfil SA rápido (--fast): recorta sweeps/reads durante el tuneo
# ---------------------------------------------------------------------------

def _fast_sa_params(srp_problem, base_reads=100, base_sweeps=1000):
    """Versión "rápida" de cafc.get_dynamic_sa_params (~2–3× más ligero).

    Adecuada para instancias pequeñas/medianas (D6, D10, D14_P1) donde la
    topología QUBO es sencilla y el SA puede discriminar buenas/malas
    configuraciones de pesos con poco presupuesto.

    El original usa  sweeps = min(1000 + 50·n_vars, 20000),
                     reads  = min( 100 +  2·n_vars,  1000).
    """
    n_vars = srp_problem.get("num_nodes", 0) * srp_problem.get("max_pos", 0)
    if n_vars == 0:
        return base_reads, base_sweeps
    dynamic_sweeps = min(int(500 + n_vars * 20), 5000)
    dynamic_reads  = min(int(50  + n_vars *  1), 300)
    return dynamic_reads, dynamic_sweeps


def _medium_fast_sa_params(srp_problem, base_reads=100, base_sweeps=1000):
    """Versión intermedia de cafc.get_dynamic_sa_params (~1.5× más ligero).

    Diseñada para instancias con restricciones ajustadas (D14_P2, D16_P1,
    D21_P2) donde el QUBO tiene penalizaciones de capacidad o deadline muy
    estrechas y el SA necesita más potencia para encontrar soluciones
    válidas en los SRP individuales.

    Usa ~65 % del presupuesto del SA completo, lo que suele ser suficiente
    para discriminar configuraciones de pesos durante el tuneo sin disparar
    el tiempo de cómputo.
    """
    n_vars = srp_problem.get("num_nodes", 0) * srp_problem.get("max_pos", 0)
    if n_vars == 0:
        return base_reads, base_sweeps
    dynamic_sweeps = min(int(700 + n_vars * 35), 12000)
    dynamic_reads  = min(int(70  + n_vars *  1), 600)
    return dynamic_reads, dynamic_sweeps


@contextlib.contextmanager
def patched_sa_params(fast: bool = False, medium_fast: bool = False):
    """Context manager que (opcionalmente) sustituye cafc.get_dynamic_sa_params.

    Prioridad: medium_fast > fast > original (completo).
    """
    if medium_fast:
        cafc.get_dynamic_sa_params = _medium_fast_sa_params
    elif fast:
        cafc.get_dynamic_sa_params = _fast_sa_params
    else:
        yield
        return
    try:
        yield
    finally:
        cafc.get_dynamic_sa_params = _original_get_dynamic_sa_params


# ---------------------------------------------------------------------------
# 2) Configuración del benchmark
# ---------------------------------------------------------------------------

BENCHMARK_ALL = [
    "D6_P0", "D6_P1",
    "D10_P0", "D10_P1",
    "D14_P1", "D14_P2",
    "D16_P1",
    "D21_P0", "D21_P2",
    "D29_P0",
]

BENCHMARK_SMOKE = ["D6_P0", "D10_P0"]


def _instance_cost_rank(name: str) -> int:
    """Coste aproximado de una instancia = nº de entregas (prefijo D{n}_)."""
    try:
        return int(name.split("_")[0].lstrip("Dd"))
    except Exception:
        return 999


def make_tiers(instances: List[str], n_tiers: int = 3) -> List[List[str]]:
    """Ordena las instancias barato→caro y las reparte en `n_tiers` bloques.

    Los bloques se procesan en orden: primero el más barato. Tras cada
    bloque el trial reporta su score parcial y puede ser podado, de modo
    que un trial con pesos malos nunca llega a ejecutar el bloque caro
    (D21_*, D29_*).
    """
    ordered = sorted(instances, key=_instance_cost_rank)
    n_tiers = max(1, min(n_tiers, len(ordered)))
    k = len(ordered)
    sizes = [k // n_tiers + (1 if i < k % n_tiers else 0) for i in range(n_tiers)]
    tiers, idx = [], 0
    for s in sizes:
        tiers.append(ordered[idx:idx + s])
        idx += s
    return [t for t in tiers if t]


@dataclass
class TrialResult:
    """Resultado agregado de un trial."""
    validity_rate: float            # fracción de (instancia, seed) válidos
    avg_cost: float                 # coste medio sobre runs válidos
    n_runs: int
    n_valid: int
    per_instance: Dict[str, Dict]   # info detallada por instancia
    elapsed_s: float
    # Fracción de entregas completadas (continua [0,1]).
    # Suma ponderada sobre todas las runs: n_delivered_total / n_total_total.
    # Permite distinguir "13/14 entregas hechas" de "0/14", aunque ninguna
    # run sea válida al 100%; da señal continua a Optuna cuando validity=0.
    delivery_fraction: float = 1.0

    def score(self) -> float:
        """Métrica lexicográfica a MINIMIZAR (tres niveles).

        1. Si delivery_fraction < 1.0  →  (1 − delivery_fraction) · 1e9 + avg_cost
           Usa la FRACCIÓN DE ENTREGAS como penalización, no el binario
           all_deliveries_done.  Así un trial que entrega 13/14 paquetes
           (penalty ≈ 71M) supera claramente a uno que entrega 0/14 (1e9),
           y Optuna recibe gradiente real incluso cuando ningún trial termina
           válido al 100 %.
        2. Si delivery_fraction == 1.0  →  avg_cost
           Todo entregado: optimizamos solo el coste.
        """
        if self.delivery_fraction < 1.0:
            return (1.0 - self.delivery_fraction) * 1e9 + (self.avg_cost or 0.0)
        return self.avg_cost


def _partial_score(delivery_fraction: float, avg_cost: float) -> float:
    """Misma fórmula que TrialResult.score(), para reportes intermedios."""
    if delivery_fraction < 1.0:
        return (1.0 - delivery_fraction) * 1e9 + (avg_cost or 0.0)
    return avg_cost


def build_problem_data(instance_key: str, omega1: float, omega2: float,
                       penalties: Dict) -> Dict:
    """Empaqueta los inputs que necesita problem_runner.run_problem."""
    return {
        "VEHICLES_DATA": copy.deepcopy(test_data.VEHICLES_DATA_DICT[instance_key]),
        "DELIVERIES_DATA": copy.deepcopy(test_data.DELIVERIES_DATA_DICT[instance_key]),
        "DISTANCES": test_data.DISTANCES_DICT[instance_key],
        "VISUALIZATION_COORDS": test_data.VISUALIZATION_COORDS_DICT[instance_key],
        "omega1": omega1,
        "omega2": omega2,
        "p_origin_penalty_w": penalties["p_origin_penalty_w"],
        "p_delivery_once_w": penalties["p_delivery_once_w"],
        "p_position_once_w": penalties["p_position_once_w"],
        "p_dest_inclusion_w": penalties["p_dest_inclusion_w"],
        "p_consecutiveness_w": penalties["p_consecutiveness_w"],
        "p_capacity_w": penalties["p_capacity_w"],
        "num_reads_dwave": None,   # dinámicos vía get_dynamic_sa_params
        "num_sweeps_dwave": None,
    }


# ---------------------------------------------------------------------------
# 3) Worker: una sola corrida (instancia × seed) — module-level y picklable
# ---------------------------------------------------------------------------

def _run_single_task(task: Tuple) -> Tuple:
    """Ejecuta UNA corrida Q4RPD-SA para (instancia, seed).

    Es el worker que se reparte en el ProcessPoolExecutor: debe vivir a
    nivel de módulo para ser picklable. Cada proceso aplica su PROPIO
    monkey-patch (builder + perfil SA), así que no hay estado global
    compartido entre trials concurrentes.

    task = (inst, seed, params, omega1, omega2, penalties, fast_sa, medium_fast_sa)
    return = (inst, seed, valid: bool, cost: float|None, err: str|None,
              n_delivered: int, n_total: int)
    """
    inst, seed, params, omega1, omega2, penalties, fast_sa, medium_fast_sa = task
    problem_data = build_problem_data(inst, omega1, omega2, penalties)
    stream = io.StringIO()  # silencia el stdout verboso de cafc/problem_runner
    try:
        with patched_qubo_builder(params), \
                patched_sa_params(fast=fast_sa, medium_fast=medium_fast_sa), \
                contextlib.redirect_stdout(stream):
            res = pr.run_problem(
                problem_data,
                QCIUse=False, DWaveUse=False,
                DWaveModel="", solver_type="",
                dynamic_omega=False,
                generate_graphics=False,
                seed=seed, use_kmedoids=False,
            )
    except Exception as e:  # noqa: BLE001 — un fallo cuenta como inválido
        return (inst, seed, False, None, repr(e), 0, 1)

    if res:
        n_del = res.get("n_delivered", 0)
        n_tot = res.get("n_total", 1)
        cost  = res["total_route_duration"] if res.get("all_deliveries_done") else None
        valid = bool(res.get("all_deliveries_done"))
        return (inst, seed, valid, cost, None, n_del, n_tot)
    return (inst, seed, False, None, None, 0, 1)


# ---------------------------------------------------------------------------
# 4) Evaluación de un trial: tiers barato→caro + pruning + paralelismo
# ---------------------------------------------------------------------------

def evaluate_trial(params: Dict, instances: List[str], seeds: List[int],
                   trial: Optional[optuna.Trial] = None, fast_sa: bool = False,
                   medium_fast_sa: bool = False,
                   executor: Optional[ProcessPoolExecutor] = None,
                   n_tiers: int = 3, verbose: bool = False) -> TrialResult:
    """Evalúa un trial recorriendo las instancias por tiers (barato→caro).

    Si se pasa `trial`, tras cada tier se reporta el score parcial y se
    consulta `trial.should_prune()`; si procede se lanza `optuna.TrialPruned`.
    Si se pasa `executor`, el bucle (instancia × seed) de cada tier se
    ejecuta en paralelo sobre procesos.
    """
    omega1 = 1.0
    omega2 = omega1 * params["omega2_over_omega1"]

    penalties = {
        "p_origin_penalty_w": params["p_origin_penalty_w"],
        "p_delivery_once_w": params["p_delivery_once_w"],
        "p_position_once_w": params["p_position_once_w"],
        "p_dest_inclusion_w": params["p_dest_inclusion_w"],
        "p_consecutiveness_w": params["p_consecutiveness_w"],
        "p_capacity_w": params["p_capacity_w"],
    }

    tiers = make_tiers(instances, n_tiers)
    t0 = time.time()
    per_instance: Dict[str, Dict] = {}
    n_runs = 0
    n_valid = 0
    costs_valid: List[float] = []
    # Acumuladores para delivery_fraction
    n_delivered_total = 0
    n_total_total = 0

    def _finalize_per_instance() -> Dict[str, Dict]:
        out = {}
        for inst, d in per_instance.items():
            costs = d["costs"]
            out[inst] = {
                "valid": d["valid"],
                "total": d["total"],
                "avg_cost": (sum(costs) / len(costs)) if costs else float('nan'),
                "avg_delivery_frac": (d["n_delivered_sum"] / d["n_total_sum"])
                                     if d["n_total_sum"] > 0 else 0.0,
            }
        return out

    def _delivery_fraction() -> float:
        return (n_delivered_total / n_total_total) if n_total_total > 0 else 0.0

    def _push_attrs(pruned: bool, tiers_done: int) -> None:
        if trial is None:
            return
        vr = n_valid / n_runs if n_runs else 0.0
        ac = (sum(costs_valid) / len(costs_valid)) if costs_valid else 0.0
        df = _delivery_fraction()
        trial.set_user_attr("validity_rate", vr)
        trial.set_user_attr("avg_cost", ac)
        trial.set_user_attr("delivery_fraction", df)
        trial.set_user_attr("n_runs", n_runs)
        trial.set_user_attr("n_valid", n_valid)
        trial.set_user_attr("elapsed_s", time.time() - t0)
        trial.set_user_attr("pruned", pruned)
        trial.set_user_attr("tiers_done", tiers_done)
        trial.set_user_attr("per_instance", json.dumps(_finalize_per_instance()))

    for step, tier in enumerate(tiers):
        tasks = [(inst, seed, params, omega1, omega2, penalties, fast_sa, medium_fast_sa)
                 for inst in tier for seed in seeds]

        if executor is not None:
            results = list(executor.map(_run_single_task, tasks))
        else:
            results = [_run_single_task(t) for t in tasks]

        for (inst, seed, valid, cost, err, n_del, n_tot) in results:
            n_runs += 1
            n_delivered_total += n_del
            n_total_total += n_tot
            d = per_instance.setdefault(
                inst, {"valid": 0, "total": 0, "costs": [],
                       "n_delivered_sum": 0, "n_total_sum": 0})
            d["total"] += 1
            d["n_delivered_sum"] += n_del
            d["n_total_sum"] += n_tot
            if valid:
                n_valid += 1
                d["valid"] += 1
                d["costs"].append(cost)
                costs_valid.append(cost)
            elif err and verbose:
                print(f"[ERR] {inst} seed={seed}: {err}")

        # --- Reporte intermedio + pruning (sólo con study activo) ---
        if trial is not None:
            df = _delivery_fraction()
            ac = (sum(costs_valid) / len(costs_valid)) if costs_valid else 0.0
            trial.report(_partial_score(df, ac), step)
            _push_attrs(pruned=False, tiers_done=step + 1)
            if trial.should_prune():
                _push_attrs(pruned=True, tiers_done=step + 1)
                raise optuna.TrialPruned()

    validity_rate = n_valid / n_runs if n_runs else 0.0
    avg_cost = (sum(costs_valid) / len(costs_valid)) if costs_valid else 0.0

    return TrialResult(
        validity_rate=validity_rate,
        avg_cost=avg_cost,
        n_runs=n_runs,
        n_valid=n_valid,
        per_instance=_finalize_per_instance(),
        elapsed_s=time.time() - t0,
        delivery_fraction=_delivery_fraction(),
    )


# ---------------------------------------------------------------------------
# 5) Espacio de búsqueda Optuna
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> Dict:
    return {
        # base_penalty se ha probado en {2, 4, 20} en el código; muestreamos
        # log-uniformemente en un rango amplio.
        "base_penalty_factor": trial.suggest_float("base_penalty_factor", 1.0, 32.0, log=True),

        # omega2/omega1: 0.15 es el valor base de los experimentos; se explora alrededor.
        "omega2_over_omega1": trial.suggest_float("omega2_over_omega1", 0.05, 1.0, log=True),

        # Pesos relativos de las restricciones (sobre base_penalty)
        "p_origin_penalty_w":  trial.suggest_float("p_origin_penalty_w", 0.5, 4.0),
        "p_delivery_once_w":   trial.suggest_float("p_delivery_once_w", 1.0, 8.0),
        "p_position_once_w":   trial.suggest_float("p_position_once_w", 1.0, 8.0),
        "p_dest_inclusion_w":  trial.suggest_float("p_dest_inclusion_w", 0.1, 4.0),
        "p_consecutiveness_w": trial.suggest_float("p_consecutiveness_w", 0.1, 4.0),
        "p_capacity_w":        trial.suggest_float("p_capacity_w", 0.5, 8.0),

        # Rewards por tipo de ruta
        "reward_regular":  trial.suggest_float("reward_regular", 0.0, 3.0),
        "reward_tp_depot": trial.suggest_float("reward_tp_depot", 0.0, 3.0),
        "reward_depot_tp": trial.suggest_float("reward_depot_tp", 0.0, 3.0),
    }


# ---------------------------------------------------------------------------
# 6) Driver principal Optuna
# ---------------------------------------------------------------------------

def run_study(study_name: str, n_trials: int, instances: List[str],
              seeds: List[int], output_dir: Path, n_workers: int = 1,
              fast_sa: bool = False, medium_fast_sa: bool = False,
              n_tiers: int = 3, resume: bool = True) -> optuna.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{output_dir / 'optuna_study.db'}"

    sampler = TPESampler(seed=42, n_startup_trials=20)
    # Pruner: esperamos a tener 25 trials completos antes de podar.
    # Con el nuevo scoring continuo (delivery_fraction) ya hay señal real desde
    # el primer trial, pero con 11 hiperparámetros el TPE necesita un buen número
    # de trials completos para construir su modelo. n_warmup_steps=1 evita podar
    # tras el primer tier (el más barato), dando margen para al menos 2 tiers.
    pruner = MedianPruner(n_startup_trials=25, n_warmup_steps=1)

    # Parámetros de partida validados manualmente en problems-test.ipynb.
    # Se inyectan como trial 0 para que el TPE arranque desde un punto conocido
    # en lugar de explorar a ciegas durante los n_startup_trials iniciales.
    # Ajusta estos valores si encuentras una combinación mejor.
    KNOWN_GOOD_PARAMS = {
        "base_penalty_factor":  4.0,
        "omega2_over_omega1":   0.15,
        "p_origin_penalty_w":   1.0,
        "p_delivery_once_w":    4.0,
        "p_position_once_w":    4.0,
        "p_dest_inclusion_w":   0.5,
        "p_consecutiveness_w":  0.5,
        "p_capacity_w":         2.0,
        "reward_regular":       1.5,
        "reward_tp_depot":      1.0,
        "reward_depot_tp":      0.75,
    }

    if resume:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=True,
        )
        # Solo encolar si el study estaba vacío (primera carga = sin trials previos)
        if len(study.trials) == 0:
            study.enqueue_trial(KNOWN_GOOD_PARAMS)
            print("[init] Study nuevo: parámetros de partida conocidos encolados como trial 0.")
    else:
        # Borrar el study si existe (peligro: hay que confirmar)
        try:
            optuna.delete_study(study_name=study_name, storage=storage_path)
        except Exception:
            pass
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
        )
        study.enqueue_trial(KNOWN_GOOD_PARAMS)
        print("[init] Parámetros de partida conocidos encolados como trial 0.")

    trials_log = output_dir / "trials.csv"
    log_header = not trials_log.exists()

    def _write_row(row: Dict) -> None:
        nonlocal log_header
        with open(trials_log, "a", encoding="utf-8") as f:
            if log_header:
                f.write(",".join(row.keys()) + "\n")
                log_header = False
            f.write(",".join(str(v) for v in row.values()) + "\n")

    # Pool de procesos: se crea UNA vez para todo el study y se reutiliza en
    # todos los trials (evita re-spawnear workers 200 veces en Windows).
    executor = None
    if n_workers and n_workers > 1:
        executor = ProcessPoolExecutor(max_workers=n_workers)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        try:
            result = evaluate_trial(
                params, instances, seeds,
                trial=trial, fast_sa=fast_sa, medium_fast_sa=medium_fast_sa,
                executor=executor, n_tiers=n_tiers, verbose=False,
            )
        except optuna.TrialPruned:
            # Trial podado: logeamos la fila con los datos parciales que
            # evaluate_trial dejó en trial.user_attrs antes de abortar.
            ua = trial.user_attrs
            vr = ua.get("validity_rate", 0.0)
            ac = ua.get("avg_cost", 0.0)
            df = ua.get("delivery_fraction", 0.0)
            _write_row({
                "trial": trial.number,
                "score": _partial_score(df, ac),
                "validity_rate": vr,
                "delivery_fraction": df,
                "avg_cost": ac,
                "n_runs": ua.get("n_runs", 0),
                "n_valid": ua.get("n_valid", 0),
                "elapsed_s": ua.get("elapsed_s", 0.0),
                "pruned": True,
                **params,
            })
            print(f"[trial {trial.number:>3}] PRUNED  "
                  f"delivered={df:.2%}  validity={vr:.2%}  "
                  f"tiers={ua.get('tiers_done', 0)}  "
                  f"t={ua.get('elapsed_s', 0.0):5.1f}s")
            raise

        # Trial completo
        _write_row({
            "trial": trial.number,
            "score": result.score(),
            "validity_rate": result.validity_rate,
            "delivery_fraction": result.delivery_fraction,
            "avg_cost": result.avg_cost,
            "n_runs": result.n_runs,
            "n_valid": result.n_valid,
            "elapsed_s": result.elapsed_s,
            "pruned": False,
            **params,
        })

        print(f"[trial {trial.number:>3}] "
              f"delivered={result.delivery_fraction:.2%}  "
              f"validity={result.validity_rate:.2%}  "
              f"cost={result.avg_cost:8.2f}  "
              f"score={result.score():12.2f}  "
              f"t={result.elapsed_s:5.1f}s")

        return result.score()

    try:
        # n_jobs=1 a propósito: la paralelización va por --workers (procesos),
        # no por hilos de Optuna. Con hilos, el monkey-patch global de
        # cafc.build_srp_qubo se pisaría entre trials concurrentes.
        study.optimize(objective, n_trials=n_trials, n_jobs=1,
                       show_progress_bar=False)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # --- Persistir mejor params + importancias ---
    best = study.best_trial
    best_payload = {
        "best_score": best.value,
        "best_params": best.params,
        "best_user_attrs": dict(best.user_attrs),
        "instances": instances,
        "seeds": seeds,
        "n_trials": n_trials,
        "fast_sa": fast_sa,
        "medium_fast_sa": medium_fast_sa,
        "n_tiers": n_tiers,
    }
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, default=str)

    try:
        importance = optuna.importance.get_param_importances(study)
        with open(output_dir / "importance.csv", "w", encoding="utf-8") as f:
            f.write("param,importance\n")
            for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
                f.write(f"{k},{v}\n")
    except Exception as e:
        print(f"[warn] Importancia no calculable: {e}")

    return study


# ---------------------------------------------------------------------------
# 7) Validación de un best_params.json (SA COMPLETO por defecto)
# ---------------------------------------------------------------------------

def validate_best_params(best_params_path, instances: Optional[List[str]] = None,
                         seeds: Optional[List[int]] = None, fast_sa: bool = False,
                         medium_fast_sa: bool = False,
                         n_workers: int = 1, n_tiers: int = 3) -> TrialResult:
    """Carga un ``best_params.json`` y lo re-evalúa de forma honesta.

    Importante: el study suele tunear con ``--fast`` o ``--medium-fast``
    (SA recortado). Esta función usa el SA COMPLETO por defecto
    (``fast_sa=False, medium_fast_sa=False``), que es lo que de verdad
    importa para la memoria. Aplica el mismo monkey-patch que el tuneo, así
    que ``base_penalty_factor`` y los tres ``reward_*`` (que NO son
    argumentos de ``run_problem``) sí se tienen en cuenta.
    """
    with open(best_params_path, encoding="utf-8") as f:
        payload = json.load(f)
    params = payload["best_params"]
    if instances is None:
        instances = payload.get("instances", BENCHMARK_ALL)
    if seeds is None:
        seeds = list(range(5))  # más semillas que en el tuneo: validar robustez

    executor = ProcessPoolExecutor(max_workers=n_workers) if (n_workers and n_workers > 1) else None
    try:
        result = evaluate_trial(params, instances, seeds, trial=None,
                                fast_sa=fast_sa, medium_fast_sa=medium_fast_sa,
                                executor=executor, n_tiers=n_tiers, verbose=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    sa_mode = ("RÁPIDO" if fast_sa else ("MEDIO" if medium_fast_sa else "COMPLETO"))
    print(f"\n=== VALIDACIÓN best_params: {best_params_path} ===")
    print(f"  SA: {sa_mode}  |  seeds={seeds}")
    print(f"  validity_rate     = {result.validity_rate:.2%}  ({result.n_valid}/{result.n_runs})")
    print(f"  delivery_fraction = {result.delivery_fraction:.2%}")
    print(f"  avg_cost          = {result.avg_cost:.2f}")
    print(f"  score             = {result.score():.2f}")
    print(f"  por instancia (barato -> caro):")
    for inst in sorted(result.per_instance, key=_instance_cost_rank):
        d = result.per_instance[inst]
        print(f"    {inst:10s}  valid {d['valid']}/{d['total']}  "
              f"avg_delivery={d.get('avg_delivery_frac', float('nan')):.2%}  "
              f"avg_cost={d['avg_cost']:.2f}")
    return result


# ---------------------------------------------------------------------------
# 8) CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Auto-tuner Optuna (TPE) para pesos del QUBO Q4RPD.")
    p.add_argument("--trials", type=int, default=200,
                   help="Número de trials a ejecutar (default 200).")
    p.add_argument("--seeds", type=int, default=3,
                   help="Número de semillas SA por (instancia, trial). Default 3.")
    p.add_argument("--instances", type=str, default="all",
                   help="`all`, `smoke`, o lista separada por comas "
                        "(D14_P1,D21_P0,...).")
    p.add_argument("--study-name", type=str, default="q4rpd_tpe",
                   help="Nombre del estudio Optuna (también prefijo del dir).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Carpeta de resultados. Default autotune-results/<study_name>")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1),
                   help="Procesos para paralelizar el bucle (instancia × seed) "
                        "dentro de cada tier. 1 = secuencial. "
                        "Default: nº de núcleos - 1.")
    p.add_argument("--fast", action="store_true",
                   help="Perfil SA rápido (~3× más ligero que el completo). "
                        "Adecuado para instancias pequeñas/medianas (D6, D10, D14_P1). "
                        "Valida después el best_params con el SA completo.")
    p.add_argument("--medium-fast", action="store_true",
                   help="Perfil SA intermedio (~1.5× más ligero que el completo). "
                        "Recomendado para instancias con restricciones ajustadas "
                        "(D14_P2, D16_P1, D21_P2) donde --fast es insuficiente.")
    p.add_argument("--n-tiers", type=int, default=3,
                   help="Nº de bloques barato→caro para el pruning. Default 3.")
    p.add_argument("--validate-best", type=str, default=None, metavar="PATH",
                   help="Ruta a un best_params.json. En vez de tunear, RE-EVALÚA "
                        "esa combinación con el SA COMPLETO (salvo que se pase "
                        "--fast) y la mide por instancia. Respeta --instances, "
                        "--seeds y --workers.")
    p.add_argument("--no-resume", action="store_true",
                   help="Borrar study existente antes de empezar.")
    p.add_argument("--smoke", action="store_true",
                   help="Atajo: trials=10, seeds=1, instances=smoke, --fast. "
                        "Ignora otros flags relacionados.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    if args.smoke:
        args.trials = 10
        args.seeds = 1
        args.instances = "smoke"
        args.fast = True
        args.study_name = args.study_name + "_smoke"

    # Resolver lista de instancias
    if args.instances == "all":
        instances = BENCHMARK_ALL
    elif args.instances == "smoke":
        instances = BENCHMARK_SMOKE
    else:
        instances = [s.strip() for s in args.instances.split(",") if s.strip()]
        for inst in instances:
            if inst not in test_data.VEHICLES_DATA_DICT:
                raise SystemExit(f"Instancia desconocida: {inst}")

    # --fast y --medium-fast son mutuamente excluyentes; medium-fast tiene prioridad
    medium_fast = getattr(args, "medium_fast", False)
    fast = args.fast and not medium_fast

    # --- MODO VALIDACIÓN: re-evaluar un best_params.json y salir ---
    if args.validate_best:
        # Si el usuario no toca --instances, usamos las del propio payload.
        val_instances = None if args.instances == "all" else instances
        validate_best_params(
            args.validate_best,
            instances=val_instances,
            seeds=list(range(args.seeds)) if args.seeds else None,
            fast_sa=fast,
            medium_fast_sa=medium_fast,
            n_workers=args.workers,
            n_tiers=args.n_tiers,
        )
        return

    seeds = list(range(args.seeds))

    # Resolver output dir
    out = Path(args.output_dir) if args.output_dir else (
        Path("autotune-results") / args.study_name)
    out.mkdir(parents=True, exist_ok=True)

    tiers = make_tiers(instances, args.n_tiers)

    sa_mode_str = "MEDIO (--medium-fast)" if medium_fast else ("RÁPIDO (--fast)" if fast else "COMPLETO")
    print(f"=== Q4RPD Auto-tuner ===")
    print(f"  Study: {args.study_name}")
    print(f"  Output: {out.resolve()}")
    print(f"  Instances: {instances}")
    print(f"  Tiers (barato->caro): {tiers}")
    print(f"  Seeds/trial: {seeds}")
    print(f"  Trials: {args.trials}")
    print(f"  Workers (procesos): {args.workers}")
    print(f"  Perfil SA: {sa_mode_str}")
    print()

    t0 = time.time()
    study = run_study(
        study_name=args.study_name,
        n_trials=args.trials,
        instances=instances,
        seeds=seeds,
        output_dir=out,
        n_workers=args.workers,
        fast_sa=fast,
        medium_fast_sa=medium_fast,
        n_tiers=args.n_tiers,
        resume=not args.no_resume,
    )
    elapsed = time.time() - t0
    print()
    print(f"=== DONE en {elapsed/60:.1f} min ===")
    print(f"  best_score = {study.best_value:.4f}")
    print(f"  best_params = {study.best_params}")
    print(f"  artefactos en {out.resolve()}")


if __name__ == "__main__":
    main()
