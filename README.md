# 2DH-PDP — Algoritmo híbrido cuántico-clásico con QUBO

Código de un Trabajo Fin de Máster del **MUIT** (Universidad Autónoma de
Madrid, curso 2024–25) sobre estrategias híbridas cuántico-clásicas para la
optimización de problemas de logística.

Este repositorio replica y extiende el algoritmo *Quantum for Real Package
Delivery* (Q4RPD) de Osaba, Villar-Rodriguez y Asla (*Scientific Reports*,
2024). La aportación principal es **sustituir la formulación CQM del paper
por un QUBO puro**, lo que permite resolver el mismo problema de reparto con
cuatro solvers distintos y compararlos sobre las mismas instancias:

- **Simulated Annealing** clásico (baseline reproducible por semilla).
- **D-Wave Leap Hybrid** (`LeapHybridBQMSampler`, BQM).
- **D-Wave Direct QPU** (Advantage / Advantage2).
- **QCI Dirac-1** (annealer fotónico). Dirac-3 está como esqueleto
  experimental.

Se añaden además varias heurísticas clásicas (clustering k-medoids
capacitado, filtrado geométrico, parámetros dinámicos de SA, *rewards* por
tipo de ruta) y un **auto-tuner bayesiano con Optuna** para ajustar los pesos
del QUBO.

---

## El problema (2DH-PDP)

*2-Dimensional and Heterogeneous Package Delivery with Priorities*. Un único
depósito, una flota heterogénea de camiones propios y de alquiler con
capacidad bidimensional (peso y dimensión), y un conjunto de entregas,
algunas con prioridad temporal (*TP*) y plazo (`deadline`). Restricciones
duras: capacidad (R1), plazos de TP (R2) y jornada del conductor de 480 min
(R3). Distancia y tiempo de viaje se tratan como el mismo valor, igual que en
el paper original.

El algoritmo Q4RPD resuelve el problema de forma **iterativa**: en cada
iteración configura un *Single Routing Problem* (SRP) para un vehículo
(clásico), lo resuelve como un QUBO (clásico o cuántico) y actualiza el
estado global (clásico).

---

## Instalación

Requiere **Python 3.10+**.

```bash
git clone https://github.com/luis0021/2DH-PDP-quantum-hybrid-QUBO-algorithm.git
cd 2DH-PDP-quantum-hybrid-QUBO-algorithm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> El auto-tuner (`autotune_penalties.py`) necesita además `optuna`
> (`pip install optuna`).

### Tokens cloud (opcional)

El solver de **Simulated Annealing no necesita ninguna configuración**.
Para los solvers cloud (D-Wave Hybrid/QPU, QCI Dirac):

1. Obtén tu token en [D-Wave Leap](https://cloud.dwavesys.com/leap/) y/o en
   [Quantum Computing Inc.](https://qci-prod.com/).
2. Copia la plantilla a un `.env` local (que **no** se sube al repo):
   ```bash
   cp .env.example .env
   ```
3. Rellena `DWAVE_TOKEN` y/o `QCI_TOKEN`.
4. Carga las variables antes de ejecutar:
   ```bash
   set -a; source .env; set +a        # bash/zsh
   # PowerShell: usar python-dotenv, o definir las variables a mano
   ```

Si lanzas un solver cloud sin su token, el código aborta con un
`RuntimeError` indicando qué variable falta. No hay tokens hard-codeados en
el código.

---

## Uso

El entrypoint recomendado y reproducible es **`main.py`**.

```bash
# Un run reproducible con Simulated Annealing (semilla 42)
python main.py --problem D14_P1 --solver SA --seed 42

# 10 runs (semillas 0..9) con agregado estadístico
python main.py --problem D14_P1 --solver SA --seed 0 --runs 10

# Usar los mejores pesos encontrados por el auto-tuner
python main.py --problem D14_P1 --solver SA --runs 5 \
    --params-file autotune-results/q4rpd_full/best_params.json

# Solvers cloud (requieren el token correspondiente en el entorno)
python main.py --problem D14_P1 --solver DWave-Hybrid
python main.py --problem D14_P1 --solver DWave-QPU
python main.py --problem D14_P1 --solver Dirac-1
```

### Banderas de `main.py`

| Flag | Descripción | Por defecto |
|---|---|---|
| `--problem` | Instancia: `D6_P0`, `D10_P0`, `D14_P1`, `D14_P2`, `D16_P1`, `D21_P0`, `D21_P2`, `D29_P0`, … | obligatorio |
| `--solver` | `SA`, `DWave-Hybrid`, `DWave-QPU`, `Dirac-1`, `Dirac-3` | obligatorio |
| `--seed` | Semilla del solver (sólo afecta a SA) | `42` |
| `--runs` | Nº de ejecuciones (semillas `seed..seed+runs-1`) | `1` |
| `--omega2-factor` | Factor para `omega2 = factor · omega1` | `0.15` |
| `--dynamic-omega` | Ajuste dinámico de omegas por tipo de ruta | desactivado |
| `--generate-graphics` | Genera figuras de las rutas | desactivado |
| `--params-file` | `best_params.json` del auto-tuner (sobreescribe pesos del QUBO) | — |
| `--output-root` | Carpeta raíz de resultados | `results/` |

### Salidas

```
results/{problema}/{solver}/run_seed{N}[_tuned]_{timestamp}/
├── output.txt      # log completo de la simulación
└── summary.json    # resumen estructurado (coste, rutas, éxito)
```

Con `--runs > 1` se añade un `aggregate_seeds{a}-{b}.json` con `mean`, `std`,
`min`, `max` y `median` del coste total.

---

## Auto-tuner de pesos del QUBO (Optuna)

`autotune_penalties.py` ajusta automáticamente los 11 hiperparámetros del
QUBO (factor de penalización base, ratio `omega2/omega1`, los seis pesos de
las restricciones y los tres *rewards* por tipo de ruta) usando un sampler
TPE bayesiano y **Simulated Annealing como validador** (no consume tiempo de
cómputo cuántico).

```bash
# Smoke test rápido
python autotune_penalties.py --smoke

# Estudio completo (200 trials, todo el benchmark, 8 procesos)
python autotune_penalties.py --trials 200 --seeds 3 --instances all \
    --workers 8 --fast --study-name q4rpd_full

# Re-evaluar un best_params.json con el SA completo
python autotune_penalties.py --validate-best \
    autotune-results/q4rpd_full/best_params.json --seeds 5
```

Salidas en `autotune-results/<study_name>/`: `best_params.json`,
`trials.csv` e `importance.csv`. El `best_params.json` resultante se puede
pasar directamente a `main.py` con `--params-file`.

---

## Reproducibilidad

Con `--solver SA --seed K` la ejecución es **determinista**: misma semilla →
mismo coste y misma ruta. Se fijan `random`, `numpy` y `PYTHONHASHSEED`, y se
pasa una semilla por iteración al `SimulatedAnnealingSampler`. Los solvers
cloud son estocásticos y no aceptan semilla pública, por lo que se ejecutan
varias veces y se reporta media ± desviación estándar.

---

## Estructura del repositorio

```
├── main.py                          # entrypoint CLI reproducible
├── problem_runner.py                # bucle Q4RPD (S1 → S2 → S3)
├── classes_and_funcs_clean.py       # QUBO, k-medoids, solvers, decodificación
├── aux_funcs.py                     # distancias y dibujo de rutas
├── draw_funcs.py                    # visualización del QUBO
├── test_data.py                     # instancias del benchmark
├── autotune_penalties.py            # auto-tuner Optuna/TPE
├── scalability_incremental.py       # estudio de escalabilidad por N
├── problems-test.ipynb              # notebook interactivo (legacy)
├── test-executer.ipynb              # validación de best_params
├── requirements.txt
├── .env.example                     # plantilla de tokens cloud
├── results/                         # runs individuales por solver
├── results_10_seeds/                # benchmark con 10 semillas por instancia
├── results_10_seeds_test/           # runs de prueba (10 semillas)
├── results_indiv_best_tuned_seed/   # mejores runs con pesos tuneados
├── test-indiv-results/              # validación individual de estudios
├── incremental-results/             # CSV/PNG del estudio de escalabilidad
└── autotune-results/                # estudios Optuna (best_params, trials)
```

---

## Instancias del benchmark

`D{entregas}_P{TPs}`. Las del paper original: `D14_P1`, `D14_P2`, `D16_P1`,
`D21_P0`, `D21_P2`, `D29_P0`. Instancias *toy* para depuración: `D6_P0`,
`D6_P1`, `D10_P0`, `D10_P1`.

---

## Notas

- `scalability_incremental.py` depende de `problem_runner_timed` y
  `reference_loader`, que no están incluidos; hay que re-derivarlos para
  reproducir el estudio de escalabilidad.
- El notebook `problems-test.ipynb` se mantiene como ejemplo de uso
  interactivo, pero el entrypoint reproducible es `main.py`.

## Referencia

Osaba, E., Villar-Rodriguez, E. & Asla, A. *Solving a real-world package
delivery routing problem using quantum annealers*. Scientific Reports 14,
24791 (2024).
