# ReviseRCA — paper supplementary code

Revision-aware root-cause analysis for dynamic microservice systems. This repository
contains the graph/evidence revision engine, the offline workbench, fixed-parameter
RCA experiments, six shared-input baseline adapters, collection tooling, and tests
for Online Boutique and Train Ticket.

## Quick start

Use Linux and Python 3.10 or newer. The retained experiment environment used
Python 3.10.4 and PyTorch 2.6.0+cu124.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python -m dyrca_rca.experiments.run_interactive \
  --data-root examples/synthetic --out-dir outputs/smoke \
  --dataset-id supplementary-smoke --system boutique \
  --limit 1 --max-rows 100 --seed 42 --planner rule
```



## Contents

| Path | Purpose |
|---|---|
| `src/dyrca_rca/interaction/` | Evidence state, correction/preview/accept/retract, graph replay, UI, and versioned behavior extension |
| `src/dyrca_rca/models/`, `graph/`, `planner/` | Fixed model configuration, graph operations, rule and validated LLM planners |
| `src/dyrca_rca/baselines/` | Six source-guided shared-input adapters and simple controls |
| `src/dyrca_rca/experiments/` | Training, scoring, paired corrections, simulated SRE, latency and audit runners |
| `tests/` | Synthetic fixtures and regression/integrity checks |
| `collector/`, `faults/`, `runner/`, `config/`, `infra/` | Original collection and injection code/configuration, for a dedicated test cluster |
| `dataset/` | Legacy Python readers and application dependency configuration; no raw dataset |
| `results/` | Selected numerical summaries, fixed split lists, and paper result mapping |
| `docs/REPRODUCIBILITY.md` | Data contract, staged reproduction, version boundaries and runtime protocol |
| `docs/THIRD_PARTY.md` | Upstream baseline references and implementation boundaries |
| `validation/` | Validation receipts from the packaged checkout |
| `SOURCE_MANIFEST.json` | Original and packaged source hashes (identical) |

