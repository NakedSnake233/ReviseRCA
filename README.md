# ReviseRCA — paper supplementary code

Revision-aware root-cause analysis for dynamic microservice systems. This repository
contains the graph/evidence revision engine, the offline workbench, fixed-parameter
RCA experiments, six shared-input baseline adapters, collection tooling, and tests
for Online Boutique and Train Ticket.

## Quick start (offline, CPU)

Use Linux and Python 3.10 or newer. The retained experiment environment used
Python 3.10.4 and PyTorch 2.6.0+cu124; the reported operation timings used **one CPU
thread**, not GPU execution. CUDA is not required for the offline workflow.

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

The smoke packet is deliberately synthetic. It checks observation loading, graph
construction, and persisted revision artifacts; it does **not** reproduce paper
accuracy or train a diagnosis model. No credentials, cluster, or network calls are
needed for these checks.

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

## Scope of the released evidence

The original temporal-v3 results and the 57-field behavior extension are **different
experimental versions**. The extension is a post-hoc, single-seed mixed-cohort
experiment. It improves Online Boutique control Hit@1 from 1/15 to 5/15, but two
adapters achieve 14/15; Train Ticket remains at 0/15. These results do not establish
overall superiority. The simulated SRE uses an LLM agent, not human participants.

Raw telemetry, large tensor caches, checkpoints, private configurations, credentials,
third-party source clones, and manuscript drafts are not included. Full numerical
reproduction needs the original dataset and receipt-bound release/cache artifacts
listed in the reproduction guide. The repository is a code supplement, not a claim
that the full experimental dataset has been publicly released.

The historical scripts retain environment-specific paths and some manipulate
Kubernetes resources; review and adapt their paths/configuration before explicitly
running them on a dedicated test cluster. None runs automatically during setup.

No new license grant is asserted by this packaging operation. See the third-party
notice; upstream code is linked rather than redistributed.
