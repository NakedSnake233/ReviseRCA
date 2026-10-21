# Dynamic Graph RCA Experiment

This repository evaluates whether dynamic graph dependency updates improve root cause analysis after topology changes. The goal is graph update quality and RCA accuracy, not model hyperparameter search.

## Rules

- Do not run HPO, AutoML, model-family selection, or hidden-dimension/dropout/lr sweeps.
- Use the fixed model configuration in `src/dyrca_rca/models/config.py` for every graph variant and experiment group.
- Do not modify fault injection scripts, collector scripts, verifier scripts, or raw data.
- Prefer `traces/trace_summary.csv`; use raw traces only when the summary is insufficient.
- Keep smoke runs bounded with `--limit`.

## Reproducible Scripts

Smoke run:

```bash
bash scripts/run_dygraph_rca_smoke.sh
```

The smoke script writes:

- `artifacts/dygraph_rca_smoke/index.csv`
- `artifacts/dygraph_rca_smoke/plans/*.json`
- `artifacts/dygraph_rca_smoke/model_data/{static_pre,rule_full}/*.pt`
- `artifacts/dygraph_rca_smoke/results/*/metrics_overall.csv`
- `artifacts/dygraph_rca_smoke/results/*/metrics_by_scenario.csv`
- `artifacts/dygraph_rca_smoke/results/*/predictions.csv`
- `artifacts/dygraph_rca_smoke/results/graph_update_eval/*.csv`
- `artifacts/dygraph_rca_smoke/report.md`

Full run:

```bash
bash scripts/run_dygraph_rca_full.sh
```

Full run with LLM graph plans and LLM evidence reranking:

```bash
export DYRCA_LLM_BASE_URL="https://api.gpt.ge/v1/responses"
export DYRCA_LLM_API_KEY="<your-api-key>"
export DYRCA_LLM_MODEL="gpt-5"
bash scripts/run_dygraph_rca_full.sh
```

When these environment variables are set, the full script first caches LLM graph update plans under `artifacts/dygraph_rca_full/llm_plans`, includes the `llm_full` graph variant, and runs `dyrca_rca.rerank.llm_reranker` for prediction reranking. Invalid LLM outputs or API failures fall back to deterministic rules with JSON schema validation.

Optional environment overrides:

```bash
DATA_ROOT=data/experiments OUT_ROOT=artifacts/my_run SEED=42 SPLIT=random bash scripts/run_dygraph_rca_smoke.sh
```

If cached LLM graph-update plans exist in `artifacts/dygraph_rca_full/llm_plans`, the full script includes `llm_full`. Otherwise, it skips that variant unless LLM API environment variables are configured.

## Manual Commands

Build an index:

```bash
python -m dyrca_rca.data.index \
  --data-root data/experiments \
  --out artifacts/dygraph_rca_smoke/index.csv \
  --limit 10
```

Generate rule plans:

```bash
python -m dyrca_rca.planner.rule_planner \
  --index artifacts/dygraph_rca_smoke/index.csv \
  --out-dir artifacts/dygraph_rca_smoke/plans \
  --limit 10
```

Materialize graphs:

```bash
python -m dyrca_rca.dataset.materialize \
  --index artifacts/dygraph_rca_smoke/index.csv \
  --variant rule_full \
  --out-dir artifacts/dygraph_rca_smoke/model_data \
  --limit 10
```

Run a fixed-config model:

```bash
python -m dyrca_rca.experiments.run_experiment \
  --data-root data/experiments \
  --index artifacts/dygraph_rca_smoke/index.csv \
  --graph-variant rule_full \
  --model weighted_graphsage \
  --split random \
  --seed 42 \
  --limit 10 \
  --out-dir artifacts/dygraph_rca_smoke/results/rule_full_weighted_graphsage
```

Evaluate graph-update quality:

```bash
python -m dyrca_rca.experiments.evaluate_graph_updates \
  --index artifacts/dygraph_rca_smoke/index.csv \
  --plans-dir artifacts/dygraph_rca_smoke/plans \
  --graphs-dir artifacts/dygraph_rca_smoke/model_data/rule_full \
  --out-dir artifacts/dygraph_rca_smoke/results/graph_update_eval \
  --limit 10
```

Run the deterministic reranker:

```bash
python -m dyrca_rca.rerank.rule_reranker \
  --predictions artifacts/dygraph_rca_smoke/results/rule_full_weighted_graphsage/predictions.csv \
  --data-root data/experiments \
  --graphs-root artifacts/dygraph_rca_smoke/model_data/rule_full \
  --out-dir artifacts/dygraph_rca_smoke/results/rule_full_weighted_graphsage/rerank \
  --top-k 5 \
  --limit 10
```

Run the LLM evidence reranker:

```bash
python -m dyrca_rca.rerank.llm_reranker \
  --predictions artifacts/dygraph_rca_full/results/llm_full_weighted_graphsage/predictions.csv \
  --data-root data/experiments \
  --graphs-root artifacts/dygraph_rca_full/model_data/llm_full \
  --out-dir artifacts/dygraph_rca_full/results/llm_full_weighted_graphsage/rerank_llm \
  --top-k 5 \
  --fallback rule
```

By default, the LLM reranker uses `target_service` as an incident prior when that service appears inside the GNN Top-K. It does not introduce candidates outside Top-K. Use `--disable-target-prior` for a stricter no-target-prior ablation, and `--allow-promotions` only when testing direct evidence-based LLM promotions.

Generate a report:

```bash
python -m dyrca_rca.reporting.make_report \
  --index artifacts/dygraph_rca_smoke/index.csv \
  --results-root artifacts/dygraph_rca_smoke/results \
  --graph-update-dir artifacts/dygraph_rca_smoke/results/graph_update_eval \
  --out artifacts/dygraph_rca_smoke/report.md
```

## Report Contents

The report includes:

- Dataset summary
- Graph update metrics
- RCA metrics overall
- RCA metrics by scenario
- Ablation table by graph variant and model
- Top examples where graph updates fixed static-GNN Top@1 misses
- Top examples where the reranker changed ranking
- Failure analysis for T3 empty endpoints and T4 low-weight canary cases

## Expected Variants

- `static_pre`: pre-change topology only
- `naive_during`: during-change topology directly
- `rule_node`: rule plan node operations only
- `rule_edge`: rule plan edge and weight operations only
- `rule_full`: full rule plan
- `llm_full`: cached LLM plan when available, with rule fallback elsewhere in the planner interface

## Fixed Model Config

Both `weighted_graphsage` and `edge_aware_gat` use the same constants across all graph variants:

- `hidden_dim = 128`
- `num_layers = 2`
- `dropout = 0.1`
- `lr = 3e-3`
- `weight_decay = 1e-5`
- `epochs = 200`
- `batch_size = 1 graph`
- `seed = 42`
- `early_stop_patience = 30`
- `pairwise_ranking_coef = 1.0`

These constants are shared across graph variants. They are not a search space, and the experiment does not perform HPO, AutoML, or per-variant tuning.
