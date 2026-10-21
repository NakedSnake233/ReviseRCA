# Reproduction guide

## Levels of reproduction

1. **Code validation:** the README smoke and full `pytest` suite use synthetic data
   and run offline. They test invariants, not paper accuracy.
2. **Result inspection:** `results/paper_registered_values.json` retains the
   original paper's registered values; `paper_tables/` gives selected LaTeX layouts.
   `\Result{...}` table cells resolve against the registered values, so these table
   snippets are not standalone manuscripts. Mixed-cohort and behavior-extension
   CSVs are separate and must not overwrite original RQ3/RQ4 results.
3. **Full experiment reproduction:** requires the original raw case directories,
   audited index, frozen grouped splits, cached tensors, and fitted checkpoints.
   These data-dependent prerequisites are not bundled in this code supplement.

## Input data contract

Each application has case directories containing:

```text
CASE/
  metadata.json                 # orchestration identity; never infer the root from its name
  ground_truth.json             # root_cause_service; scoring/supervised training only
  topology/pre.json
  topology/during.json
  topology/post.json            # future evidence, excluded from cutoff-scoped inference
  traces/trace_summary.csv       # prefer this over traces_raw.json
  metrics/latency_p99.csv        # timestamp, labels_json, value
```

Snapshots contain `timestamp`, `pods`, `services`, and `deployments`. Span summaries
contain `trace_id`, `span_id`, `parent_span_id`, `service`,
`start_time_unix_nano`, `duration_nano`, and `status_code`. Unknown metric identities
remain unavailable; never repair attribution with the injection target. See
`examples/synthetic/` for a minimal concrete packet and `tests/` for control fixtures.

## Runner map

Run `python -m MODULE --help` for the complete parameter contract. Always pass an
explicit data root and fresh output directory. Keep seed 42 and fixed model settings.
Start with `--limit` before launching data-heavy processing.

| Stage | Module under `dyrca_rca.experiments` | Additional prerequisite |
|---|---|---|
| Four-copy index | `index_four_copy_collection` | Batch summaries and frozen collection plan |
| Verified release preparation | `prepare_tosem_release` | Collection audit |
| Temporal model and caches | `optimize_harness` | Verified case packets and compatible earlier checkpoint for paired comparison |
| Six baselines and controls | `run_dynamic_baselines` | Release cache/checkpoint/split |
| Paired evidence revision | `complete_revised_experiments` | Receipt-bound original experiment stages |
| Simulated SRE | `simulate_sre_agent` | Pilot protocol and compatible checkpoint |
| Mixed T/C cohort | `rebalance_shortcut` | Original release, ranking and audit |
| Behavior + tag-separated encoding | `run_behavior_update` | Frozen mixed protocol, original and mixed caches |
| Construction/inference latency | `run_tosem_latency` | Original temporal-v3 checkpoint and verified cases |

For the behavior extension, with these prerequisites available:

```bash
python -m dyrca_rca.experiments.run_behavior_update \
  --data-root "$RCA_DATA_ROOT" --mixed-root "$RCA_MIXED_ROOT" \
  --system boutique --out-dir outputs/behavior-boutique \
  --limit 12 --seed 42 --planner rule --budget-seconds 600
```

`results/shortcut_review_20261002/*/split.json` publishes the exact case lists, but
is **not** a substitute for the full receipt-bound mixed run directory required by
this command. Remove `--limit` only for a planned full run. Original protocols
contain absolute paths: relocate paths explicitly, regenerate derived manifests,
and record the relocation; never edit historical hash receipts to claim identity.
The 2026-10-02 two-platform full extension experiment took about 310 seconds on its
recorded host. This is historical timing, not a promise on another computer.

## LLM simulation

The model used for the paper's simulated SRE role was `gpt-5.6-luna`. It is not a
human study. Credentials are supplied only through environment variables:
`DYRCA_LLM_API_KEY`, `DYRCA_LLM_BASE_URL`, and `DYRCA_LLM_MODEL`.
Do not save actual keys in source/configuration. The simulated-agent CLI defaults to
`--planner rule`; LLM execution requires explicit `--planner llm` and
`--allow-external-evidence`. Validate provider/model availability separately.
Rule fallback is deterministic and model output uses schema validation; fallback
records must not be reported as successful LLM responses. Setup/tests do not call
an external API.

## Version boundaries and limitations

- Original paired revision/SRE experiments use the original temporal-v3 schema.
- The behavior extension appends 14 fields to 43 existing fields and has separate
  checkpoint/schema validation. All six adapters receive the same additional input.
- T/C mixed splits were fixed after historical results had been seen; one held-out
  control root-service group per application does not establish broad generalization.
- The 10,000-row trace-prefix cap truncates 279/297 Boutique and 61/291 Train Ticket
  mixed cases. Train Ticket's 16,392 eligible metric rows have unresolved identity.
- Keep the host's available memory above 10 GiB; the recorded extension was run
  sequentially with an external 8 GiB per-process address-space cap and total
  20-minute deadline. The runner checks memory/time between stages; reproducing the
  hard wall-time/process cap requires an external supervisor.
- No HPO, per-method test tuning, or case filtering by method performance.

## Actual runtime protocol

There were **200 measured samples per operation per application**, not 500: ten
cases, five excluded warm-ups, twenty measured repetitions per case, one CPU
thread. Construction/inference selected the first ten lexicographically sorted
verified cases; interaction operations selected the first ten frozen test cases.
The OS page cache was not cleared. Each interaction sequence reset its base state;
preview/accept/conflict then ran with their prescribed intermediate states.
Preview excludes proposal preparation; accept/conflict include persistence.
Base graph ranges: Boutique 28–29 nodes / 29–30 edges; Train Ticket 131 nodes /
146–147 edges. These edge counts exclude the computational reverse edges (tensor
counts are 58–60 and 292–294). CUDA was not used for the timing measurements.

## Packaging changes

`SOURCE_MANIFEST.json` binds copied source to the local snapshot. The Train Ticket dependency configuration is included under `dataset/`, matching
the source-layout lookup. All copied code files retain their original hashes; no
experiment, data, collection or injection source was changed. Historical
source scripts/docs are preserved as reference, not endorsed as one-click portable
launchers. Use module CLIs and explicit paths for a relocated checkout.
