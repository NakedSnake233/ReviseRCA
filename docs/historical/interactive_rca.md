# Offline interactive RCA, first implementation

This entry point adds revisioned corrections to observations and graph plans.
It does not operate a Kubernetes cluster. The initial implementation consumes
timestamped topology snapshots and bounded trace summaries, then rebuilds the
active graph, fixed-coefficient edge features, tensors and optional frozen-GNN
predictions. Metrics/logs, diagnosis reranking, online recovery,
automatic question selection and cross-incident memory are subsequent work.

## Three-task visual workbench

The local workbench supports T2 endpoint migration, T3 endpoint redirect/empty
endpoints, and T4 observed version-group splits. It uses the same feedback state
machine and frozen inference as the offline runner. Start it with:

```bash
python -m dyrca_rca.experiments.serve_interactive \
  --data-root /home/user/dyrca/data/experiments \
  --out-dir /home/user/dyrca/artifacts/interactive_workbench \
  --limit 3 --max-rows 2000 --seed 42 --planner rule \
  --checkpoint /home/user/dyrca/artifacts/interactive_route_training/model.pt \
  --port 8765
```

Open `http://127.0.0.1:8765`. The default bounded selection alternates T2/T3/T4;
`--selection /absolute/path/names.json` selects an explicit list of experiment
directory names. Select a service, enter endpoint IPs or version proportions,
provide the review reason, then save a proposal. Preview compares the current
formal result with the hypothetical result. Accept/reject/retract buttons follow
the feedback lifecycle. A failed commit retains the previous result. Events can
be exported as JSONL for offline use. The interface contains no root-cause labels.

The graph highlights added/removed/changed nodes and typed edges, uses aligned
positions across revisions, and can focus on a selected service and its neighbors.
The explanation lists exact graph/evidence differences and model rank/score
changes; it does not assert causal proof. Without a checkpoint, rankings remain
absent. Per-commit before/after explanations and audit sessions are atomically
saved in the workbench output directory and restored on restart.

T2/T3 forms correct endpoint lists in the latest snapshot. T4 constructs a
validated full replacement plan with observed version children and route weights
summing to one. Existing local plan edits must be retired before this full-plan
form is used. `unknown` is an explicitly unknown version group, never inferred
to be v1. Version children currently share service-level trace aggregates; this
does not constitute version-level fault localization. Route proportions are
operator inputs, not inferred from Pod counts. The UI is a local Python tool,
binds to loopback, and uses a request token for writes; it is not a multi-user
authentication service or a Kubernetes control plane.

## Frozen-model paired evaluation

To simulate workbench collection over 100 distinct incidents using the existing
checkpoint, run:

```bash
python -m dyrca_rca.experiments.collect_simulated_feedback \
  --data-root /home/user/dyrca/data/experiments \
  --out-dir /home/user/dyrca/artifacts/interactive_simulated_100 \
  --limit 100 --seed 42 --planner rule --max-rows 2000 \
  --checkpoint /home/user/dyrca/artifacts/interactive_route_training/model.pt \
  --training-manifest /home/user/dyrca/artifacts/interactive_route_training/training.json
```

Selection is seeded and alternates Boutique T2/T3/T4 cases, excluding the
checkpoint's training and validation samples. `--limit` counts feedback proposals
and distinct incidents, not audit commands. The 100-proposal run writes 100
`feedback.jsonl` lines, 200 proposal/acceptance `events.jsonl` lines, 300
`workbench_commands.jsonl` lines (including previews), and 100 outcome rows in
`feedback_results.jsonl`. Sessions under `workbench/` are native resumable UI
bundles. Repeating the same configuration resumes without duplicating events.
All proposals are collected through `Workbench.command` before scoring reads
any ground-truth labels. The actor is explicitly `simulated_operator`.

Open these collected sessions using `serve_interactive` with
`--out-dir artifacts/interactive_simulated_100/workbench --limit 100
--selection artifacts/interactive_simulated_100/selection.json`, keeping the
checkpoint and trace bound identical. Use a separate port from an active human
workbench. For separate re-evaluation, provide that workbench directory to
`evaluate_interactive` and set `--feedback-source simulation`; simulated sessions
cannot be silently scored as real operator feedback.

The simulation confirms observed endpoint lists and adds observed version groups
with assumed equal proportions. If a T4 case has no observable split, an endpoint
confirmation fallback is explicitly recorded. This collection does not fabricate
errors in the automatic baseline or select corrections based on root labels.

The evaluator compares static pre-change topology (with the same current trace
window), automatic rule updates, and recorded operator feedback using one
compatible checkpoint. Model weights are checked unchanged. Train/validation
overlap is rejected using the checkpoint-bound training manifest. Predictions
and feedback are generated before root labels are read for scoring.

```bash
python -m dyrca_rca.experiments.evaluate_interactive \
  --data-root /home/user/dyrca/data/experiments \
  --out-dir /home/user/dyrca/artifacts/interactive_operator_evaluation \
  --workbench-dir /home/user/dyrca/artifacts/interactive_workbench \
  --checkpoint /home/user/dyrca/artifacts/interactive_route_training/model.pt \
  --training-manifest /home/user/dyrca/artifacts/interactive_route_training/training.json \
  --limit 3 --max-rows 2000 --seed 42 --planner rule
```

Use the same selection and trace bound as the workbench. Cases without committed
feedback are marked `awaiting_operator` and excluded from the operator group;
they are never silently reported as completed human evaluation. Group sizes and
label availability are explicit. Missing root labels are not scored; a labeled
root absent from the model's service candidates scores zero. Hit@1, Hit@3, MRR,
per-case deltas, graph versions and checkpoint hashes are written to JSONL/CSV.
`comparisons.json` contains structural explanations and `REPORT.md` summarizes
the run. Group means with different coverage must not be treated as paired gains;
use the per-incident comparisons for that interpretation.

`--simulate-feedback` runs a separate scripted pilot instead of reading human
feedback. It confirms observed endpoint changes for T2/T3 and assigns equal
proportions to observed version groups for T4. These are explicit simulation
assumptions, not expert judgments, traffic measurements or label-derived fixes.
Simulation cannot use `--workbench-dir` and never writes operator sessions.

The shipped pilot in `artifacts/interactive_route_pilot/` uses three real cases
and a checkpoint trained with the existing two-train/one-validation smoke split.
It validates the workflow only: T2/T3 confirmation does not improve an already
represented topology; T4 adds version structure but the pilot showed no ranking
gain. A larger held-out evaluation with real operator feedback remains necessary
for a human-interaction efficacy claim. Model hyperparameters were not tuned.

## Run a bounded replay

From `/home/user/dyrca`:

```bash
python -m pip install -r /home/user/dyrca/requirements-interactive.txt
python -m dyrca_rca.experiments.run_interactive \
  --data-root /home/user/dyrca/data/experiments \
  --out-dir /home/user/dyrca/artifacts/interactive_smoke \
  --dataset-id dyrca-v1 --system boutique \
  --limit 3 --max-rows 2000 --seed 42 --planner rule
```

Without `--checkpoint`, `diagnosis_status` is `graph_only_no_checkpoint` and
`prediction` is null. This mode validates graph correction mechanics; it does
not report RCA accuracy or invent a ranking. With a compatible checkpoint,
replay uses the same model weights for every correction and emits actual scores
for every service candidate. Candidate selection retains the existing
service-level mask; version-level evaluation is not implemented yet.

The runner defaults to five incidents and 50,000 scanned trace rows per incident.
`--limit` bounds incidents; `--max-rows` independently bounds trace work.
Truncated, missing-summary and unknown-timestamp conditions are recorded.
The new path never automatically loads raw traces.

`--as-of` selects the observation cutoff, defaulting to the latest pre/during
snapshot timestamp. `--start-at` restricts the trace query window; it does not
discard earlier topology needed as context. Trace spans must finish by the
cutoff. Only snapshots at or before the cutoff are eligible. Snapshot/span
timestamps approximate availability because ingestion arrival times are not
available. `post.json`, injection times and recovery times are never read by
the observation loader. Metrics/logs are marked `not_loaded_in_v1`, not normal.

The built-in dependency registry is Boutique only. Other systems require
`--system-config /absolute/path/dependencies.json`, with a JSON mapping such as
`{"trainticket": [["service-a", "service-b"]]}` containing the actual reviewed
dependency pairs. An empty list explicitly selects no static service edges.
TrainTicket's absent summaries remain missing until a separate bounded summary
adapter is supplied; do not treat an empty trace view as evidence of health.

## State and outputs

`manifest.jsonl` maps anonymous incident IDs to raw directories and result
paths for the local operator. This mapping is never sent to the planner.
`summary.csv` records seed, planner, incident limit, group, graph version/path
and diagnosis status. Each incident has a branch for observation/model/planner
configuration and immutable revision directories containing:

- `session.json`: the original observation, full event history and execution receipts at this commit;
- `events.jsonl`: the immutable input commands included in this commit;
- `result.json`: the effective plan, graph, evidence pack and optional ranking;
- `graph.pt`: label-free inference tensors.

`current.json` is atomically replaced only after the entire revision is written.
The runner serializes writers with a local filesystem lock. Repeating a command
resumes its matching branch. Changed observations or checkpoints start a separate
branch. Cached LLM results are replayed rather than requested again. Back up
the revision files together; current pointers alone do not contain state.

State semantics use `interaction_state_version: 2`. `audit_seq` counts recorded
commands; `state_revision` counts successful formal commits (`revision` is its
compatibility alias). Proposals, rejections and failed decisions advance only
the audit sequence. `base_revision` always refers to the formal state revision.
The branch-level `audit.json` atomically stores the latest session, including
pending proposals and failed decision receipts. Resume reads it together with
`current.json`; back up both and the immutable revision directories.

The runner writes proposal previews to `preview.json`, tagged with the feedback
ID and base state revision. Preview errors contain an error class. A preview is
a historical hypothetical result: compare its revision and feedback lifecycle
with the latest audit session before using it. It never updates `current.json`.
The Python API exposes `replay(..., mode="preview", feedback_id=...)` without
mutating the input session or writing files.

V2 uses a separate branch/cache namespace. Existing v1 artifacts remain untouched;
the v2 API rejects v1 sessions rather than silently replaying their old proposed
events under new semantics. Start a fresh observation baseline to continue in v2.

## Submit corrections

Read `audit.json` when present, otherwise `session.json`, for the incident's
current revision, feedback IDs, evidence IDs and scope.
Write one JSON object per line to a feedback file. Example (replace placeholders
with values from that session):

```json
{
  "event_id": "review-001",
  "event_kind": "feedback",
  "incident_id": "incident-<anonymous-id>",
  "base_revision": 0,
  "operation": "exclude_evidence",
  "target_id": "trace_service:<evidence-id>",
  "payload": {},
  "reason": "This aggregate is outside the intended investigation scope.",
  "actor": "engineer",
  "scope": {"system_id": "boutique", "as_of": "<session-as-of>"},
  "verification_status": "proposed"
}
```

Pass `--feedback /absolute/path/feedback.jsonl` alongside the same replay
arguments. New `event_kind: "feedback"` commands always start as proposed and
have no formal effect. Submit a separate command with a new `event_id`,
`event_kind: "decision"`, `operation: "accept_feedback"`,
`target_id: "review-001"`, `payload: {}`, and the current `base_revision` to
accept it. Retain the same incident/scope fields and supply actor/reason;
omit `verification_status` on decisions. The operation vocabulary is:

| Operation | Target | Effect |
| --- | --- | --- |
| `exclude_evidence` | Existing evidence ID | Remove its use by every downstream consumer |
| `restore_evidence` | Existing evidence ID | Restore that observation to the effective view |
| `replace_plan` | `graph_plan` | Replace the entire supplemental graph plan |
| `retract_event` | Earlier non-retraction event ID | Remove that event's effect and replay remaining events |
| `accept_feedback` | Proposed feedback ID | Validate/recompute and commit its effect |
| `reject_feedback` | Proposed feedback ID | Mark rejected, retaining the current formal result |
| `retract_feedback` | Accepted feedback ID | Remove its effect after successful recomputation |
| `supersede_feedback` | Accepted feedback ID | Retire its effect after successful recomputation |

Lifecycle is derived from immutable commands and system-owned `outcomes`:
`proposed → accepted/rejected`, then `accepted → retracted/superseded`.
Rejected, retracted and superseded feedback cannot be accepted again; submit a
new proposal. Execution failure records `execution_status: failed`, keeps the
previous lifecycle and formal revision, and may be retried with a new decision
ID. Identical command IDs are idempotent, including failed attempts. Invalid
schema, scope, references, stale revisions or illegal transitions reject the
command before recording it. Execution receipts store only error classes.

Accepted feedback is applied in acceptance order. Accepting a new replacement
plan supersedes the previously active replacement in the same transaction.
Failure preserves the old plan; retracting the new plan does not revive a
superseded plan. Evidence exclusions/restorations replay the remaining accepted
commands in order. `retract_event` is a compatibility alias for
`retract_feedback`, and requires an accepted feedback target.

Topology evidence retains a whole sanitized source snapshot, but feedback can
now address individual entities and fields. Trace evidence is a service/route
aggregate and can be corrected independently. Original records are retained in
session history; only the effective view reaches planning and inference.

### Local corrections

Use the same event envelope, `event_kind: "feedback"`, and acceptance/retraction
decisions described above. The following are operation-specific payloads.
All three operations support preview without modifying the formal result.

| Operation | Target | Local scope |
| --- | --- | --- |
| `patch_topology` | Topology evidence ID | One named entity and one whitelisted field |
| `correct_trace` | Trace evidence ID | One service or route aggregate's counts |
| `edit_plan` | `graph_plan` | Add, remove, or replace one node/edge/weight operation |

For “service a's endpoint list is stale; this pod is still serving”, submit
`patch_topology` with:

```json
{"collection":"services","entity":"a","field":"endpoint_ips","expected":[],"value":["1"]}
```

`entity` matches exactly one row's `name`. Allowed fields are service
`endpoint_ips`; pod `service`, `version`, `status`, `ip`, `node`; deployment
`replicas`, `readyReplicas`, `version`. Counts must be nonnegative integers and
endpoint lists contain unique nonempty strings. Entity identity, snapshot time,
incident scope and labels cannot be edited. `expected: null` matches an absent
or null field. Correction acceptance checks the effective old value; another
accepted correction changing it produces a conflict, even if the decision uses
the current formal revision. Replanning propagates the corrected snapshot to
the graph, including the disappearance of an unjustified blackhole.

For “the service aggregate has one observed span and no errors”, submit
`correct_trace` against that service evidence ID:

```json
{"expected":{"span_count":1,"status_count":1,"error_count":1},"value":{"span_count":1,"status_count":1,"error_count":0}}
```

All three counts are required, with `error_count <= status_count <= span_count`.
For route aggregates, use `{"expected":{"count":1},"value":{"count":0}}`.
A zero route count no longer creates a trace-supported dependency; static
registry edges still exist independently. Service/route identities are fixed.
Missingness masks and anomaly rates are rebuilt from the corrected counts.

For “this inferred blackhole operation is incorrect”, find its stable ID in
`result.json`'s `plan_operations`, then submit `edit_plan`:

```json
{"section":"node_ops","action":"remove","operation_id":"op:<hash-from-result>","value":{}}
```

For a single route-weight change, use:

```json
{"section":"weight_ops","action":"add","operation_id":"","value":{"op":"REWEIGHT_EDGE","source":"a","target":"b","weight":0.2,"attrs":{"route_weight":0.2},"evidence":["<active-evidence-id>"]}}
```

`replace` uses the existing operation ID and a complete replacement operation.
IDs hash the section and full operation content, rather than array positions.
Adds/replacements require explicit active evidence references. Duplicate adds,
missing/ambiguous IDs, invalid graph operations, and excluded references fail
validation. Final plans still pass the existing full graph validation.

Local evidence corrections replay in acceptance order before planning. Local
plan edits then replay in acceptance order over the current full replacement
plan, or the rule/LLM plan when none is active. Full `replace_plan` feedback does
not implicitly erase local edits. If a changed base plan no longer contains a
targeted operation, the transaction fails; retract the dependent edit first.
Likewise, excluding corrected evidence or retracting a correction on which a
later correction depends fails rather than dropping that dependent feedback.
Retract dependent corrections in reverse order to recover the original result.

Effective evidence keeps its source ID and adds `correction_ids` linking to
immutable audit events. This ID identifies the source record, not a fresh hash
of the corrected value. Original session evidence is never changed. Cache
branches now include `granular-feedback-v1`; feature dimensions and checkpoint
compatibility remain those of the missingness protocol above.

This implementation supports structured local corrections. Raw-span editing,
free-text-to-operation translation and candidate-specific ranking judgments
remain outside this interface.

`replace_plan` applies to the **latest eligible observed graph**, not the legacy
pre-change graph. Its payload must match `PLAN_SCHEMA` in
`src/dyrca_rca/interaction/pipeline.py`. It includes the anonymous `experiment_id`,
node/edge/weight operation arrays, active evidence IDs, `state_transfer: {}` and
`causal_role: "unknown"` or `"hypothesis"`. Formal acceptance requires successful
graph validation and replay. For compatibility, inputs without `event_kind`
and with `verification_status: "verified"` submit and accept atomically;
legacy `proposed` inputs remain pending. Failed compatibility submissions raise
without committing. The verification field records an assertion by the local operator;
this CLI is not an authentication or authorization service.

The automatic rule plan adds blackholes only for explicitly empty endpoint lists
and supplements service dependencies supported by eligible parent/child spans.
It does not infer a proven fault cause from a topology change.

For a route proportion correction, use `REWEIGHT_EDGE` with
`attrs.route_weight` in [0,1], plus the required `weight` field. Final GNN weights
are derived using the existing fixed coefficient constants. Direct editing of
the final GNN weight is deliberately not the contract. Other node/edge operators
reuse the existing executor, with warnings treated as failed transactions.
This first version does not enforce a route group's weights summing to one;
complete version-split route validation remains follow-up work.

Every event checks schema, incident, revision, scope and references. Reusing an
event ID with identical content is an idempotent retry; different content is
rejected. References to excluded facts reject a replacement plan instead of
silently reactivating them. Removing all usable topology is rejected. A rejected
event leaves the last committed revision intact. A file containing multiple
events commits valid events sequentially; the file is not one transaction.

## Train once, then reuse the checkpoint

The current feature protocol is `interactive-topology-trace-v2-missingness`.
It uses current topology and bounded trace error rates. Missing trace evidence
is represented by a null anomaly in graph JSON, with a zero tensor placeholder
and an explicit observation mask. An observed zero error rate has mask 1;
missing summaries, empty windows, absent service aggregates, excluded evidence
and aggregates without recognized statuses have mask 0. A zero placeholder
alone must never be interpreted as evidence of health.

The 25 node features include trace anomaly, observation mask, recognized-status
coverage, `log1p` of recognized-status sample count, and a partial-data flag.
The 8 edge features include the target service's anomaly mask, coverage and
partial flag. `final_weight` remains the last column for both model families.
Node masks also reach isolated service candidates without incoming edges.
Recognized OTel UNSET/OK codes count as observed non-errors, ERROR as an error;
missing/unrecognized statuses are excluded from the rate denominator. Truncated
reads and skipped unknown timestamps flag the retained data as partial. These
flags describe available samples, not a guarantee of complete telemetry.

For missing anomalies, omit that term and renormalize the remaining fixed
coefficient sum. Observed rates retain the original weighting formula. This is
an available-evidence weighting rule, not an estimated anomaly or uncertainty
calibration. Metrics/logs remain unloaded and contribute no health evidence.

Training and inference use the same schema; cache branches include the protocol.
V1 interactive checkpoints cannot be reused: train once under the new protocol
with the same fixed model settings, then reuse that checkpoint for corrections.
Legacy materialization retains its original 20 node and 5 edge features.
It is different from the legacy metadata-guided, multimodal feature pipeline.
Legacy checkpoints must not be relabeled as compatible.

Prepare an explicit JSON split with disjoint `train`, `val`, `test` lists of
experiment directory names. Nonempty train and validation sets are required;
there is no fallback to training on test data. Ensure `--limit` includes all
selected train/validation directories. Example command:

```bash
python -m dyrca_rca.experiments.run_interactive \
  --data-root /home/user/dyrca/data/experiments \
  --out-dir /home/user/dyrca/artifacts/interactive_training \
  --dataset-id dyrca-v1 --system boutique --limit 30 \
  --max-rows 50000 --seed 42 --planner rule \
  --train-split /absolute/path/split.json
```

Training keeps the repository's fixed GNN configuration. It reads root labels
only after creating label-free observation tensors, and writes `model.pt` plus
`training.json`. Replay adds
`--checkpoint /home/user/dyrca/artifacts/interactive_training/model.pt`.
Checkpoint loading verifies model configuration, feature names and protocol.
No checkpoint training occurs during correction replay. The runner does not
compute test accuracy; use held-out incidents and an evaluator with isolated
labels when adding the experiment comparison stage.

## LLM and legacy behavior

`--planner llm` uses the existing environment-configured client. Its only input
is the effective observation and the operation schema. Responses must pass JSON
Schema, reference and graph-execution validation. Missing clients or invalid
responses use deterministic rules on that same effective view. Fallback records
the exception class, not arbitrary provider error text.

Legacy evidence packs no longer export `original_label_node_id`; prompt creation
also strips labels from old caches. Explicitly disabling target prior now removes
it from the prompt as well. The legacy pipeline still uses injection metadata
and retains its prior-enabled compatibility defaults: these limited fixes do not
make old runs equivalent to the new observation protocol. Legacy prediction CSVs
generated after this change include real per-candidate scores; old CSVs retain
their historical limitations.

## Validation

```bash
python -m pytest /home/user/dyrca/tests/test_interaction.py -q
python -m pytest /home/user/dyrca/tests -q
```

Tests cover label/future-data isolation, bounded reads, missing versus empty
endpoints, exclusion propagation, retraction, retry, stale events, atomic graph
rejection, route weights reaching tensors and frozen-model scores, checkpoint
protocol validation, schema-checked LLM fallback and disk resume. Random model
weights used by unit tests are fixtures, not reported RCA results.
