"""Deterministic before/after explanations; no causal or label-based claims."""
from dyrca_rca.interaction.observation import canonical


def compare_results(before, after):
    def changes(left, right, key):
        a, b = {key(x): x for x in left}, {key(x): x for x in right}
        return {"added": [b[k] for k in sorted(b.keys() - a.keys())],
                "removed": [a[k] for k in sorted(a.keys() - b.keys())],
                "changed": [{"id": k, "before": a[k], "after": b[k]}
                            for k in sorted(a.keys() & b.keys()) if canonical(a[k]) != canonical(b[k])]}
    ranks = []
    p, q = before.get("prediction"), after.get("prediction")
    if p is not None and q is not None:
        old = {n: i + 1 for i, n in enumerate(p["ranked_node_ids"])}
        new = {n: i + 1 for i, n in enumerate(q["ranked_node_ids"])}
        ranks = [{"candidate": n, "before_rank": old.get(n), "after_rank": new.get(n),
                  "before_score": p["scores"].get(n), "after_score": q["scores"].get(n)}
                 for n in sorted(old.keys() | new.keys(), key=lambda n: (new.get(n, 10**9), n))]
    return {"nodes": changes(before["graph"]["nodes"], after["graph"]["nodes"], lambda n: n["id"]),
            "edges": changes(before["graph"]["edges"], after["graph"]["edges"],
                             lambda e: canonical([e["source"], e["target"], e["edge_type"]])),
            "evidence": changes(before["evidence_pack"]["evidence"], after["evidence_pack"]["evidence"], lambda e: e["id"]),
            "ranking": ranks, "prediction_available": p is not None and q is not None,
            "interpretation": "结构、证据和固定模型输出的变化；不代表已证明故障因果。"}
