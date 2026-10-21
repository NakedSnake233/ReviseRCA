import csv

from dyrca_rca.experiments.analyze_shortcut_inputs import trace_coverage, metric_coverage


def test_trace_coverage_respects_completion_cutoff_and_distinguishes_missing_status(tmp_path):
    path = tmp_path / "trace.csv"
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["service", "start_time_unix_nano", "duration_nano", "status_code"])
        writer.writerows([["root", 1000000000, 1000000000, ""], ["root", 2000000000, 1000000000, "OK"],
                          ["root", 3000000000, 2000000000, "ERROR"], ["root", 1000000000, -1, "ERROR"]])
    r = trace_coverage(path, "root", 2., 3.)
    assert r["root_spans"] == 2
    assert r["root_between_snapshots"] == 1
    assert r["root_status_known"] == 1
    assert r["before_pre"] == 1
    assert trace_coverage(path, "root", 2., 3., max_rows=1)["truncated"]


def test_global_latency_cannot_be_attributed_to_the_labelled_root(tmp_path):
    path = tmp_path / "metric.csv"
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "labels_json", "value"])
        writer.writerows([["1970-01-01T00:00:02Z", "{}", 1000],
                          ["1970-01-01T00:00:02Z", '{"destination_workload":"unknown"}', 900],
                          ["1970-01-01T00:00:02Z", '{"destination_workload":"root"}', 10],
                          ["1970-01-01T00:00:04Z", '{"destination_workload":"root"}', 20]])
    r = metric_coverage(path, "root", 1., 3.)
    assert r["eligible"] == 3
    assert r["service_attributed"] == r["root_between_snapshots"] == 1
