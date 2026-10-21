"""Measurement boundaries in the manuscript's descriptive observation study."""
import copy

from dyrca_rca.experiments.analyze_design_observations import characterize, summarize


def view():
    old = {"timestamp": "2026-01-01T00:00:00Z", "observed_collections": ["services", "pods"],
           "services": [{"name": "app", "endpoint_ips": ["ip0"]},
                        {"name": "public", "endpoint_ips": ["ip0"]},
                        {"name": "empty", "endpoint_ips": []}],
           "pods": [{"name": "pod0", "service": "app", "ip": "ip0", "version": "v1"}]}
    new = copy.deepcopy(old)
    new["timestamp"] = "2026-01-01T00:01:00Z"
    new["pods"][0].update(name="pod1", ip="ip1", version="v2")
    for service in new["services"][:2]:
        service["endpoint_ips"] = ["ip1"]
    return {"incident_id": "example", "as_of": new["timestamp"], "observation_version": "fixture",
            "evidence": [{"id": "before", "kind": "topology", "data": old},
                         {"id": "after", "kind": "topology", "data": new}],
            "modalities": {"traces": {"status": "truncated", "rows_scanned": 2000,
                                       "eligible_rows": 100, "unknown_timestamp_rows": 0}}}


def test_stable_service_names_do_not_hide_endpoint_or_instance_change():
    row = characterize(view())
    assert row["service_set_stable"] is True
    assert row["endpoint_set_changed"] is True
    assert row["pod_name_set_changed"] is True
    assert row["endpoint_pairs_observed"] == 3
    assert row["endpoint_stable_empty_services"] == ["empty"]
    assert row["endpoint_became_empty_services"] == []


def test_version_coverage_requires_populated_groups_and_known_labels():
    observation = view()
    assert characterize(observation)["populated_pod_services"] == ["app"]
    assert characterize(observation)["version_pairs_observed"] == 1
    observation["evidence"][0]["data"]["pods"][0]["version"] = "unknown"
    row = characterize(observation)
    assert row["populated_pod_service_pairs"] == 1
    assert row["version_pairs_observed"] == 0
    assert summarize([row])["metrics"]["version_pair_coverage"]["denominator"] == 1


def test_existing_cross_name_relation_survives_pod_replacement_without_becoming_new():
    observation = view()
    row = characterize(observation)
    assert row["cross_names_before"] == [("public", "app")]
    assert row["has_cross_names_after"] is True
    assert row["has_new_cross_names"] is False
    observation["evidence"][1]["data"]["services"][2]["endpoint_ips"] = ["ip1"]
    assert characterize(observation)["new_cross_names"] == [("empty", "app")]


def test_unknown_endpoints_and_non_distinct_times_do_not_become_zero_change():
    observation = view()
    for service in observation["evidence"][0]["data"]["services"]:
        service.pop("endpoint_ips")
    row = characterize(observation)
    assert row["endpoint_set_changed"] is None
    assert row["has_new_cross_names"] is None
    assert summarize([row])["metrics"]["endpoint_set_changed"]["denominator"] == 0
    observation = view()
    observation["evidence"][1]["data"]["services"][0]["endpoint_ips"] = []
    assert characterize(observation)["endpoint_became_empty_services"] == ["app"]
    observation = view()
    observation["evidence"][0]["data"]["timestamp"] = observation["as_of"]
    row = characterize(observation)
    assert row["service_set_stable"] is None
    assert row["version_pairs_observed"] == row["populated_pod_service_pairs"] == 0
