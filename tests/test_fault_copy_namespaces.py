import os
import subprocess
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,namespace", [("common.sh", "boutique-c2"),
                                               ("common_trainticket.sh", "trainticket-c4")])
def test_chaos_verification_uses_active_copy_namespace(script, namespace):
    command = 'source "$1"; kubectl() { CALL="$*"; }; verify_chaos StressChaos fixture; printf "CALL: %s\\n" "$CALL"'
    result = subprocess.run(["bash", "-c", command, "test", str(PROJECT / "faults" / script)],
                            env=dict(os.environ, NAMESPACE=namespace), capture_output=True, text=True)
    assert result.returncode == 0
    assert f"CALL: get StressChaos fixture -n {namespace}" in result.stdout


def test_copy_counterfactual_keeps_platform_specific_decoy_mapping():
    # Execute only argument/default selection; do not create directories or
    # call a real cluster. The sentinel exits just before experiment creation.
    prefix = (PROJECT / "faults" / "C_decoy_control.sh").read_text().split('EXP_ID="')[0]
    prefix = prefix.replace('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                            'SCRIPT_DIR="$FAULT_DIR"')
    command = prefix + '\nprintf "SELECTED %s %s %s\\n" "$PLATFORM" "$DECOY" "$DECOY2"\n'
    result = subprocess.run(["bash", "-c", command, "test", "ts-travel-service", "scale", "auto", "auto"],
                            env=dict(os.environ, NAMESPACE="trainticket-c3", FAULT_DIR=str(PROJECT / "faults")),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "SELECTED trainticket ts-contacts-service ts-config-service" in result.stdout


@pytest.mark.parametrize("script", ["common.sh", "common_trainticket.sh"])
def test_replica_recovery_requires_ready_replicas_and_is_bounded(script):
    base = ['bash', '-c']
    ready = 'source "$1"; kubectl() { printf "1 1"; }; wait_replica_recovery demo 1 1'
    result = subprocess.run(base + [ready, "test", str(PROJECT / "faults" / script)], capture_output=True, text=True)
    assert result.returncode == 0
    not_ready = 'source "$1"; kubectl() { printf "1 0"; }; wait_replica_recovery demo 1 0'
    result = subprocess.run(base + [not_ready, "test", str(PROJECT / "faults" / script)], capture_output=True, text=True)
    assert result.returncode == 1
