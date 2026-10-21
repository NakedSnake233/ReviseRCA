"""Fair, dynamic-evidence baseline adapters for the temporal RCA release."""

from .evidence import BASELINE_PROTOCOL, ReleaseEvidence, load_release_evidence

__all__ = ["BASELINE_PROTOCOL", "ReleaseEvidence", "load_release_evidence"]
