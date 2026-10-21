"""Host memory admission policy for isolated data-collection copies."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

GIB = 1024 ** 3


def resume_exclusions(path):
    if not path or not Path(path).exists():
        return set()
    records = json.loads(Path(path).read_text()).get("excluded_output_dirs")
    if not isinstance(records, list) or not all(isinstance(item, str) and Path(item).is_absolute() for item in records):
        raise ValueError("Invalid resume-exclusion manifest; refusing to reuse old exports")
    return {str(Path(item).resolve()) for item in records}


def read_memory_state(path=Path("/proc/meminfo")):
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ("MemTotal:", "MemAvailable:"):
            values[fields[0][:-1]] = int(fields[1]) * 1024
    if not all(key in values for key in ("MemTotal", "MemAvailable")):
        raise RuntimeError("Cannot read host MemAvailable; collection is not admitted")
    return {"total_gib": values["MemTotal"] / GIB,
            "available_gib": values["MemAvailable"] / GIB}


@dataclass(frozen=True)
class MemoryPolicy:
    admit_gib: float = 10.0
    reserve_gib: float = 10.0
    emergency_gib: float = 10.0
    # Host-only startup admission; dedicated cgroup hard limits remain in force.
    host_only: bool = True

    def __post_init__(self):
        if not all(math.isfinite(value) for value in
                   (self.admit_gib, self.reserve_gib, self.emergency_gib)) or not (
                0 < self.emergency_gib <= self.reserve_gib <= self.admit_gib):
            raise ValueError("Require finite 0 < emergency <= reserve <= admission threshold")

    def state(self, available_gib):
        if available_gib < self.emergency_gib:
            return "emergency"
        if available_gib < self.reserve_gib:
            return "drain"
        if available_gib < self.admit_gib:
            return "hold"
        return "admit"
