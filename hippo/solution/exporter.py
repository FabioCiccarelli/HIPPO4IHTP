"""Solution exporter — builds IHTC-compatible JSON from solved Gurobi variables.

Supports both the *light* (Phase 1 only) and *full* (Phase 1 + Phase 2) output
formats required by the competition validator.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from hippo.instance import Instance

logger = logging.getLogger(__name__)

SHIFT_MAP = {0: "early", 1: "late", 2: "night"}


# ===================================================================== #
#  Patient section                                                      #
# ===================================================================== #

def _build_patients(
    instance: Instance,
    phase1_vars: dict[str, Any],
) -> list[dict]:
    """Build the ``patients`` list for the IHTC JSON output."""

    alpha = phase1_vars["alpha"]
    gamma = phase1_vars["gamma"]
    offset = len(instance.PA)
    n_incoming = len(instance.PM) + len(instance.PO)
    zfill_p = len(str(n_incoming - 1))
    zfill_r = len(str(len(instance.R) - 1))

    # Pre-compute admission day + room for each patient
    admission: dict[int, int] = {}
    room_of: dict[int, int] = {}
    ot_of: dict[int, int] = {}
    incoming_patients = set(instance.incoming_patients)

    for (p, r, d), var in alpha.items():
        if p in incoming_patients and var.X > 0.5:
            admission[p] = d
            room_of[p] = r
    for (p, o, d), var in gamma.items():
        if p in incoming_patients and var.X > 0.5:
            ot_of[p] = o

    patients: list[dict] = []
    for p in instance.incoming_patients:
        pid = f"p{str(p - offset).zfill(zfill_p)}"
        if p not in admission:
            patients.append({"id": pid, "admission_day": "none"})
        else:
            patients.append({
                "id": pid,
                "admission_day": admission[p],
                "room": f"r{str(room_of[p]).zfill(zfill_r)}",
                "operating_theater": f"t{ot_of[p]}",
            })

    return patients


# ===================================================================== #
#  Nurse section                                                        #
# ===================================================================== #

def _build_nurses(
    instance: Instance,
    z_vars: dict[tuple, Any],
) -> list[dict]:
    """Build the ``nurses`` list for the IHTC JSON output."""

    n_incoming = len(instance.PM) + len(instance.PO)
    zfill_n = len(str(n_incoming - 1))
    zfill_r = len(str(len(instance.R) - 1))

    # Collect non-zero z assignments
    nurse_schedule: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for (n, r, s), var in z_vars.items():
        if var.X > 0.5:
            day, shift_idx = divmod(s, 3)
            nurse_schedule[n].append((day, SHIFT_MAP[shift_idx], r))

    nurses: list[dict] = []
    for n in sorted(nurse_schedule):
        grouped: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for day, shift, room in nurse_schedule[n]:
            grouped[day][shift].add(f"r{str(room).zfill(zfill_r)}")

        assignments = [
            {"day": d, "shift": sh, "rooms": sorted(rooms)}
            for d in sorted(grouped)
            for sh, rooms in sorted(grouped[d].items())
        ]
        nurses.append({"id": f"n{str(n).zfill(zfill_n)}", "assignments": assignments})

    return nurses


# ===================================================================== #
#  Public API                                                           #
# ===================================================================== #

def export_phase1_solution(
    instance: Instance,
    phase1_vars: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export Phase-1 (light) solution to JSON."""

    patients = _build_patients(instance, phase1_vars)
    solution = {"patients": patients, "nurses": [], "costs": []}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(solution, fh, indent=4)
    logger.info("Phase-1 solution exported to %s", output_path)


def export_full_solution(
    instance: Instance,
    phase1_vars: dict[str, Any],
    phase2_vars: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export the combined Phase-1 + Phase-2 (full) solution to JSON."""

    patients = _build_patients(instance, phase1_vars)
    nurses = _build_nurses(instance, phase2_vars["z"])
    solution = {"patients": patients, "nurses": nurses, "costs": []}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(solution, fh, indent=4)
    logger.info("Full solution exported to %s", output_path)
