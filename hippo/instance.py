"""Parsing e struttura dati per le istanze IHTC 2024."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Instance:
    """Istanza IHTC completamente parsata.

    Insiemi come list[int], parametri come dict indicizzati per gli indici rilevanti.
    """

    # insiemi di pazienti
    P: list[int] = field(default_factory=list)       # all patients
    PA: list[int] = field(default_factory=list)      # occupants (already admitted)
    PM: list[int] = field(default_factory=list)      # mandatory incoming patients
    PO: list[int] = field(default_factory=list)      # optional incoming patients
    Pfemale: list[int] = field(default_factory=list)
    Pmale: list[int] = field(default_factory=list)

    R: list[int] = field(default_factory=list)       # rooms
    D: list[int] = field(default_factory=list)       # days
    S: list[int] = field(default_factory=list)       # shifts
    ST: list[int] = field(default_factory=list)      # shift types
    Searly: list[int] = field(default_factory=list)
    Slate: list[int] = field(default_factory=list)
    Snight: list[int] = field(default_factory=list)

    N: list[int] = field(default_factory=list)       # nurses
    L: list[int] = field(default_factory=list)       # skill levels
    C: list[int] = field(default_factory=list)       # surgeons
    OT: list[int] = field(default_factory=list)      # operating theatres
    A: list[int] = field(default_factory=list)       # age groups

    # Extended sets (include sentinel value for boundary conditions)
    D_ext: list[int] = field(default_factory=list)
    S_ext: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Objective weights                                                  #
    # ------------------------------------------------------------------ #
    weights: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Patient parameters                                                 #
    # ------------------------------------------------------------------ #
    surgery_release_day: dict[int, int] = field(default_factory=dict)
    surgery_due_day: dict[int, int] = field(default_factory=dict)
    surgery_release_shift: dict[int, int] = field(default_factory=dict)
    surgery_due_shift: dict[int, int] = field(default_factory=dict)
    age_group: dict[int, int] = field(default_factory=dict)
    length_of_stay: dict[int, int] = field(default_factory=dict)
    workload_produced: dict[int, list[float]] = field(default_factory=dict)
    skill_level_required: dict[int, list[int]] = field(default_factory=dict)
    room_compatible: dict[int, dict[int, int]] = field(default_factory=dict)
    surgery_duration: dict[int, int] = field(default_factory=dict)
    surgeon_assignment: dict[int, dict[int, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Room parameters                                                    #
    # ------------------------------------------------------------------ #
    room_capacity: list[int] = field(default_factory=list)
    room_occupant: dict[int, dict[int, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Nurse parameters                                                   #
    skill_level_nurse: dict[int, int] = field(default_factory=dict)
    availability: dict[int, dict[int, int]] = field(default_factory=dict)
    max_load: dict[int, dict[int, float]] = field(default_factory=dict)

    max_surgery_time: dict[int, dict[int, int]] = field(default_factory=dict)
    max_ot_availability: dict[int, dict[int, int]] = field(default_factory=dict)

    # parametri aggregati per il proxy (media workload/skill per paziente)
    skill_avg: dict[int, list[float]] = field(default_factory=dict)
    wl_avg: dict[int, list[float]] = field(default_factory=dict)
    wl_daily: dict[int, float] = field(default_factory=dict)
    skill_daily: dict[int, float] = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return len(self.D)

    @property
    def n_shifts(self) -> int:
        return len(self.S)

    @property
    def n_rooms(self) -> int:
        return len(self.R)

    @property
    def max_age_group(self) -> int:
        return max(self.A) if self.A else 0

    @property
    def incoming_patients(self) -> list[int]:
        return self.PM + self.PO


SHIFT_OFFSET = {"early": 0, "late": 1, "night": 2}
WEIGHT_KEY_MAP = {
    "room_mixed_age": "W1",
    "room_nurse_skill": "W2",
    "continuity_of_care": "W3",
    "nurse_eccessive_workload": "W4",
    "open_operating_theater": "W5",
    "surgeon_transfer": "W6",
    "patient_delay": "W7",
    "unscheduled_optional": "W8",
}


def load_instance(path: str | Path) -> Instance:
    """Carica e parsa un'istanza IHTC da file JSON."""

    path = Path(path)
    with open(path) as fh:
        raw: dict[str, Any] = json.load(fh)

    inst = Instance()

    # --- insiemi di indici ---
    n_occupants = len(raw["occupants"])
    n_patients = len(raw["patients"])

    inst.PA = list(range(n_occupants))
    raw_pm = [i for i, p in enumerate(raw["patients"]) if p["mandatory"]]
    raw_po = [i for i, p in enumerate(raw["patients"]) if not p["mandatory"]]
    inst.PM = [n_occupants + i for i in raw_pm]
    inst.PO = [n_occupants + i for i in raw_po]
    inst.P = inst.PA + inst.PM + inst.PO

    for p in inst.P:
        gender = (
            raw["occupants"][p]["gender"] if p < n_occupants
            else raw["patients"][p - n_occupants]["gender"]
        )
        if gender == "A":
            inst.Pfemale.append(p)
        else:
            inst.Pmale.append(p)

    n_days = raw["days"]
    inst.R = list(range(len(raw["rooms"])))
    inst.D = list(range(n_days))
    inst.S = list(range(3 * n_days))
    inst.ST = list(range(len(raw["shift_types"])))
    inst.Searly = [3 * d for d in inst.D]
    inst.Slate = [3 * d + 1 for d in inst.D]
    inst.Snight = [3 * d + 2 for d in inst.D]

    inst.N = list(range(len(raw["nurses"])))
    inst.L = list(range(raw["skill_levels"]))
    inst.C = list(range(len(raw["surgeons"])))
    inst.OT = list(range(len(raw["operating_theaters"])))
    inst.A = list(range(len(raw["age_groups"])))

    inst.D_ext = inst.D + [n_days]
    inst.S_ext = inst.S + [3 * n_days]

    # --- pesi dell'obiettivo ---
    inst.weights = {
        WEIGHT_KEY_MAP[k]: v for k, v in raw["weights"].items()
    }

    # --- stanze ---
    room_id_map = {room["id"]: idx for idx, room in enumerate(raw["rooms"])}
    inst.room_capacity = [room["capacity"] for room in raw["rooms"]]

    # ----- Patient parameters ----------------------------------------- #
    age_group_names = raw["age_groups"]

    for p in inst.P:
        if p < n_occupants:
            pat = raw["occupants"][p]
        else:
            pat = raw["patients"][p - n_occupants]
            inst.surgery_release_day[p] = pat["surgery_release_day"]
            inst.surgery_due_day[p] = (
                pat["surgery_due_day"] if pat["mandatory"] else n_days - 1
            )
            inst.surgery_release_shift[p] = 3 * pat["surgery_release_day"]
            inst.surgery_due_shift[p] = (
                3 * pat["surgery_due_day"] if pat["mandatory"] else 3 * n_days - 1
            )
            inst.surgery_duration[p] = pat["surgery_duration"]
            surgeon_idx = int(pat["surgeon_id"][1:])
            inst.surgeon_assignment[p] = {
                c: int(c == surgeon_idx) for c in inst.C
            }

        inst.age_group[p] = age_group_names.index(pat["age_group"])
        inst.length_of_stay[p] = pat["length_of_stay"]
        inst.workload_produced[p] = pat["workload_produced"]
        inst.skill_level_required[p] = pat["skill_level_required"]
        inst.room_compatible[p] = {r: 1 for r in inst.R}
        for r_id in pat.get("incompatible_room_ids", []):
            inst.room_compatible[p][room_id_map[r_id]] = 0

    # ----- Occupant room mapping -------------------------------------- #
    for p_id, occ in enumerate(raw["occupants"]):
        room_idx = room_id_map[occ["room_id"]]
        inst.room_occupant[p_id] = {r: int(r == room_idx) for r in inst.R}

    # ----- Nurse parameters ------------------------------------------- #
    inst.availability = {n: {s: 0 for s in inst.S} for n in inst.N}
    inst.max_load = {n: {s: 0 for s in inst.S} for n in inst.N}

    for n_id, nurse in enumerate(raw["nurses"]):
        inst.skill_level_nurse[n_id] = nurse["skill_level"]
        for shift in nurse["working_shifts"]:
            s = 3 * shift["day"] + SHIFT_OFFSET[shift["shift"].strip().lower()]
            inst.availability[n_id][s] = 1
            inst.max_load[n_id][s] = shift["max_load"]

    # ----- Surgeon parameters ----------------------------------------- #
    inst.max_surgery_time = {
        c: {d: raw["surgeons"][c]["max_surgery_time"][d] for d in inst.D}
        for c in inst.C
    }

    # ----- OT parameters ---------------------------------------------- #
    inst.max_ot_availability = {
        o: {d: raw["operating_theaters"][o]["availability"][d] for d in inst.D}
        for o in inst.OT
    }

    # ----- Proxy / averaged parameters -------------------------------- #
    _compute_proxy_parameters(inst)

    return inst


def _compute_proxy_parameters(inst: Instance) -> None:
    """Compute shift-type-averaged skill and workload per patient."""

    n_shifts = len(inst.S)
    n_days = n_shifts // 3

    for p in inst.P:
        skill_vec = inst.skill_level_required[p]
        wl_vec = inst.workload_produced[p]

        early_skill = [skill_vec[i] for i in range(0, len(skill_vec), 3)]
        late_skill = [skill_vec[i] for i in range(1, len(skill_vec), 3)]
        night_skill = [skill_vec[i] for i in range(2, len(skill_vec), 3)]

        skill_pattern = [
            math.ceil(sum(early_skill) / len(early_skill)),
            math.ceil(sum(late_skill) / len(late_skill)),
            math.ceil(sum(night_skill) / len(night_skill)),
        ]
        inst.skill_avg[p] = skill_pattern * n_days

        early_wl = [wl_vec[i] for i in range(0, len(wl_vec), 3)]
        late_wl = [wl_vec[i] for i in range(1, len(wl_vec), 3)]
        night_wl = [wl_vec[i] for i in range(2, len(wl_vec), 3)]

        wl_pattern = [
            math.ceil(sum(early_wl) / len(early_wl)),
            math.ceil(sum(late_wl) / len(late_wl)),
            math.ceil(sum(night_wl) / len(night_wl)),
        ]
        inst.wl_avg[p] = wl_pattern * n_days

        # Scalar daily averages (sum of shift-type averages)
        inst.skill_daily[p] = float(sum(skill_pattern))
        inst.wl_daily[p] = float(sum(wl_pattern))
