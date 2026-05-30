"""Constructive heuristic + ALNS warm start for HIPPO Phase 1.

The heuristic mirrors the Phase-1 MILP objective, including the configured
proxy strategy, and produces a feasible admission/room/surgery incumbent that
can be injected into the Gurobi model as a MIP start.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

from hippo.config import ProxyStrategy
from hippo.instance import Instance

logger = logging.getLogger(__name__)


@dataclass
class PatientAssignment:
    admitted: bool = False
    day: Optional[int] = None
    room: Optional[int] = None
    ot: Optional[int] = None


@dataclass
class Phase1HeuristicConfig:
    time_limit: float = 60.0
    seed: int = 1234
    constructive_runs: int = 8
    repair_seconds: float = 8.0
    proxy_strategy: ProxyStrategy = ProxyStrategy.NONE
    proxy_weight: float = 0.0
    destroy_weights: tuple[float, float, float, float] = (0.22, 0.30, 0.26, 0.22)
    repair_weights: tuple[float, float, float] = (0.50, 0.30, 0.20)
    family_weights: tuple[float, float] = (0.55, 0.45)
    micro_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    small_frac_range: tuple[float, float] = (0.02, 0.03)
    medium_frac_range: tuple[float, float] = (0.04, 0.05)
    large_frac_range: tuple[float, float] = (0.06, 0.08)
    destroy_size_probs: tuple[float, float, float] = (0.70, 0.25, 0.05)
    destroy_abs_min: int = 3
    destroy_abs_max: int = 18
    repair_top_k: int = 3
    repair_top_k_large_instance: int = 5
    large_instance_threshold: int = 120
    greedy_pick_random_prob: float = 0.34
    regret_pick_random_prob: float = 0.24
    proxy_pick_random_prob: float = 0.26
    accept_worse: bool = True
    sa_initial_temperature: float = 0.35
    sa_final_temperature: float = 0.05
    equal_move_accept_prob: float = 0.60
    reheat_after_no_improve: int = 150
    reheat_multiplier: float = 1.8
    max_reheated_temperature: float = 0.80
    sa_scale_fraction: float = 0.01
    sa_scale_floor: float = 100.0
    scaled_delta_small_threshold: float = 0.10
    scaled_delta_hard_reject: float = 2.50
    reset_to_best_after_no_improve: int = 300
    relative_drift_limit: float = 0.08


class Phase1HeuristicSolution:
    def __init__(self, instance: Instance, config: Phase1HeuristicConfig, *, randomized: bool = True, random_seed: int | None = None) -> None:
        self.instance = instance
        self.config = config
        self.randomized = randomized
        self.random = random.Random(config.seed if random_seed is None else random_seed)
        self.alns_random = random.Random((config.seed if random_seed is None else random_seed) + 40007)

        self.assignment: dict[int, PatientAssignment] = {
            p: PatientAssignment() for p in instance.P
        }
        self.room_patients: dict[int, dict[int, list[int]]] = {
            r: {d: [] for d in instance.D} for r in instance.R
        }
        self.room_occupancy: dict[int, dict[int, int]] = {
            r: {d: 0 for d in instance.D} for r in instance.R
        }
        self.room_gender: dict[int, dict[int, Optional[str]]] = {
            r: {d: None for d in instance.D} for r in instance.R
        }
        self.room_age_min: dict[int, dict[int, Optional[int]]] = {
            r: {d: None for d in instance.D} for r in instance.R
        }
        self.room_age_max: dict[int, dict[int, Optional[int]]] = {
            r: {d: None for d in instance.D} for r in instance.R
        }
        self.room_wl_total: dict[int, dict[int, float]] = {
            r: {d: 0.0 for d in instance.D} for r in instance.R
        }
        self.room_skill_total: dict[int, dict[int, float]] = {
            r: {d: 0.0 for d in instance.D} for r in instance.R
        }
        self.occupied_rooms_day: dict[int, int] = {d: 0 for d in instance.D}
        self.ot_minutes: dict[int, dict[int, int]] = {
            o: {d: 0 for d in instance.D} for o in instance.OT
        }
        self.surgeon_minutes: dict[int, dict[int, int]] = {
            c: {d: 0 for d in instance.D} for c in instance.C
        }
        self.unassigned_mandatory: set[int] = set()
        self.unassigned_optional: set[int] = set()

    def build(self) -> bool:
        self._fix_already_admitted_patients()
        self._insert_mandatory_patients()
        self._insert_optional_patients()
        return self.summary()["mandatory_unassigned"] == 0

    def objective_value(self) -> float:
        return float(self.compute_objective()["total_with_proxy"])

    def admitted_incoming(self) -> list[int]:
        return [p for p in self.instance.incoming_patients if self.assignment[p].admitted]

    def _patient_gender(self, p: int) -> str:
        if p in self.instance.Pfemale:
            return "F"
        if p in self.instance.Pmale:
            return "M"
        raise ValueError(f"Cannot determine gender for patient {p}")

    def _patient_surgeon(self, p: int) -> int:
        for c, value in self.instance.surgeon_assignment.get(p, {}).items():
            if value == 1:
                return c
        raise ValueError(f"No assigned surgeon found for patient {p}")

    def _stay_days(self, p: int, admission_day: int) -> list[int]:
        los = self.instance.length_of_stay[p]
        return list(range(admission_day, admission_day + los))

    def _num_compatible_rooms(self, p: int) -> int:
        return sum(self.instance.room_compatible[p][r] for r in self.instance.R)

    def _count_room_feasible_days(self, p: int) -> int:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        cnt = 0
        for day in inst.D:
            if not (release <= day <= due):
                continue
            if any(self._can_assign_room(p, room, day) for room in inst.R):
                cnt += 1
        return cnt

    def _count_ot_feasible_days(self, p: int) -> int:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        cnt = 0
        for day in inst.D:
            if not (release <= day <= due):
                continue
            if any(self._can_assign_surgery(p, ot, day) for ot in inst.OT):
                cnt += 1
        return cnt

    def _count_joint_feasible_days(self, p: int) -> int:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        cnt = 0
        for day in inst.D:
            if not (release <= day <= due):
                continue
            room_ok = any(self._can_assign_room(p, room, day) for room in inst.R)
            if not room_ok:
                continue
            ot_ok = any(self._can_assign_surgery(p, ot, day) for ot in inst.OT)
            if ot_ok:
                cnt += 1
        return cnt

    def _count_room_feasible_starts(self, p: int) -> int:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        cnt = 0
        for day in inst.D:
            if release <= day <= due and any(self._can_assign_room(p, room, day) for room in inst.R):
                cnt += 1
        return cnt

    def _count_joint_feasible_triples(self, p: int) -> int:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        total = 0
        for day in inst.D:
            if not (release <= day <= due):
                continue
            n_rooms = sum(1 for room in inst.R if self._can_assign_room(p, room, day))
            if n_rooms == 0:
                continue
            n_ots = sum(1 for ot in inst.OT if self._can_assign_surgery(p, ot, day))
            if n_ots == 0:
                continue
            total += n_rooms * n_ots
        return total

    def _residual_criticality_key(self, p: int) -> tuple:
        inst = self.instance
        window = inst.surgery_due_day[p] - inst.surgery_release_day[p] + 1
        los = inst.length_of_stay[p]
        dur = inst.surgery_duration[p]
        incompatible = sum(1 for r in inst.R if inst.room_compatible[p][r] == 0)
        joint_days = self._count_joint_feasible_days(p)
        room_starts = self._count_room_feasible_starts(p)
        triple_count = self._count_joint_feasible_triples(p)
        return (triple_count, joint_days, room_starts, -los, window, -incompatible, -dur, p)

    def _reprioritize_remaining_mandatory(self, remaining: set[int], head_size: int = 12) -> list[int]:
        ordered = sorted(remaining, key=self._dynamic_mandatory_key)
        if len(ordered) <= 1:
            return ordered
        head_n = min(head_size, len(ordered))
        head = ordered[:head_n]
        tail = ordered[head_n:]
        head.sort(key=self._residual_criticality_key)
        if self.randomized and len(head) > 2:
            strict = [p for p in head if self._count_joint_feasible_triples(p) <= 2]
            others = [p for p in head if self._count_joint_feasible_triples(p) > 2]
            if len(strict) > 1:
                frozen = strict[: min(2, len(strict))]
                mix = strict[min(2, len(strict)):]
                self.random.shuffle(mix)
                strict = frozen + mix
            head = strict + others
        return head + tail

    def _difficulty_key(self, p: int) -> tuple:
        inst = self.instance
        mandatory_flag = 0 if p in inst.PM else 1
        window = inst.surgery_due_day[p] - inst.surgery_release_day[p] + 1
        n_rooms = self._num_compatible_rooms(p)
        los = inst.length_of_stay[p]
        dur = inst.surgery_duration[p]
        surgeon = self._patient_surgeon(p)
        joint_days = self._count_joint_feasible_days(p)
        ot_days = self._count_ot_feasible_days(p)
        room_days = self._count_room_feasible_days(p)
        total_surgeon_cap = sum(inst.max_surgery_time[surgeon][d] for d in inst.D)
        return (mandatory_flag, joint_days, ot_days, room_days, n_rooms, -dur, -los, window, total_surgeon_cap, p)

    def _dynamic_mandatory_key(self, p: int) -> tuple:
        inst = self.instance
        window = inst.surgery_due_day[p] - inst.surgery_release_day[p] + 1
        n_rooms = self._num_compatible_rooms(p)
        los = inst.length_of_stay[p]
        dur = inst.surgery_duration[p]
        surgeon = self._patient_surgeon(p)
        joint_days = self._count_joint_feasible_days(p)
        ot_days = self._count_ot_feasible_days(p)
        room_days = self._count_room_feasible_days(p)
        residual_surgeon_cap = sum(
            max(0, inst.max_surgery_time[surgeon][d] - self.surgeon_minutes[surgeon][d])
            for d in inst.D
            if inst.surgery_release_day[p] <= d <= inst.surgery_due_day[p]
        )
        return (joint_days, ot_days, room_days, n_rooms, -dur, -los, window, residual_surgeon_cap, p)

    def _repair_mandatory_key(self, p: int) -> tuple:
        dynamic = self._dynamic_mandatory_key(p)
        candidate_days = self._candidate_days(p, mandatory_phase=True, full_window=True)[:3]
        best_local_product = 10**9
        best_local_rooms = 10**9
        best_local_ots = 10**9
        for day in candidate_days:
            n_rooms = len(self._sorted_feasible_rooms(p, day))
            n_ots = len(self._sorted_feasible_ots(p, day))
            if n_rooms == 0 or n_ots == 0:
                continue
            product = n_rooms * n_ots
            if product < best_local_product:
                best_local_product = product
                best_local_rooms = n_rooms
                best_local_ots = n_ots
        if best_local_product == 10**9:
            best_local_product = 10**9 - 1
            best_local_rooms = 10**9 - 1
            best_local_ots = 10**9 - 1
        return (dynamic[0], best_local_product, best_local_rooms, best_local_ots, *dynamic[1:])

    def _fix_already_admitted_patients(self) -> None:
        inst = self.instance
        for p in inst.PA:
            fixed_room = next(r for r, value in inst.room_occupant[p].items() if value == 1)
            self._assign_occupant(p, 0, fixed_room)

    def _insert_mandatory_patients(self) -> None:
        inst = self.instance
        remaining = set(inst.PM)
        missed_fast: list[int] = []
        rigid = [p for p in inst.PM if self._is_super_critical(p)]
        rigid.sort(key=self._super_critical_key)
        for p in rigid:
            if p not in remaining:
                continue
            best = self._find_assignment_bpp(p, mandatory_phase=True, full_window=True, fast_mode=False)
            remaining.remove(p)
            if best is None:
                missed_fast.append(p)
            else:
                self._assign_patient(p, *best)
        refresh_every = 5
        head_size = 12
        steps_since_refresh = refresh_every
        ordered: list[int] = []
        while remaining:
            if steps_since_refresh >= refresh_every or not ordered:
                ordered = self._reprioritize_remaining_mandatory(remaining, head_size=head_size)
                steps_since_refresh = 0
            p = ordered.pop(0)
            if p not in remaining:
                continue
            best = self._find_assignment_bpp(p, mandatory_phase=True, full_window=True, fast_mode=True)
            remaining.remove(p)
            steps_since_refresh += 1
            if best is None:
                missed_fast.append(p)
            else:
                self._assign_patient(p, *best)
                if remaining:
                    sample = sorted(remaining, key=self._dynamic_mandatory_key)[: min(6, len(remaining))]
                    if sample and min(self._count_joint_feasible_triples(q) for q in sample) <= 2:
                        steps_since_refresh = refresh_every
        retry_set = set(missed_fast)
        still_missed: list[int] = []
        while retry_set:
            ordered = sorted(retry_set, key=lambda p: (self._residual_criticality_key(p), self._dynamic_mandatory_key(p)))
            p = ordered[0]
            retry_set.remove(p)
            if self.assignment[p].admitted:
                continue
            best = self._find_assignment_bpp(p, mandatory_phase=True, full_window=True, fast_mode=False)
            if best is None:
                still_missed.append(p)
            else:
                self._assign_patient(p, *best)
        still_missed.sort(key=lambda p: (self._residual_criticality_key(p), self._dynamic_mandatory_key(p)))
        for p in still_missed:
            if self.assignment[p].admitted:
                continue
            if not self._targeted_rescue_for_mandatory(p):
                self.unassigned_mandatory.add(p)

    def _insert_optional_patients(self) -> None:
        inst = self.instance
        optional = sorted(inst.PO, key=self._difficulty_key)
        if self.randomized:
            optional = self._shuffle_in_blocks(optional, block_size=8)
        missed: list[int] = []
        for p in optional:
            best = self._find_assignment_bpp(p, mandatory_phase=False, full_window=False, fast_mode=True)
            if best is None:
                missed.append(p)
            else:
                self._assign_patient(p, *best)
        if self.randomized and len(missed) > 1:
            self.random.shuffle(missed)
        for p in missed:
            if self.assignment[p].admitted:
                continue
            best = self._find_assignment_bpp(p, mandatory_phase=False, full_window=True, fast_mode=False)
            if best is None:
                self.unassigned_optional.add(p)
            else:
                self._assign_patient(p, *best)

    def _shuffle_in_blocks(self, items: list[int], block_size: int = 5) -> list[int]:
        if len(items) <= 1:
            return items[:]
        out: list[int] = []
        for idx in range(0, len(items), block_size):
            block = items[idx:idx + block_size]
            self.random.shuffle(block)
            out.extend(block)
        return out

    def _can_assign_room(self, p: int, room: int, admission_day: int) -> bool:
        inst = self.instance
        if inst.room_compatible[p][room] == 0:
            return False
        gender = self._patient_gender(p)
        capacity = inst.room_capacity[room]
        for d in self._stay_days(p, admission_day):
            if d not in inst.D:
                break
            if self.room_occupancy[room][d] + 1 > capacity:
                return False
            current_gender = self.room_gender[room][d]
            if current_gender is not None and current_gender != gender:
                return False
        return True

    def _can_assign_surgery(self, p: int, ot: int, admission_day: int) -> bool:
        inst = self.instance
        duration = inst.surgery_duration[p]
        surgeon = self._patient_surgeon(p)
        if self.ot_minutes[ot][admission_day] + duration > inst.max_ot_availability[ot][admission_day]:
            return False
        if self.surgeon_minutes[surgeon][admission_day] + duration > inst.max_surgery_time[surgeon][admission_day]:
            return False
        return True

    def _can_assign(self, p: int, day: int, room: int, ot: int) -> bool:
        return self._can_assign_room(p, room, day) and self._can_assign_surgery(p, ot, day)

    def _super_critical_key(self, p: int) -> tuple:
        inst = self.instance
        window = inst.surgery_due_day[p] - inst.surgery_release_day[p] + 1
        dur = inst.surgery_duration[p]
        los = inst.length_of_stay[p]
        n_rooms = self._num_compatible_rooms(p)
        return (window, -dur, -los, n_rooms, p)

    def _is_super_critical(self, p: int) -> bool:
        inst = self.instance
        window = inst.surgery_due_day[p] - inst.surgery_release_day[p] + 1
        dur = inst.surgery_duration[p]
        los = inst.length_of_stay[p]
        return (window <= 1) or (window <= 2 and dur >= 240) or (window <= 2 and los >= 7)

    def _candidate_days(self, p: int, mandatory_phase: bool, full_window: bool = False) -> list[int]:
        inst = self.instance
        release = inst.surgery_release_day[p]
        due = inst.surgery_due_day[p]
        days = [d for d in inst.D if release <= d <= due]
        if not days:
            return []
        if mandatory_phase or full_window:
            cand = days[:]
        else:
            mid = (release + due) // 2
            cand = [release, release + 1, mid, due - 1, due]
            cand = [d for d in cand if d in inst.D and release <= d <= due]
            cand = sorted(set(cand))
        cand.sort(key=lambda d: self._day_priority(p, d))
        if self.randomized and len(cand) > 1:
            prefix = cand[: min(3, len(cand))]
            rest = cand[min(3, len(cand)):]
            self.random.shuffle(prefix)
            cand = prefix + rest
        return cand

    def _day_priority(self, p: int, day: int) -> tuple:
        inst = self.instance
        surgeon = self._patient_surgeon(p)
        dur = inst.surgery_duration[p]
        release = inst.surgery_release_day[p]
        feasible_ot_count = sum(1 for ot in inst.OT if self._can_assign_surgery(p, ot, day))
        rem_surgeon = inst.max_surgery_time[surgeon][day] - self.surgeon_minutes[surgeon][day]
        return (max(0, day - release), -feasible_ot_count, -(rem_surgeon - dur), day)

    def _sorted_feasible_ots(self, p: int, day: int) -> list[int]:
        inst = self.instance
        surgeon = self._patient_surgeon(p)
        dur = inst.surgery_duration[p]
        feasible = [ot for ot in inst.OT if self._can_assign_surgery(p, ot, day)]

        def ot_key(ot: int) -> tuple:
            already_open = 0 if self.ot_minutes[ot][day] > 0 else 1
            same_surgeon_same_ot = 0
            for q in inst.incoming_patients:
                a = self.assignment[q]
                if not a.admitted:
                    continue
                if a.day == day and a.ot == ot and inst.surgeon_assignment[q].get(surgeon, 0) == 1:
                    same_surgeon_same_ot = 1
                    break
            residual = inst.max_ot_availability[ot][day] - (self.ot_minutes[ot][day] + dur)
            return (already_open, -same_surgeon_same_ot, residual, ot)

        feasible.sort(key=ot_key)
        return feasible

    def _sorted_feasible_rooms(self, p: int, day: int) -> list[int]:
        inst = self.instance
        gender = self._patient_gender(p)
        feasible = [room for room in inst.R if self._can_assign_room(p, room, day)]

        def room_key(room: int) -> tuple:
            empty_gender_days = 0
            cap_slack_sum = 0
            proxy_delta = self._proxy_delta_if_assigned(p, room, day)
            for d in self._stay_days(p, day):
                if d not in inst.D:
                    break
                occ = self.room_occupancy[room][d]
                cap = inst.room_capacity[room]
                cap_slack_sum += (cap - (occ + 1))
                current_gender = self.room_gender[room][d]
                if current_gender is None:
                    empty_gender_days += 1
                elif current_gender != gender:
                    empty_gender_days += 100000
            return (empty_gender_days, proxy_delta, cap_slack_sum, room)

        feasible.sort(key=room_key)
        return feasible

    def _occupied_room_days_delta_if_assigned(self, room: int, admission_day: int, los: int) -> dict[int, int]:
        delta_by_day: dict[int, int] = {}
        for d in range(admission_day, admission_day + los):
            if d not in self.instance.D:
                break
            delta_by_day[d] = 1 if self.room_occupancy[room][d] == 0 else 0
        return delta_by_day

    def _room_age_delta_if_assigned(self, p: int, room: int, day: int) -> int:
        delta = 0
        age = self.instance.age_group[p]
        for d in self._stay_days(p, day):
            if d not in self.instance.D:
                break
            pats = self.room_patients[room][d]
            if not pats:
                old_span = 0
                new_span = 0
            else:
                ages = [self.instance.age_group[q] for q in pats]
                old_span = max(ages) - min(ages)
                new_span = max(max(ages), age) - min(min(ages), age)
            delta += (new_span - old_span)
        return delta

    def _surgeon_transfer_delta_if_assigned(self, p: int, ot: int, day: int) -> int:
        surgeon = self._patient_surgeon(p)
        used_ots = set()
        for q in self.instance.incoming_patients:
            a = self.assignment[q]
            if not a.admitted or a.day != day:
                continue
            if self.instance.surgeon_assignment[q].get(surgeon, 0) == 1:
                used_ots.add(a.ot)
        old_transfers = max(0, len(used_ots) - 1)
        used_ots.add(ot)
        new_transfers = max(0, len(used_ots) - 1)
        return new_transfers - old_transfers

    def _room_balance_delta_if_assigned(self, p: int, room: int, day: int, *, attr: str) -> float:
        inst = self.instance
        patient_val = float(getattr(inst, attr)[p])
        total_delta = 0.0
        n_rooms = len(inst.R)
        room_totals = self.room_wl_total if attr == "wl_daily" else self.room_skill_total
        for d in self._stay_days(p, day):
            if d not in inst.D:
                break
            old_total = sum(room_totals[r][d] for r in inst.R)
            new_total = old_total + patient_val
            old_mean = old_total / n_rooms
            new_mean = new_total / n_rooms
            old_dev = 0.0
            new_dev = 0.0
            for r in inst.R:
                old_val = room_totals[r][d]
                new_val = old_val + patient_val if r == room else old_val
                old_dev += abs(old_val - old_mean)
                new_dev += abs(new_val - new_mean)
            total_delta += (new_dev - old_dev)
        return total_delta

    def _proxy_delta_if_assigned(self, p: int, room: int, day: int) -> float:
        inst = self.instance
        strategy = self.config.proxy_strategy
        pw = self.config.proxy_weight
        if strategy == ProxyStrategy.NONE or pw == 0.0:
            return 0.0
        occupied_delta = self._occupied_room_days_delta_if_assigned(room, day, inst.length_of_stay[p])
        if strategy == ProxyStrategy.MAXIMIZE_ROOMS:
            return -pw * sum(occupied_delta.values())
        if strategy == ProxyStrategy.MINIMIZE_ROOMS:
            return pw * sum(occupied_delta.values())
        if strategy == ProxyStrategy.STABLE_ROOMS:
            pairs: set[tuple[int, int]] = set()
            for d in occupied_delta:
                if d - 1 in inst.D:
                    pairs.add((d - 1, d))
                if d + 1 in inst.D:
                    pairs.add((d, d + 1))
            delta = 0.0
            for d1, d2 in pairs:
                old_1 = self.occupied_rooms_day[d1]
                old_2 = self.occupied_rooms_day[d2]
                new_1 = old_1 + occupied_delta.get(d1, 0)
                new_2 = old_2 + occupied_delta.get(d2, 0)
                delta += abs(new_2 - new_1) - abs(old_2 - old_1)
            return pw * delta
        if strategy == ProxyStrategy.BALANCE_WORKLOAD:
            return self._nurse_balance_proxy_weight(attr="wl_daily") * self._room_balance_delta_if_assigned(
                p,
                room,
                day,
                attr="wl_daily",
            )
        if strategy == ProxyStrategy.BALANCE_SKILL:
            return self._nurse_balance_proxy_weight(attr="skill_daily") * self._room_balance_delta_if_assigned(
                p,
                room,
                day,
                attr="skill_daily",
            )
        if strategy == ProxyStrategy.HYBRID:
            return (
                self._nurse_balance_proxy_weight(attr="skill_daily")
                * self._room_balance_delta_if_assigned(p, room, day, attr="skill_daily")
                + self._nurse_balance_proxy_weight(attr="wl_daily")
                * self._room_balance_delta_if_assigned(p, room, day, attr="wl_daily")
            )
        raise ValueError(f"Unsupported proxy strategy: {strategy}")

    def _nurse_balance_proxy_weight(self, *, attr: str) -> float:
        """Return the weighted lambda used by nurse-related balance proxies."""

        if attr == "skill_daily":
            return self.config.proxy_weight * self.instance.weights["W2"]
        if attr == "wl_daily":
            return self.config.proxy_weight * self.instance.weights["W4"]
        raise ValueError(f"Unsupported nurse-balance attribute: {attr}")

    def _weighted_assignment_delta(self, p: int, day: int, room: int, ot: int) -> float:
        inst = self.instance
        w = inst.weights
        delta_age = self._room_age_delta_if_assigned(p, room, day)
        delta_open_ot = 0 if self.ot_minutes[ot][day] > 0 else 1
        delta_transfer = self._surgeon_transfer_delta_if_assigned(p, ot, day)
        delta_delay = max(0, day - inst.surgery_release_day[p])
        delta_proxy = self._proxy_delta_if_assigned(p, room, day)
        return (
            w["W1"] * delta_age
            + w["W5"] * delta_open_ot
            + w["W6"] * delta_transfer
            + w["W7"] * delta_delay
            + delta_proxy
        )

    def _local_assignment_score(self, p: int, day: int, room: int, ot: int) -> tuple:
        inst = self.instance
        weighted_delta = self._weighted_assignment_delta(p, day, room, ot)
        if p in inst.PO:
            weighted_delta -= inst.weights["W8"]
        surgeon = self._patient_surgeon(p)
        dur = inst.surgery_duration[p]
        room_slack = 0
        for d in self._stay_days(p, day):
            if d not in inst.D:
                break
            room_slack += inst.room_capacity[room] - (self.room_occupancy[room][d] + 1)
        ot_slack = inst.max_ot_availability[ot][day] - (self.ot_minutes[ot][day] + dur)
        surg_slack = inst.max_surgery_time[surgeon][day] - (self.surgeon_minutes[surgeon][day] + dur)
        delay = max(0, day - inst.surgery_release_day[p])
        return (weighted_delta, delay, ot_slack + surg_slack, room_slack, room, ot)

    def _removal_penalty_estimate(self, q: int) -> float:
        a = self.assignment[q]
        if not a.admitted:
            return 0.0
        inst = self.instance
        base = self._weighted_assignment_delta(q, a.day, a.room, a.ot)
        if q in inst.PO:
            return inst.weights["W8"] - base
        return base

    def _find_assignment_bpp(self, p: int, *, mandatory_phase: bool, full_window: bool = False, fast_mode: bool = False) -> Optional[tuple[int, int, int]]:
        if mandatory_phase and self._is_super_critical(p):
            fast_mode = False
        candidate_days = self._candidate_days(p, mandatory_phase, full_window=full_window)
        if fast_mode:
            candidate_days = candidate_days[: min(4 if p in self.instance.PM else 3, len(candidate_days))]
        best = None
        best_score = None
        for day in candidate_days:
            feasible_ots = self._sorted_feasible_ots(p, day)
            feasible_rooms = self._sorted_feasible_rooms(p, day)
            if fast_mode:
                feasible_ots = feasible_ots[: min(3, len(feasible_ots))]
                feasible_rooms = feasible_rooms[: min(3, len(feasible_rooms))]
            if not feasible_ots or not feasible_rooms:
                continue
            for ot in feasible_ots:
                for room in feasible_rooms:
                    if not self._can_assign(p, day, room, ot):
                        continue
                    score = self._local_assignment_score(p, day, room, ot)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (day, room, ot)
                    if fast_mode and best is not None:
                        if (p in self.instance.PO) and score[0] > 0:
                            continue
                        return best
        if best is None:
            return None
        if (p in self.instance.PO) and best_score[0] > 0:
            return None
        return best

    def _find_assignment_forced_day(self, p: int, day: int, *, fast_mode: bool = False) -> Optional[tuple[int, int, int]]:
        feasible_ots = self._sorted_feasible_ots(p, day)
        feasible_rooms = self._sorted_feasible_rooms(p, day)
        if fast_mode:
            feasible_ots = feasible_ots[: min(4, len(feasible_ots))]
            feasible_rooms = feasible_rooms[: min(4, len(feasible_rooms))]
        if not feasible_ots or not feasible_rooms:
            return None
        best = None
        best_score = None
        for ot in feasible_ots:
            for room in feasible_rooms:
                if not self._can_assign(p, day, room, ot):
                    continue
                score = self._local_assignment_score(p, day, room, ot)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (day, room, ot)
                if fast_mode and best is not None:
                    if (p in self.instance.PO) and score[0] > 0:
                        continue
                    return best
        if best is None:
            return None
        if (p in self.instance.PO) and best_score[0] > 0:
            return None
        return best

    def _recompute_room_day_state(self, room: int, day: int) -> None:
        if day not in self.instance.D:
            return
        pats = self.room_patients[room][day]
        if not pats:
            self.room_gender[room][day] = None
            self.room_age_min[room][day] = None
            self.room_age_max[room][day] = None
            return
        genders = {self._patient_gender(p) for p in pats}
        self.room_gender[room][day] = next(iter(genders)) if genders else None
        ages = [self.instance.age_group[p] for p in pats]
        self.room_age_min[room][day] = min(ages)
        self.room_age_max[room][day] = max(ages)

    def _assign_occupant(self, p: int, admission_day: int, room: int) -> None:
        inst = self.instance
        age = inst.age_group[p]
        gender = self._patient_gender(p)
        wl_daily = float(inst.wl_daily.get(p, 0.0))
        skill_daily = float(inst.skill_daily.get(p, 0.0))
        self.assignment[p] = PatientAssignment(admitted=True, day=admission_day, room=room, ot=None)
        for d in self._stay_days(p, admission_day):
            if d not in inst.D:
                break
            if self.room_occupancy[room][d] == 0:
                self.occupied_rooms_day[d] += 1
            self.room_patients[room][d].append(p)
            self.room_occupancy[room][d] += 1
            self.room_wl_total[room][d] += wl_daily
            self.room_skill_total[room][d] += skill_daily
            if self.room_gender[room][d] is None:
                self.room_gender[room][d] = gender
            if self.room_age_min[room][d] is None:
                self.room_age_min[room][d] = age
                self.room_age_max[room][d] = age
            else:
                self.room_age_min[room][d] = min(self.room_age_min[room][d], age)
                self.room_age_max[room][d] = max(self.room_age_max[room][d], age)

    def _assign_patient(self, p: int, admission_day: int, room: int, ot: int) -> None:
        inst = self.instance
        age = inst.age_group[p]
        gender = self._patient_gender(p)
        duration = inst.surgery_duration[p]
        surgeon = self._patient_surgeon(p)
        wl_daily = float(inst.wl_daily.get(p, 0.0))
        skill_daily = float(inst.skill_daily.get(p, 0.0))
        self.assignment[p] = PatientAssignment(admitted=True, day=admission_day, room=room, ot=ot)
        for d in self._stay_days(p, admission_day):
            if d not in inst.D:
                break
            if self.room_occupancy[room][d] == 0:
                self.occupied_rooms_day[d] += 1
            self.room_patients[room][d].append(p)
            self.room_occupancy[room][d] += 1
            self.room_wl_total[room][d] += wl_daily
            self.room_skill_total[room][d] += skill_daily
            if self.room_gender[room][d] is None:
                self.room_gender[room][d] = gender
            if self.room_age_min[room][d] is None:
                self.room_age_min[room][d] = age
                self.room_age_max[room][d] = age
            else:
                self.room_age_min[room][d] = min(self.room_age_min[room][d], age)
                self.room_age_max[room][d] = max(self.room_age_max[room][d], age)
        self.ot_minutes[ot][admission_day] += duration
        self.surgeon_minutes[surgeon][admission_day] += duration

    def _remove_patient(self, p: int) -> None:
        a = self.assignment[p]
        if not a.admitted:
            return
        room = a.room
        day = a.day
        ot = a.ot
        wl_daily = float(self.instance.wl_daily.get(p, 0.0))
        skill_daily = float(self.instance.skill_daily.get(p, 0.0))
        for d in self._stay_days(p, day):
            if d not in self.instance.D:
                continue
            if p in self.room_patients[room][d]:
                self.room_patients[room][d].remove(p)
            self.room_occupancy[room][d] -= 1
            if self.room_occupancy[room][d] == 0:
                self.occupied_rooms_day[d] -= 1
            self.room_wl_total[room][d] -= wl_daily
            self.room_skill_total[room][d] -= skill_daily
            self._recompute_room_day_state(room, d)
        if ot is not None:
            duration = self.instance.surgery_duration[p]
            surgeon = self._patient_surgeon(p)
            self.ot_minutes[ot][day] -= duration
            self.surgeon_minutes[surgeon][day] -= duration
        self.assignment[p] = PatientAssignment()

    def _refresh_unassigned_sets(self) -> None:
        self.unassigned_mandatory = {p for p in self.instance.PM if not self.assignment[p].admitted}
        self.unassigned_optional = {p for p in self.instance.PO if not self.assignment[p].admitted}

    def _snapshot_state(self) -> dict:
        return {
            "assignment": copy.deepcopy(self.assignment),
            "room_patients": copy.deepcopy(self.room_patients),
            "room_occupancy": copy.deepcopy(self.room_occupancy),
            "room_gender": copy.deepcopy(self.room_gender),
            "room_age_min": copy.deepcopy(self.room_age_min),
            "room_age_max": copy.deepcopy(self.room_age_max),
            "room_wl_total": copy.deepcopy(self.room_wl_total),
            "room_skill_total": copy.deepcopy(self.room_skill_total),
            "occupied_rooms_day": copy.deepcopy(self.occupied_rooms_day),
            "ot_minutes": copy.deepcopy(self.ot_minutes),
            "surgeon_minutes": copy.deepcopy(self.surgeon_minutes),
            "unassigned_mandatory": set(self.unassigned_mandatory),
            "unassigned_optional": set(self.unassigned_optional),
        }

    def _restore_state(self, state: dict) -> None:
        self.assignment = state["assignment"]
        self.room_patients = state["room_patients"]
        self.room_occupancy = state["room_occupancy"]
        self.room_gender = state["room_gender"]
        self.room_age_min = state["room_age_min"]
        self.room_age_max = state["room_age_max"]
        self.room_wl_total = state["room_wl_total"]
        self.room_skill_total = state["room_skill_total"]
        self.occupied_rooms_day = state["occupied_rooms_day"]
        self.ot_minutes = state["ot_minutes"]
        self.surgeon_minutes = state["surgeon_minutes"]
        self.unassigned_mandatory = set(state["unassigned_mandatory"])
        self.unassigned_optional = set(state["unassigned_optional"])

    def _count_alternative_days(self, p: int) -> int:
        if p in self.instance.PA:
            return -1
        cnt = 0
        for d in self._candidate_days(p, mandatory_phase=(p in self.instance.PM), full_window=True):
            if self._sorted_feasible_ots(p, d) and self._sorted_feasible_rooms(p, d):
                cnt += 1
        return cnt

    def _candidate_blockers(self, p: int, day: int) -> list[int]:
        inst = self.instance
        blockers: set[int] = set()
        gender = self._patient_gender(p)
        for q in inst.incoming_patients:
            a = self.assignment[q]
            if a.admitted and a.day == day:
                blockers.add(q)
        for room in inst.R:
            for d in self._stay_days(p, day):
                if d not in inst.D:
                    continue
                if self.room_gender[room][d] not in (None, gender):
                    blockers.update(self.room_patients[room][d])
                elif self.room_occupancy[room][d] >= inst.room_capacity[room]:
                    blockers.update(self.room_patients[room][d])

        def blocker_key(q: int) -> tuple:
            mandatory_pen = 1 if q in inst.PM else 0
            removal_pen = self._removal_penalty_estimate(q)
            mobility = -self._count_alternative_days(q)
            dur = -inst.surgery_duration[q]
            los = -inst.length_of_stay[q]
            return (mandatory_pen, removal_pen, mobility, dur, los, q)

        out = [q for q in blockers if q not in inst.PA]
        out.sort(key=blocker_key)
        return out[:8]

    def _try_reinsert_removed(self, q: int) -> bool:
        best = self._find_assignment_bpp(q, mandatory_phase=(q in self.instance.PM), full_window=True, fast_mode=False)
        if best is None:
            return False
        self._assign_patient(q, *best)
        return True

    def _targeted_rescue_for_mandatory(self, p: int) -> bool:
        candidate_days = self._candidate_days(p, mandatory_phase=True, full_window=True)
        rescue_cap = 7 if self._is_super_critical(p) else 5
        candidate_days = candidate_days[: min(rescue_cap, len(candidate_days))]
        for day in candidate_days:
            best_here = self._find_assignment_forced_day(p, day, fast_mode=False)
            if best_here is not None:
                self._assign_patient(p, *best_here)
                return True
            blockers = self._candidate_blockers(p, day)
            if not blockers:
                continue
            for q in blockers[:6]:
                aq = self.assignment[q]
                if not aq.admitted:
                    continue
                old_q = (aq.day, aq.room, aq.ot)
                self._remove_patient(q)
                best_p = self._find_assignment_forced_day(p, day, fast_mode=False)
                if best_p is not None:
                    self._assign_patient(p, *best_p)
                    if self._try_reinsert_removed(q):
                        return True
                    if q in self.instance.PO:
                        self.unassigned_optional.discard(q)
                        return True
                    self._remove_patient(p)
                    self._assign_patient(q, *old_q)
                else:
                    self._assign_patient(q, *old_q)
            top_blockers = blockers[:4]
            for idx, q1 in enumerate(top_blockers):
                a1 = self.assignment[q1]
                if not a1.admitted:
                    continue
                old_q1 = (a1.day, a1.room, a1.ot)
                self._remove_patient(q1)
                for q2 in [q for q in top_blockers if q != q1]:
                    a2 = self.assignment[q2]
                    if not a2.admitted:
                        continue
                    old_q2 = (a2.day, a2.room, a2.ot)
                    self._remove_patient(q2)
                    best_p = self._find_assignment_forced_day(p, day, fast_mode=False)
                    if best_p is not None:
                        self._assign_patient(p, *best_p)
                        ok_q1 = self._try_reinsert_removed(q1)
                        ok_q2 = self._try_reinsert_removed(q2)
                        if ok_q1 and ok_q2:
                            return True
                        bad = False
                        if (q1 in self.instance.PM) and (not ok_q1):
                            bad = True
                        if (q2 in self.instance.PM) and (not ok_q2):
                            bad = True
                        if not bad:
                            if q1 in self.instance.PO and not ok_q1:
                                self.unassigned_optional.add(q1)
                            if q2 in self.instance.PO and not ok_q2:
                                self.unassigned_optional.add(q2)
                            return True
                        self._remove_patient(p)
                        if ok_q1:
                            self._remove_patient(q1)
                        if ok_q2:
                            self._remove_patient(q2)
                        self._assign_patient(q1, *old_q1)
                        self._assign_patient(q2, *old_q2)
                    else:
                        self._assign_patient(q2, *old_q2)
                if not self.assignment[q1].admitted:
                    self._assign_patient(q1, *old_q1)
        return False

    def repair_feasibility(self, *, max_seconds: float | None = None, max_destroy_size: int = 6) -> bool:
        self._refresh_unassigned_sets()
        if not self.unassigned_mandatory:
            return True
        start = time.time()
        budget = self.config.repair_seconds if max_seconds is None else max_seconds
        improved = True
        while self.unassigned_mandatory and improved and (time.time() - start) < budget:
            improved = False
            targets = sorted(self.unassigned_mandatory, key=self._repair_mandatory_key)[: min(3, len(self.unassigned_mandatory))]
            for p in targets:
                if self.assignment[p].admitted:
                    continue
                if self._targeted_rescue_for_mandatory(p):
                    self._refresh_unassigned_sets()
                    improved = True
                    break
                days = self._candidate_days(p, mandatory_phase=True, full_window=True)[:3]
                local_success = False
                for day in days:
                    for destroy_size in (3, 4, max_destroy_size):
                        if (time.time() - start) >= budget:
                            break
                        if self._attempt_destroy_repair(p, day, destroy_size):
                            self._refresh_unassigned_sets()
                            improved = True
                            local_success = True
                            break
                    if local_success or (time.time() - start) >= budget:
                        break
                if improved:
                    break
        self._refresh_unassigned_sets()
        if not self.unassigned_mandatory:
            missed_optional = [q for q in self.instance.PO if not self.assignment[q].admitted]
            for q in sorted(missed_optional, key=self._difficulty_key):
                best = self._find_assignment_bpp(q, mandatory_phase=False, full_window=True, fast_mode=False)
                if best is not None:
                    self._assign_patient(q, *best)
            self._refresh_unassigned_sets()
        return len(self.unassigned_mandatory) == 0

    def _collect_destroy_candidates(self, p: int, focus_day: int) -> list[int]:
        inst = self.instance
        surgeon = self._patient_surgeon(p)
        candidates: set[int] = set()
        candidates.update(self._candidate_blockers(p, focus_day))
        cand_days = set(self._candidate_days(p, mandatory_phase=True, full_window=True)[:4])
        for q in inst.incoming_patients:
            a = self.assignment[q]
            if not a.admitted or q in inst.PA or a.day not in cand_days:
                continue
            if inst.surgeon_assignment[q].get(surgeon, 0) == 1:
                candidates.add(q)
        for q in inst.incoming_patients:
            a = self.assignment[q]
            if not a.admitted or q in inst.PA:
                continue
            stay = set(self._stay_days(q, a.day))
            if focus_day in stay:
                candidates.add(q)
        candidates.discard(p)

        def cand_key(q: int) -> tuple:
            mandatory_pen = 1 if q in inst.PM else 0
            mobility = -self._count_alternative_days(q)
            penalty = self._removal_penalty_estimate(q)
            dur = -inst.surgery_duration[q]
            los = -inst.length_of_stay[q]
            return (mandatory_pen, mobility, penalty, dur, los, q)

        out = [q for q in candidates if q not in inst.PA and self.assignment[q].admitted]
        out.sort(key=cand_key)
        return out

    def _rebuild_after_destroy(self, primary_mandatory: int, removed: list[int]) -> None:
        inst = self.instance
        removed_mandatory = [q for q in removed if q in inst.PM]
        removed_optional = [q for q in removed if q in inst.PO]
        other_unassigned_mandatory = [
            q for q in sorted(inst.PM, key=self._repair_mandatory_key)
            if (q != primary_mandatory) and (not self.assignment[q].admitted)
        ]
        ordered = [primary_mandatory]
        ordered.extend(q for q in other_unassigned_mandatory if q not in ordered)
        ordered.extend(q for q in removed_mandatory if q not in ordered)
        ordered.extend(q for q in removed_optional if q not in ordered)
        for q in ordered:
            if self.assignment[q].admitted:
                continue
            best = self._find_assignment_bpp(q, mandatory_phase=(q in inst.PM), full_window=True, fast_mode=False)
            if best is not None:
                self._assign_patient(q, *best)
        self._refresh_unassigned_sets()

    def _attempt_destroy_repair(self, p: int, focus_day: int, destroy_size: int) -> bool:
        state = self._snapshot_state()
        before = self.summary()
        before_obj = self.compute_objective()["total_with_proxy"]
        candidates = self._collect_destroy_candidates(p, focus_day)
        if not candidates:
            return False
        removed: list[int] = []
        for q in candidates[:destroy_size]:
            if not self.assignment[q].admitted:
                continue
            self._remove_patient(q)
            removed.append(q)
        self._refresh_unassigned_sets()
        self._rebuild_after_destroy(p, removed)
        after = self.summary()
        after_obj = self.compute_objective()["total_with_proxy"]
        accepted = False
        if after["mandatory_unassigned"] < before["mandatory_unassigned"]:
            accepted = True
        elif after["mandatory_unassigned"] == before["mandatory_unassigned"] and after_obj + 1e-9 < before_obj:
            accepted = True
        if not accepted:
            self._restore_state(state)
            return False
        return True

    def summary(self) -> dict[str, int]:
        admitted_total = sum(1 for p in self.instance.P if self.assignment[p].admitted)
        admitted_incoming = sum(1 for p in self.instance.incoming_patients if self.assignment[p].admitted)
        admitted_mandatory = sum(1 for p in self.instance.PM if self.assignment[p].admitted)
        admitted_optional = sum(1 for p in self.instance.PO if self.assignment[p].admitted)
        mandatory_unassigned = sum(1 for p in self.instance.PM if not self.assignment[p].admitted)
        optional_unassigned = sum(1 for p in self.instance.PO if not self.assignment[p].admitted)
        return {
            "admitted_total": admitted_total,
            "admitted_incoming": admitted_incoming,
            "admitted_mandatory": admitted_mandatory,
            "admitted_optional": admitted_optional,
            "mandatory_unassigned": mandatory_unassigned,
            "optional_unassigned": optional_unassigned,
        }

    def _balance_proxy_objective(self, *, attr: str) -> float:
        inst = self.instance
        room_totals = self.room_wl_total if attr == "wl_daily" else self.room_skill_total
        total_obj = 0.0
        n_rooms = len(inst.R)
        for d in inst.D:
            total_d = sum(room_totals[r][d] for r in inst.R)
            mean_d = total_d / n_rooms
            for r in inst.R:
                total_obj += abs(room_totals[r][d] - mean_d)
        return total_obj

    def _proxy_objective_value(self) -> float:
        inst = self.instance
        strategy = self.config.proxy_strategy
        pw = self.config.proxy_weight
        if strategy == ProxyStrategy.NONE or pw == 0.0:
            return 0.0
        total_room_days = sum(self.occupied_rooms_day[d] for d in inst.D)
        if strategy == ProxyStrategy.MAXIMIZE_ROOMS:
            return -pw * total_room_days
        if strategy == ProxyStrategy.MINIMIZE_ROOMS:
            return pw * total_room_days
        if strategy == ProxyStrategy.STABLE_ROOMS:
            return pw * sum(
                abs(self.occupied_rooms_day[d2] - self.occupied_rooms_day[d1])
                for d1, d2 in zip(inst.D[:-1], inst.D[1:])
            )
        if strategy == ProxyStrategy.BALANCE_WORKLOAD:
            return self._nurse_balance_proxy_weight(attr="wl_daily") * self._balance_proxy_objective(attr="wl_daily")
        if strategy == ProxyStrategy.BALANCE_SKILL:
            return self._nurse_balance_proxy_weight(attr="skill_daily") * self._balance_proxy_objective(attr="skill_daily")
        if strategy == ProxyStrategy.HYBRID:
            return (
                self._nurse_balance_proxy_weight(attr="skill_daily") * self._balance_proxy_objective(attr="skill_daily")
                + self._nurse_balance_proxy_weight(attr="wl_daily") * self._balance_proxy_objective(attr="wl_daily")
            )
        raise ValueError(f"Unsupported proxy strategy: {strategy}")

    def compute_objective(self) -> dict[str, float]:
        inst = self.instance
        w = inst.weights
        obj_age = 0.0
        for r in inst.R:
            for d in inst.D:
                a_min = self.room_age_min[r][d]
                a_max = self.room_age_max[r][d]
                if a_min is not None and a_max is not None:
                    obj_age += (a_max - a_min)
        obj_open_ot = 0.0
        for o in inst.OT:
            for d in inst.D:
                if self.ot_minutes[o][d] > 0:
                    obj_open_ot += 1
        obj_transfer = 0.0
        for c in inst.C:
            for d in inst.D:
                ots_used = 0
                for o in inst.OT:
                    used_here = False
                    for p in inst.incoming_patients:
                        a = self.assignment[p]
                        if not a.admitted:
                            continue
                        if a.day != d or a.ot != o:
                            continue
                        if inst.surgeon_assignment[p].get(c, 0) == 1:
                            used_here = True
                            break
                    if used_here:
                        ots_used += 1
                if ots_used > 0:
                    obj_transfer += max(0, ots_used - 1)
        obj_delay = 0.0
        for p in inst.incoming_patients:
            a = self.assignment[p]
            if a.admitted:
                obj_delay += max(0, a.day - inst.surgery_release_day[p])
        obj_unscheduled = sum(1 for p in inst.PO if not self.assignment[p].admitted)
        proxy_workload = self._balance_proxy_objective(attr="wl_daily")
        proxy_skill = self._balance_proxy_objective(attr="skill_daily")
        proxy_value = self._proxy_objective_value()
        total_phase1 = (
            w["W1"] * obj_age
            + w["W5"] * obj_open_ot
            + w["W6"] * obj_transfer
            + w["W7"] * obj_delay
            + w["W8"] * obj_unscheduled
        )
        return {
            "age_mix": obj_age,
            "open_ot": obj_open_ot,
            "surgeon_transfer": obj_transfer,
            "delay": obj_delay,
            "unscheduled_optional": obj_unscheduled,
            "proxy_workload_balance": proxy_workload,
            "proxy_skill_balance": proxy_skill,
            "proxy_value": proxy_value,
            "total_phase1_like": total_phase1,
            "total_with_proxy": total_phase1 + proxy_value,
        }

    def validate(self) -> tuple[bool, list[str]]:
        inst = self.instance
        errors: list[str] = []
        for p in inst.PM:
            if not self.assignment[p].admitted:
                errors.append(f"Mandatory patient {p} not admitted")
        for r in inst.R:
            for d in inst.D:
                pats = self.room_patients[r][d]
                if len(pats) > inst.room_capacity[r]:
                    errors.append(f"Room capacity violated in room {r}, day {d}")
                genders = {self._patient_gender(p) for p in pats}
                if len(genders) > 1:
                    errors.append(f"Gender mixing in room {r}, day {d}")
        for o in inst.OT:
            for d in inst.D:
                if self.ot_minutes[o][d] > inst.max_ot_availability[o][d]:
                    errors.append(f"OT capacity violated in OT {o}, day {d}")
        for c in inst.C:
            for d in inst.D:
                if self.surgeon_minutes[c][d] > inst.max_surgery_time[c][d]:
                    errors.append(f"Surgeon capacity violated for surgeon {c}, day {d}")
        for p in inst.incoming_patients:
            a = self.assignment[p]
            if not a.admitted:
                continue
            if a.day is None or a.room is None or a.ot is None:
                errors.append(f"Patient {p} has incomplete triple")
                continue
            if not (inst.surgery_release_day[p] <= a.day <= inst.surgery_due_day[p]):
                errors.append(f"Patient {p} outside admission window")
            if inst.room_compatible[p][a.room] == 0:
                errors.append(f"Patient {p} assigned to incompatible room")
        return (len(errors) == 0, errors)

    def _iter_candidate_triples_weighted(self, p: int) -> list[tuple[tuple[float, ...], tuple[int, int, int]]]:
        triples: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
        candidate_days = self._candidate_days(p, mandatory_phase=(p in self.instance.PM), full_window=True)
        for day in candidate_days:
            feasible_ots = self._sorted_feasible_ots(p, day)
            feasible_rooms = self._sorted_feasible_rooms(p, day)
            if not feasible_ots or not feasible_rooms:
                continue
            for ot in feasible_ots:
                for room in feasible_rooms:
                    if not self._can_assign(p, day, room, ot):
                        continue
                    weighted = self._weighted_assignment_delta(p, day, room, ot)
                    if p in self.instance.PO:
                        weighted -= self.instance.weights["W8"]
                    delay = max(0, day - self.instance.surgery_release_day[p])
                    los = self.instance.length_of_stay[p]
                    dur = self.instance.surgery_duration[p]
                    triples.append(((weighted, delay, -los, -dur, room, ot), (day, room, ot)))
        triples.sort(key=lambda item: item[0])
        return triples

    def _iter_candidate_triples_proxy_aware(self, p: int) -> list[tuple[tuple[float, ...], tuple[int, int, int]]]:
        triples: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
        candidate_days = self._candidate_days(p, mandatory_phase=(p in self.instance.PM), full_window=True)
        for day in candidate_days:
            feasible_ots = self._sorted_feasible_ots(p, day)
            feasible_rooms = self._sorted_feasible_rooms(p, day)
            if not feasible_ots or not feasible_rooms:
                continue
            for ot in feasible_ots:
                for room in feasible_rooms:
                    if not self._can_assign(p, day, room, ot):
                        continue
                    phase1_delta = self._weighted_assignment_delta(p, day, room, ot) - self._proxy_delta_if_assigned(p, room, day)
                    proxy_delta = self._proxy_delta_if_assigned(p, room, day)
                    if p in self.instance.PO:
                        phase1_delta -= self.instance.weights["W8"]
                    delay = max(0, day - self.instance.surgery_release_day[p])
                    room_slack = 0
                    for d in self._stay_days(p, day):
                        if d not in self.instance.D:
                            break
                        room_slack += self.instance.room_capacity[room] - (self.room_occupancy[room][d] + 1)
                    triples.append(((phase1_delta, proxy_delta, delay, -room_slack, room, ot), (day, room, ot)))
        triples.sort(key=lambda item: item[0])
        return triples

    def _effective_top_k(self) -> int:
        n = len(self.instance.incoming_patients)
        if n >= self.config.large_instance_threshold:
            return max(self.config.repair_top_k, self.config.repair_top_k_large_instance)
        return self.config.repair_top_k

    def _pick_from_top_k(self, triples, top_k: int, random_prob: float):
        if not triples:
            return None
        k = min(max(1, top_k), len(triples))
        if k == 1:
            return triples[0][1]
        if self.alns_random.random() < random_prob:
            return triples[self.alns_random.randrange(k)][1]
        return triples[0][1]

    def _choose_costly_admitted(self, top_n: int = 12, optional_bias: bool = False) -> int | None:
        admitted = self.admitted_incoming()
        if not admitted:
            return None
        ranked = sorted(
            admitted,
            key=lambda p: (
                0 if (optional_bias and p in self.instance.PO) else 1,
                -self._patient_cost_score(p),
                -self.instance.length_of_stay[p],
                -self.instance.surgery_duration[p],
                p,
            ),
        )
        head = ranked[: max(1, min(top_n, len(ranked)))]
        return self.alns_random.choice(head)

    def _patient_cost_score(self, p: int) -> float:
        if not self.assignment[p].admitted:
            return -1e18
        inst = self.instance
        a = self.assignment[p]
        if a.day is None or a.room is None or a.ot is None:
            return -1e18
        score = 0.0
        score += max(0, a.day - inst.surgery_release_day[p]) * inst.weights["W7"]
        if p in inst.PO:
            score += 0.20 * inst.weights["W8"]
        for d in self._stay_days(p, a.day):
            if d not in inst.D:
                continue
            a_min = self.room_age_min[a.room][d]
            a_max = self.room_age_max[a.room][d]
            if a_min is not None and a_max is not None:
                score += inst.weights["W1"] * (a_max - a_min) / max(1, inst.length_of_stay[p])
            score += 0.05 * abs(self._proxy_delta_if_assigned(p, a.room, a.day))
        pats_same_ot_day = [
            q for q in inst.incoming_patients
            if self.assignment[q].admitted and self.assignment[q].day == a.day and self.assignment[q].ot == a.ot
        ]
        if len(pats_same_ot_day) <= 1:
            score += inst.weights["W5"]
        surgeon = self._patient_surgeon(p)
        used_ots = {
            self.assignment[q].ot
            for q in inst.incoming_patients
            if self.assignment[q].admitted and self.assignment[q].day == a.day and self._patient_surgeon(q) == surgeon
        }
        if len(used_ots) > 1:
            score += inst.weights["W6"] * (len(used_ots) - 1)
        score += 0.02 * inst.length_of_stay[p] + 0.01 * inst.surgery_duration[p]
        return score

    def destroy_random(self, k: int) -> list[int]:
        admitted = self.admitted_incoming()
        if not admitted:
            return []
        optionals = [p for p in admitted if p in self.instance.PO]
        mandatory = [p for p in admitted if p in self.instance.PM]
        self.alns_random.shuffle(optionals)
        self.alns_random.shuffle(mandatory)
        removed = (optionals + mandatory)[:k]
        for p in removed:
            self._remove_patient(p)
        self._refresh_unassigned_sets()
        return removed

    def destroy_expensive(self, k: int) -> list[int]:
        admitted = self.admitted_incoming()
        if not admitted:
            return []
        ranked = sorted(
            admitted,
            key=lambda p: (
                -self._patient_cost_score(p),
                0 if p in self.instance.PO else 1,
                -self.instance.length_of_stay[p],
                -self.instance.surgery_duration[p],
                p,
            ),
        )
        removed = ranked[:k]
        for p in removed:
            self._remove_patient(p)
        self._refresh_unassigned_sets()
        return removed

    def destroy_bottleneck_day_ot(self, k: int) -> list[int]:
        inst = self.instance
        admitted = self.admitted_incoming()
        if not admitted:
            return []
        day_scores: list[tuple[float, int]] = []
        for d in inst.D:
            total_ot = sum(self.ot_minutes[o][d] for o in inst.OT)
            n_pats = sum(1 for p in admitted if self.assignment[p].day == d)
            transfer = 0.0
            for c in inst.C:
                ots_used = {
                    self.assignment[p].ot
                    for p in admitted
                    if self.assignment[p].day == d and self._patient_surgeon(p) == c
                }
                transfer += max(0, len(ots_used) - 1)
            score = n_pats + 0.002 * total_ot + 2.0 * transfer
            day_scores.append((score, d))
        _, chosen_day = max(day_scores)
        if self.alns_random.random() < 0.60:
            ot_scores = []
            for o in inst.OT:
                cnt = sum(1 for p in admitted if self.assignment[p].day == chosen_day and self.assignment[p].ot == o)
                minutes = self.ot_minutes[o][chosen_day]
                ot_scores.append((cnt + 0.001 * minutes, o))
            _, chosen_ot = max(ot_scores)
            candidates = [p for p in admitted if self.assignment[p].day == chosen_day and self.assignment[p].ot == chosen_ot]
        else:
            candidates = [p for p in admitted if self.assignment[p].day == chosen_day]
        candidates = sorted(candidates, key=lambda p: (0 if p in inst.PO else 1, -self._patient_cost_score(p), p))
        removed = candidates[:k]
        for p in removed:
            self._remove_patient(p)
        self._refresh_unassigned_sets()
        return removed

    def destroy_related(self, k: int) -> list[int]:
        inst = self.instance
        admitted = self.admitted_incoming()
        if not admitted:
            return []
        seed = self.alns_random.choice(admitted)
        a = self.assignment[seed]
        seed_surgeon = self._patient_surgeon(seed)
        scored = []
        for p in admitted:
            ap = self.assignment[p]
            same_day = 1 if ap.day == a.day else 0
            same_ot = 1 if ap.ot == a.ot else 0
            same_room = 1 if ap.room == a.room else 0
            same_surgeon = 1 if self._patient_surgeon(p) == seed_surgeon else 0
            los_gap = abs(inst.length_of_stay[p] - inst.length_of_stay[seed])
            dur_gap = abs(inst.surgery_duration[p] - inst.surgery_duration[seed])
            score = 5.0 * same_day + 4.0 * same_surgeon + 3.0 * same_ot + 2.5 * same_room - 0.08 * los_gap - 0.01 * dur_gap + (0.4 if p in inst.PO else 0.0)
            scored.append((score, p))
        scored.sort(key=lambda item: (-item[0], 0 if item[1] in inst.PO else 1, -self._patient_cost_score(item[1]), item[1]))
        removed = [p for _, p in scored[:k]]
        for p in removed:
            self._remove_patient(p)
        self._refresh_unassigned_sets()
        return removed

    def repair_greedy_randomized(self, removed: list[int]) -> None:
        ordered = self._ordered_repair_patients(removed)
        top_k = self._effective_top_k()
        for p in ordered:
            if self.assignment[p].admitted:
                continue
            triples = self._iter_candidate_triples_weighted(p)
            triple = self._pick_from_top_k(triples, top_k, self.config.greedy_pick_random_prob)
            if triple is not None:
                self._assign_patient(p, *triple)
        self._refresh_unassigned_sets()

    def repair_regret2_randomized(self, removed: list[int]) -> None:
        pending = self._ordered_repair_patients(removed)
        top_k = self._effective_top_k()
        while pending:
            best_choice = None
            best_key = None
            for p in pending:
                if self.assignment[p].admitted:
                    continue
                triples = self._iter_candidate_triples_weighted(p)
                if not triples:
                    continue
                first_cost = triples[0][0][0]
                second_cost = triples[1][0][0] if len(triples) > 1 else first_cost + 1e6
                regret = second_cost - first_cost
                difficulty = self._repair_mandatory_key(p) if p in self.instance.PM else self._difficulty_key(p)
                key = (0 if p in self.instance.PM else 1, -regret, len(triples), difficulty)
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = (p, self._pick_from_top_k(triples, top_k, self.config.regret_pick_random_prob))
            if best_choice is None:
                break
            p, triple = best_choice
            if triple is not None:
                self._assign_patient(p, *triple)
            pending = [q for q in pending if q != p and not self.assignment[q].admitted]
        self._refresh_unassigned_sets()

    def repair_proxy_aware(self, removed: list[int]) -> None:
        ordered = self._ordered_repair_patients(removed)
        top_k = self._effective_top_k()
        for p in ordered:
            if self.assignment[p].admitted:
                continue
            triples = self._iter_candidate_triples_proxy_aware(p)
            triple = self._pick_from_top_k(triples, top_k, self.config.proxy_pick_random_prob)
            if triple is not None:
                self._assign_patient(p, *triple)
        self._refresh_unassigned_sets()

    def _ordered_repair_patients(self, removed: list[int]) -> list[int]:
        inst = self.instance
        removed = list(dict.fromkeys(removed))
        missing_mandatory = [p for p in inst.PM if not self.assignment[p].admitted]
        removed_mandatory = [p for p in removed if p in inst.PM and not self.assignment[p].admitted]
        removed_optional = [p for p in removed if p in inst.PO and not self.assignment[p].admitted]
        extra_missing_optional = [
            p for p in inst.PO
            if (not self.assignment[p].admitted) and (p not in set(removed))
        ]
        missing_mandatory = sorted(missing_mandatory, key=self._repair_mandatory_key)
        removed_mandatory = sorted(removed_mandatory, key=self._repair_mandatory_key)
        removed_optional = sorted(removed_optional, key=self._difficulty_key)
        extra_missing_optional = sorted(extra_missing_optional[:30], key=self._difficulty_key)
        out: list[int] = []
        for bucket in (missing_mandatory, removed_mandatory, removed_optional, extra_missing_optional):
            for p in bucket:
                if p not in out:
                    out.append(p)
        return out

    def _best_alt_triple_for_patient(self, p: int, *, avoid_triples: set[tuple[int, int, int]] | None = None, prefer_diff_day_ot: bool = False):
        avoid_triples = set() if avoid_triples is None else avoid_triples
        triples = self._iter_candidate_triples_weighted(p)
        if not triples:
            return None
        chosen = []
        current = self.assignment[p]
        current_day = current.day
        current_ot = current.ot
        for _, triple in triples:
            if triple in avoid_triples:
                continue
            if prefer_diff_day_ot and current_day is not None and current_ot is not None and triple[0] == current_day and triple[2] == current_ot:
                continue
            chosen.append(triple)
            if len(chosen) >= max(6, self._effective_top_k() + 2):
                break
        if not chosen:
            return None
        k = min(len(chosen), self._effective_top_k())
        if k <= 1:
            return chosen[0]
        if self.alns_random.random() < self.config.greedy_pick_random_prob:
            return chosen[self.alns_random.randrange(k)]
        return chosen[0]

    def micro_relocate(self) -> bool:
        p = self._choose_costly_admitted(top_n=14, optional_bias=True)
        if p is None:
            return False
        a = self.assignment[p]
        if a.day is None or a.room is None or a.ot is None:
            return False
        old = (a.day, a.room, a.ot)
        snap = self._snapshot_state()
        self._remove_patient(p)
        self._refresh_unassigned_sets()
        triple = self._best_alt_triple_for_patient(p, avoid_triples={old}, prefer_diff_day_ot=False)
        if triple is None:
            self._restore_state(snap)
            self._refresh_unassigned_sets()
            return False
        self._assign_patient(p, *triple)
        self._refresh_unassigned_sets()
        return triple != old

    def micro_pair_swap(self) -> bool:
        admitted = self.admitted_incoming()
        if len(admitted) < 2:
            return False
        p = self._choose_costly_admitted(top_n=12, optional_bias=False)
        if p is None:
            return False
        pa = self.assignment[p]
        if pa.day is None or pa.room is None or pa.ot is None:
            return False
        candidates = []
        for q in admitted:
            if q == p:
                continue
            qa = self.assignment[q]
            if qa.day is None or qa.room is None or qa.ot is None:
                continue
            same_day = 1 if qa.day == pa.day else 0
            same_ot = 1 if qa.ot == pa.ot else 0
            same_room = 1 if qa.room == pa.room else 0
            score = 3.0 * same_day + 2.0 * same_ot + 1.5 * same_room + 0.01 * self.instance.surgery_duration[q] + (0.3 if q in self.instance.PO else 0.0)
            candidates.append((score, q))
        if not candidates:
            return False
        candidates.sort(key=lambda item: -item[0])
        q = self.alns_random.choice([qq for _, qq in candidates[: min(10, len(candidates))]])
        qa = self.assignment[q]
        old_p = (pa.day, pa.room, pa.ot)
        old_q = (qa.day, qa.room, qa.ot)
        snap = self._snapshot_state()
        self._remove_patient(p)
        self._remove_patient(q)
        self._refresh_unassigned_sets()
        patterns = [
            (old_q, old_p),
            ((pa.day, qa.room, pa.ot), (qa.day, pa.room, qa.ot)),
            ((qa.day, pa.room, qa.ot), (pa.day, qa.room, pa.ot)),
        ]
        for tp, tq in patterns:
            if not self._can_assign(p, tp[0], tp[1], tp[2]):
                continue
            self._assign_patient(p, *tp)
            if self._can_assign(q, tq[0], tq[1], tq[2]):
                self._assign_patient(q, *tq)
                self._refresh_unassigned_sets()
                if self.assignment[p].admitted and self.assignment[q].admitted:
                    changed = (tp != old_p) or (tq != old_q)
                    if changed:
                        return True
            self._restore_state(snap)
            self._refresh_unassigned_sets()
        self._restore_state(snap)
        self._refresh_unassigned_sets()
        return False

    def micro_optional_move(self) -> bool:
        missing_optional = [p for p in self.instance.PO if not self.assignment[p].admitted]
        admitted_optional = [p for p in self.instance.PO if self.assignment[p].admitted]
        do_insert = bool(missing_optional) and (not admitted_optional or self.alns_random.random() < 0.60)
        if do_insert:
            ranked = sorted(missing_optional, key=self._difficulty_key)
            p = self.alns_random.choice(ranked[: min(12, len(ranked))])
            triple = self._best_alt_triple_for_patient(p)
            if triple is None:
                return False
            self._assign_patient(p, *triple)
            self._refresh_unassigned_sets()
            return True
        if not admitted_optional:
            return False
        ranked = sorted(admitted_optional, key=lambda p: -self._patient_cost_score(p))
        p = self.alns_random.choice(ranked[: min(10, len(ranked))])
        self._remove_patient(p)
        self._refresh_unassigned_sets()
        return True

    def micro_targeted_day_ot(self) -> bool:
        admitted = self.admitted_incoming()
        if not admitted:
            return False
        scored = []
        for p in admitted:
            a = self.assignment[p]
            if a.day is None or a.ot is None:
                continue
            surgeon = self._patient_surgeon(p)
            same_ot_day = sum(1 for q in admitted if self.assignment[q].day == a.day and self.assignment[q].ot == a.ot)
            used_ots = {self.assignment[q].ot for q in admitted if self.assignment[q].day == a.day and self._patient_surgeon(q) == surgeon}
            transfer = max(0, len(used_ots) - 1)
            score = 2.0 * transfer + (1.5 if same_ot_day <= 1 else 0.0) + 0.01 * self.instance.surgery_duration[p]
            scored.append((score, p))
        if not scored:
            return False
        scored.sort(key=lambda item: -item[0])
        p = self.alns_random.choice([pp for _, pp in scored[: min(12, len(scored))]])
        a = self.assignment[p]
        old = (a.day, a.room, a.ot)
        snap = self._snapshot_state()
        self._remove_patient(p)
        self._refresh_unassigned_sets()
        triple = self._best_alt_triple_for_patient(p, avoid_triples={old}, prefer_diff_day_ot=True)
        if triple is None:
            self._restore_state(snap)
            self._refresh_unassigned_sets()
            return False
        self._assign_patient(p, *triple)
        self._refresh_unassigned_sets()
        return triple != old

    def _draw_destroy_size(self, no_improve_iters: int) -> int:
        n = len(self.admitted_incoming())
        if n <= 0:
            return 0
        size_probs = list(self.config.destroy_size_probs)
        if no_improve_iters >= 180:
            size_probs = [0.35, 0.40, 0.25]
        if no_improve_iters >= 260:
            size_probs = [0.20, 0.40, 0.40]
        size_label = self.alns_random.choices(["small", "medium", "large"], weights=size_probs, k=1)[0]
        if size_label == "small":
            lo, hi = self.config.small_frac_range
        elif size_label == "medium":
            lo, hi = self.config.medium_frac_range
        else:
            lo, hi = self.config.large_frac_range
        frac = self.alns_random.uniform(lo, hi)
        k = int(math.ceil(frac * n))
        k = max(self.config.destroy_abs_min, k)
        k = min(self.config.destroy_abs_max, k)
        k = min(k, n)
        return max(1, k)

    def _temperature(self, progress: float, no_improve_iters: int) -> float:
        t0 = max(1e-9, self.config.sa_initial_temperature)
        tf = max(1e-9, self.config.sa_final_temperature)
        progress = min(max(progress, 0.0), 1.0)
        base = t0 * ((tf / t0) ** progress)
        if no_improve_iters >= self.config.reheat_after_no_improve:
            base *= self.config.reheat_multiplier
        return min(base, self.config.max_reheated_temperature)

    def _accept(self, current_obj: float, cand_obj: float, progress: float, no_improve_iters: int) -> bool:
        # criterio di accettazione simulated-annealing: migliora sempre, peggiora con probabilità decrescente
        delta = cand_obj - current_obj
        if delta < -1e-9:
            return True
        if abs(delta) <= 1e-9:
            return self.alns_random.random() < self.config.equal_move_accept_prob
        if not self.config.accept_worse:
            return False
        scale = max(self.config.sa_scale_floor, self.config.sa_scale_fraction * max(1.0, current_obj))
        delta_scaled = delta / scale
        if delta_scaled <= self.config.scaled_delta_small_threshold:
            return self.alns_random.random() < 0.60
        if delta_scaled > self.config.scaled_delta_hard_reject:
            return False
        temp = self._temperature(progress, no_improve_iters)
        prob = math.exp(-delta_scaled / max(temp, 1e-9))
        return self.alns_random.random() < prob

    def run_alns(self, max_seconds: float) -> dict[str, int | float]:
        start = time.time()
        self.alns_random.seed(self.config.seed + 40007)
        self._refresh_unassigned_sets()
        valid, errors = self.validate()
        # deve partire da una soluzione già feasible, altrimenti ALNS non ha senso
        if (not valid) or self.summary()["mandatory_unassigned"] > 0:
            raise RuntimeError(
                "ALNS must start from a feasible Phase-1 solution with all mandatory admitted. "
                f"Validation errors: {errors[:5]}"
            )
        best_state = self._snapshot_state()
        best_obj = self.objective_value()
        current_state = self._snapshot_state()
        current_obj = best_obj
        no_improve_iters = 0
        destroy_ops = [self.destroy_random, self.destroy_expensive, self.destroy_bottleneck_day_ot, self.destroy_related]
        repair_ops = [self.repair_greedy_randomized, self.repair_regret2_randomized, self.repair_proxy_aware]
        micro_ops = [self.micro_relocate, self.micro_pair_swap, self.micro_optional_move, self.micro_targeted_day_ot]
        stats: dict[str, int | float] = {
            "iterations": 0,
            "accepted": 0,
            "improved": 0,
            "rejected_infeasible": 0,
            "rejected_objective": 0,
            "resets_to_best": 0,
            "best_obj": best_obj,
            "micro_used": 0,
            "macro_used": 0,
        }
        while time.time() - start < max_seconds:
            self._restore_state(current_state)
            family = self.alns_random.choices(["macro", "micro"], weights=self.config.family_weights, k=1)[0]
            changed = False
            if family == "macro":
                stats["macro_used"] = int(stats["macro_used"]) + 1
                k = self._draw_destroy_size(no_improve_iters)
                destroy_fun = self.alns_random.choices(destroy_ops, weights=self.config.destroy_weights, k=1)[0]
                repair_fun = self.alns_random.choices(repair_ops, weights=self.config.repair_weights, k=1)[0]
                removed = destroy_fun(k)
                if removed:
                    repair_fun(removed)
                    changed = True
            else:
                stats["micro_used"] = int(stats["micro_used"]) + 1
                changed = self.alns_random.choices(micro_ops, weights=self.config.micro_weights, k=1)[0]()
            if not changed:
                stats["iterations"] = int(stats["iterations"]) + 1
                no_improve_iters += 1
                continue
            self._refresh_unassigned_sets()
            valid, _ = self.validate()
            if (not valid) or self.summary()["mandatory_unassigned"] > 0:
                self._restore_state(current_state)
                stats["rejected_infeasible"] = int(stats["rejected_infeasible"]) + 1
                stats["iterations"] = int(stats["iterations"]) + 1
                no_improve_iters += 1
                continue
            cand_obj = self.objective_value()
            progress = (time.time() - start) / max(max_seconds, 1e-9)
            if self._accept(current_obj, cand_obj, progress, no_improve_iters):
                current_state = self._snapshot_state()
                current_obj = cand_obj
                stats["accepted"] = int(stats["accepted"]) + 1
                if cand_obj + 1e-9 < best_obj:
                    best_state = self._snapshot_state()  # daje, soluzione migliorata
                    best_obj = cand_obj
                    stats["best_obj"] = best_obj
                    stats["improved"] = int(stats["improved"]) + 1
                    no_improve_iters = 0
                else:
                    no_improve_iters += 1
            else:
                self._restore_state(current_state)
                stats["rejected_objective"] = int(stats["rejected_objective"]) + 1
                no_improve_iters += 1
            if no_improve_iters >= self.config.reset_to_best_after_no_improve or current_obj > best_obj * (1.0 + self.config.relative_drift_limit):
                self._restore_state(best_state)
                current_state = self._snapshot_state()
                current_obj = best_obj
                no_improve_iters = 0
                stats["resets_to_best"] = int(stats["resets_to_best"]) + 1
            stats["iterations"] = int(stats["iterations"]) + 1
        self._restore_state(best_state)
        self._refresh_unassigned_sets()
        return stats

    def apply_mip_start(self, builder) -> None:
        inst = self.instance
        v = builder.variables
        alpha_values: dict[tuple[int, int, int], float] = {}
        y_values: dict[tuple[int, int, int], float] = {}
        gamma_values: dict[tuple[int, int, int], float] = {}
        j_values: dict[tuple[int, int], float] = {}
        w_values: dict[tuple[int, int, int], float] = {}
        t_values: dict[tuple[int, int], float] = {}
        phi_values: dict[tuple[int, int], float] = {}
        mu_values: dict[tuple[int, int], float] = {}
        pi_min_values: dict[tuple[int, int], float] = {}
        pi_max_values: dict[tuple[int, int], float] = {}
        delta_values: dict[int, float] = {}
        surgeon_ots: dict[tuple[int, int], set[int]] = {}
        for p in inst.P:
            a = self.assignment[p]
            if not a.admitted or a.day is None or a.room is None:
                continue
            alpha_values[p, a.room, a.day] = 1.0
            for d in self._stay_days(p, a.day):
                if d not in inst.D:
                    break
                y_values[p, a.room, d] = 1.0
            if p in inst.incoming_patients and a.ot is not None:
                gamma_values[p, a.ot, a.day] = 1.0
                j_values[a.ot, a.day] = 1.0
                surgeon = self._patient_surgeon(p)
                surgeon_ots.setdefault((surgeon, a.day), set()).add(a.ot)
                delta_values[p] = float(max(0, a.day - inst.surgery_release_day[p]))
        for (surgeon, day), ots in surgeon_ots.items():
            for ot in ots:
                w_values[surgeon, ot, day] = 1.0
            t_values[surgeon, day] = float(max(0, len(ots) - 1))
        for r in inst.R:
            for d in inst.D:
                occupancy = self.room_occupancy[r][d]
                if occupancy <= 0:
                    pi_min_values[r, d] = 0.0
                    pi_max_values[r, d] = 0.0
                    continue
                gender = self.room_gender[r][d]
                if gender == "F":
                    phi_values[r, d] = 1.0
                elif gender == "M":
                    mu_values[r, d] = 1.0
                ages = [inst.age_group[p] for p in self.room_patients[r][d]]
                pi_min_values[r, d] = float(min(ages))
                pi_max_values[r, d] = float(max(ages))
        starts = {
            "alpha": alpha_values,
            "y": y_values,
            "gamma": gamma_values,
            "j": j_values,
            "w": w_values,
            "t": t_values,
            "phi": phi_values,
            "mu": mu_values,
            "pi_min": pi_min_values,
            "pi_max": pi_max_values,
            "delta": delta_values,
        }
        for name, values in starts.items():
            if name not in v:
                continue
            vardict = v[name]
            for key, value in values.items():
                if key in vardict:
                    vardict[key].Start = value
        builder.model.update()


def run_phase1_heuristic(instance: Instance, config: Phase1HeuristicConfig) -> Phase1HeuristicSolution | None:
    rng = random.Random(config.seed)
    best_solution: Phase1HeuristicSolution | None = None
    best_key: tuple | None = None
    constructive_deadline = time.perf_counter() + max(0.0, min(config.time_limit * 0.20, 10.0))
    runs = max(1, config.constructive_runs)
    for _ in range(runs):
        sol = Phase1HeuristicSolution(instance, config, randomized=True, random_seed=rng.randint(0, 10**9))
        sol.build()
        valid, _ = sol.validate()
        summ = sol.summary()
        obj = sol.compute_objective()
        key = (summ["mandatory_unassigned"], obj["total_with_proxy"], -summ["admitted_optional"])
        if best_key is None or key < best_key:
            best_key = key
            best_solution = sol
        if time.perf_counter() >= constructive_deadline:
            break
    if best_solution is None:
        return None
    if best_solution.summary()["mandatory_unassigned"] > 0:
        best_solution.repair_feasibility(max_seconds=min(config.repair_seconds, max(0.0, config.time_limit * 0.25)))
    valid, errors = best_solution.validate()
    if (not valid) or best_solution.summary()["mandatory_unassigned"] > 0:
        logger.warning("Phase-1 heuristic did not produce a fully feasible incumbent: %s", errors[:5])
        return None
    constructive_elapsed = max(0.0, time.perf_counter() - (constructive_deadline - max(0.0, min(config.time_limit * 0.20, 10.0))))
    remaining = max(0.0, config.time_limit - constructive_elapsed)
    if remaining > 0.0:
        try:
            best_solution.run_alns(remaining)
        except RuntimeError as exc:
            logger.warning("Phase-1 ALNS skipped: %s", exc)
    return best_solution