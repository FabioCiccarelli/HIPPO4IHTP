"""Greedy + ALNS warm start for HIPPO Phase 2."""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hippo.models.phase2 import Phase2Builder

logger = logging.getLogger(__name__)

FIXED_VAR_TOL = 1e-6


def _value(obj: Any) -> float:
    return float(obj.X) if hasattr(obj, "X") else float(obj)


@dataclass
class Phase2HeuristicConfig:
    time_limit: float = 60.0
    seed: int = 1234
    family_weights: tuple[float, float] = (0.60, 0.40)
    destroy_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)
    greedy_random_prob: float = 0.20
    repair_top_k: int = 3
    destroy_frac_range: tuple[float, float] = (0.05, 0.12)
    destroy_abs_min: int = 2
    destroy_abs_max: int = 18
    accept_worse: bool = True
    sa_initial_temperature: float = 0.30
    sa_final_temperature: float = 0.04
    equal_move_accept_prob: float = 0.40
    reheat_after_no_improve: int = 120
    reheat_multiplier: float = 1.8
    max_reheated_temperature: float = 0.70
    sa_scale_fraction: float = 0.02
    sa_scale_floor: float = 25.0
    scaled_delta_small_threshold: float = 0.10
    scaled_delta_hard_reject: float = 2.50
    reset_to_best_after_no_improve: int = 220
    relative_drift_limit: float = 0.08


class Phase2HeuristicSolution:
    def __init__(self, builder: Phase2Builder, config: Phase2HeuristicConfig) -> None:
        self.builder = builder
        self.instance = builder.instance
        self.config = config
        self.random = random.Random(config.seed)
        self.assignment: dict[tuple[int, int], int] = {}

        self.room_patients_by_shift: dict[tuple[int, int], list[int]] = defaultdict(list)
        for (p, s), rooms in builder._patient_rooms_by_shift.items():
            for r, y_val in rooms:
                if y_val > FIXED_VAR_TOL:
                    self.room_patients_by_shift[r, s].append(p)
        self.room_patients_by_shift = {
            key: sorted(values) for key, values in self.room_patients_by_shift.items()
        }

        self.skill_requirement: dict[tuple[int, int], float] = {}
        for key, terms in builder._skill_terms_by_patient_shift.items():
            if not terms:
                continue
            self.skill_requirement[key] = max(alpha_val * req for alpha_val, req in terms)

        self.room_workload: dict[tuple[int, int], float] = {}
        self.room_skill_violation: dict[tuple[int, int], float] = {}
        for room_shift, patients in self.room_patients_by_shift.items():
            _, s = room_shift
            self.room_workload[room_shift] = sum(
                builder._workload_coeff_by_patient_shift.get((p, s), 0.0) for p in patients
            )

        self.nurse_load: dict[int, dict[int, float]] = {
            n: {s: 0.0 for s in self.instance.S}
            for n in self.instance.N
        }
        self.patient_nurse_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.patient_shift_nurse: dict[tuple[int, int], int] = {}
        self.skill_violation_by_patient_shift: dict[tuple[int, int], float] = defaultdict(float)
        self.workload_violation_by_nurse_shift: dict[tuple[int, int], float] = defaultdict(float)
        self.skill_total = 0.0
        self.continuity_total = 0.0
        self.workload_total = 0.0

    def _reset_caches(self) -> None:
        self.nurse_load = {
            n: {s: 0.0 for s in self.instance.S}
            for n in self.instance.N
        }
        self.patient_nurse_counts = defaultdict(Counter)
        self.patient_shift_nurse = {}
        self.skill_violation_by_patient_shift = defaultdict(float)
        self.workload_violation_by_nurse_shift = defaultdict(float)
        self.skill_total = 0.0
        self.continuity_total = 0.0
        self.workload_total = 0.0

    def _rebuild_from_assignment(self) -> None:
        assigned = dict(self.assignment)
        self._reset_caches()
        self.assignment = {}
        for room_shift in sorted(assigned, key=self._room_shift_order_key):
            self._assign(room_shift, assigned[room_shift])

    def objective_value(self) -> float:
        self._rebuild_from_assignment()
        w = self.instance.weights
        return (
            w["W2"] * self.skill_total
            + w["W3"] * self.continuity_total
            + w["W4"] * self.workload_total
        )

    def summary(self) -> dict[str, float]:
        self._rebuild_from_assignment()
        return {
            "assigned_room_shifts": float(len(self.assignment)),
            "skill_total": self.skill_total,
            "continuity_total": self.continuity_total,
            "workload_total": self.workload_total,
            "objective": self.objective_value(),
        }

    def _room_shift_order_key(self, room_shift: tuple[int, int]) -> tuple:
        r, s = room_shift
        patients = self.room_patients_by_shift.get(room_shift, [])
        total_skill = sum(self.skill_requirement.get((p, s), 0.0) for p in patients)
        total_wl = self.room_workload.get(room_shift, 0.0)
        return (-total_skill, -total_wl, -len(patients), s, r)

    def _active_nurses(self, s: int) -> list[int]:
        return self.builder._active_nurses_by_shift.get(s, [])

    def _room_shift_skill_cost(self, room_shift: tuple[int, int], nurse: int) -> float:
        _, s = room_shift
        return sum(
            max(0.0, self.skill_requirement.get((p, s), 0.0) - self.instance.skill_level_nurse[nurse])
            for p in self.room_patients_by_shift.get(room_shift, [])
        )

    def _room_shift_continuity_delta(self, room_shift: tuple[int, int], nurse: int) -> float:
        patients = self.room_patients_by_shift.get(room_shift, [])
        return float(sum(1 for p in patients if self.patient_nurse_counts[p].get(nurse, 0) == 0))

    def _room_shift_workload_delta(self, room_shift: tuple[int, int], nurse: int) -> float:
        _, s = room_shift
        current = self.nurse_load[nurse][s]
        add = self.room_workload.get(room_shift, 0.0)
        old_vio = max(0.0, current - self.instance.max_load[nurse][s])
        new_vio = max(0.0, current + add - self.instance.max_load[nurse][s])
        return new_vio - old_vio

    def _incremental_cost(self, room_shift: tuple[int, int], nurse: int, *, continuity_bias: bool = False) -> tuple[float, float, float, float, int]:
        skill = self._room_shift_skill_cost(room_shift, nurse)
        continuity = self._room_shift_continuity_delta(room_shift, nurse)
        workload = self._room_shift_workload_delta(room_shift, nurse)
        weight = self.instance.weights
        total = weight["W2"] * skill + weight["W3"] * continuity + weight["W4"] * workload
        if continuity_bias:
            total -= 0.05 * weight["W3"] * continuity
        return (total, continuity, workload, skill, nurse)

    def _assign(self, room_shift: tuple[int, int], nurse: int) -> None:
        if room_shift in self.assignment:
            raise ValueError(f"Room shift {room_shift} already assigned")
        _, s = room_shift
        patients = self.room_patients_by_shift.get(room_shift, [])
        skill_cost = self._room_shift_skill_cost(room_shift, nurse)
        continuity_delta = self._room_shift_continuity_delta(room_shift, nurse)
        workload_delta = self._room_shift_workload_delta(room_shift, nurse)

        self.skill_total += skill_cost
        self.continuity_total += continuity_delta
        self.workload_total += workload_delta
        self.assignment[room_shift] = nurse
        self.nurse_load[nurse][s] += self.room_workload.get(room_shift, 0.0)
        self.workload_violation_by_nurse_shift[nurse, s] = max(
            0.0,
            self.nurse_load[nurse][s] - self.instance.max_load[nurse][s],
        )
        for p in patients:
            self.patient_nurse_counts[p][nurse] += 1
            self.patient_shift_nurse[p, s] = nurse
            vio = max(0.0, self.skill_requirement.get((p, s), 0.0) - self.instance.skill_level_nurse[nurse])
            self.skill_violation_by_patient_shift[p, s] = vio

    def _unassign(self, room_shift: tuple[int, int]) -> None:
        nurse = self.assignment.pop(room_shift)
        _, s = room_shift
        patients = self.room_patients_by_shift.get(room_shift, [])
        skill_cost = self._room_shift_skill_cost(room_shift, nurse)
        continuity_delta = float(sum(1 for p in patients if self.patient_nurse_counts[p].get(nurse, 0) == 1))
        workload_before = self.workload_violation_by_nurse_shift.get((nurse, s), 0.0)
        self.nurse_load[nurse][s] -= self.room_workload.get(room_shift, 0.0)
        workload_after = max(0.0, self.nurse_load[nurse][s] - self.instance.max_load[nurse][s])
        workload_delta = workload_before - workload_after

        self.skill_total -= skill_cost
        self.continuity_total -= continuity_delta
        self.workload_total -= workload_delta
        self.workload_violation_by_nurse_shift[nurse, s] = workload_after
        for p in patients:
            self.patient_nurse_counts[p][nurse] -= 1
            if self.patient_nurse_counts[p][nurse] <= 0:
                del self.patient_nurse_counts[p][nurse]
            self.patient_shift_nurse.pop((p, s), None)
            self.skill_violation_by_patient_shift.pop((p, s), None)

    def validate(self) -> tuple[bool, list[str]]:
        self._rebuild_from_assignment()
        errors: list[str] = []
        for room_shift in self.builder._room_shifts:
            if room_shift not in self.assignment:
                errors.append(f"Missing nurse assignment for room shift {room_shift}")
                continue
            nurse = self.assignment[room_shift]
            _, s = room_shift
            if nurse not in self._active_nurses(s):
                errors.append(f"Assigned unavailable nurse {nurse} to room shift {room_shift}")
        return (len(errors) == 0, errors)

    def build_greedy(self) -> bool:
        room_shifts = sorted(self.builder._room_shifts, key=self._room_shift_order_key)
        for room_shift in room_shifts:
            _, s = room_shift
            candidates = [
                self._incremental_cost(room_shift, nurse, continuity_bias=True)
                for nurse in self._active_nurses(s)
            ]
            if not candidates:
                return False
            candidates.sort()
            top_k = min(self.config.repair_top_k, len(candidates))
            if top_k > 1 and self.random.random() < self.config.greedy_random_prob:
                chosen = candidates[self.random.randrange(top_k)][-1]
            else:
                chosen = candidates[0][-1]
            self._assign(room_shift, chosen)
        self._rebuild_from_assignment()
        return True

    def _snapshot_state(self) -> dict[str, Any]:
        return {"assignment": copy.deepcopy(self.assignment)}

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.assignment = state["assignment"]
        self._rebuild_from_assignment()

    def _draw_destroy_size(self) -> int:
        n = len(self.assignment)
        if n <= 0:
            return 0
        lo, hi = self.config.destroy_frac_range
        frac = self.random.uniform(lo, hi)
        k = int(math.ceil(frac * n))
        k = max(self.config.destroy_abs_min, k)
        k = min(self.config.destroy_abs_max, k)
        return min(max(1, k), n)

    def destroy_random(self, k: int) -> list[tuple[int, int]]:
        keys = list(self.assignment)
        self.random.shuffle(keys)
        removed = keys[:k]
        for room_shift in removed:
            self._unassign(room_shift)
        return removed

    def destroy_overloaded(self, k: int) -> list[tuple[int, int]]:
        scores: list[tuple[float, tuple[int, int]]] = []
        for room_shift, nurse in self.assignment.items():
            _, s = room_shift
            patients = self.room_patients_by_shift.get(room_shift, [])
            continuity = sum(1 for p in patients if self.patient_nurse_counts[p].get(nurse, 0) == 1)
            overload = self.workload_violation_by_nurse_shift.get((nurse, s), 0.0)
            skill = self._room_shift_skill_cost(room_shift, nurse)
            score = self.instance.weights["W4"] * overload + self.instance.weights["W2"] * skill + self.instance.weights["W3"] * continuity
            scores.append((score, room_shift))
        scores.sort(key=lambda item: item[0], reverse=True)
        removed = [room_shift for _, room_shift in scores[:k]]
        for room_shift in removed:
            self._unassign(room_shift)
        return removed

    def destroy_continuity_conflict(self, k: int) -> list[tuple[int, int]]:
        conflict_patients = sorted(
            self.patient_nurse_counts,
            key=lambda p: (len(self.patient_nurse_counts[p]), p),
            reverse=True,
        )
        target_patients = set(conflict_patients[: max(1, min(10, len(conflict_patients)))])
        candidates = [
            room_shift for room_shift in self.assignment
            if any(p in target_patients for p in self.room_patients_by_shift.get(room_shift, ()))
        ]
        if len(candidates) < k:
            extra = [room_shift for room_shift in self.assignment if room_shift not in candidates]
            self.random.shuffle(extra)
            candidates.extend(extra)
        removed = candidates[:k]
        for room_shift in removed:
            self._unassign(room_shift)
        return removed

    def _repair_order(self, removed: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted(removed, key=self._room_shift_order_key)

    def repair_greedy(self, removed: list[tuple[int, int]], *, continuity_bias: bool) -> None:
        for room_shift in self._repair_order(removed):
            _, s = room_shift
            candidates = [
                self._incremental_cost(room_shift, nurse, continuity_bias=continuity_bias)
                for nurse in self._active_nurses(s)
            ]
            if not candidates:
                continue
            candidates.sort()
            top_k = min(self.config.repair_top_k, len(candidates))
            if top_k > 1 and self.random.random() < self.config.greedy_random_prob:
                chosen = candidates[self.random.randrange(top_k)][-1]
            else:
                chosen = candidates[0][-1]
            self._assign(room_shift, chosen)

    def micro_reassign(self) -> bool:
        if not self.assignment:
            return False
        room_shift = self.random.choice(list(self.assignment))
        old_obj = self.objective_value()
        current_nurse = self.assignment[room_shift]
        _, s = room_shift
        candidates = [n for n in self._active_nurses(s) if n != current_nurse]
        if not candidates:
            return False
        snap = self._snapshot_state()
        self._unassign(room_shift)
        costs = [self._incremental_cost(room_shift, nurse, continuity_bias=True) for nurse in candidates]
        costs.sort()
        chosen = costs[0][-1]
        self._assign(room_shift, chosen)
        if self.objective_value() + 1e-9 < old_obj:
            return True
        if chosen != current_nurse:
            return True
        self._restore_state(snap)
        return False

    def micro_swap(self) -> bool:
        if len(self.assignment) < 2:
            return False
        same_shift_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for room_shift in self.assignment:
            same_shift_groups[room_shift[1]].append(room_shift)
        candidate_shifts = [s for s, room_shifts in same_shift_groups.items() if len(room_shifts) >= 2]
        if not candidate_shifts:
            return False
        s = self.random.choice(candidate_shifts)
        room_shifts = same_shift_groups[s]
        first, second = self.random.sample(room_shifts, 2)
        n1 = self.assignment[first]
        n2 = self.assignment[second]
        if n1 == n2:
            return False
        old_obj = self.objective_value()
        snap = self._snapshot_state()
        self._unassign(first)
        self._unassign(second)
        self._assign(first, n2)
        self._assign(second, n1)
        if self.validate()[0] and self.objective_value() <= old_obj + 1e-9:
            return True
        self._restore_state(snap)
        return False

    def _temperature(self, progress: float, no_improve_iters: int) -> float:
        t0 = max(1e-9, self.config.sa_initial_temperature)
        tf = max(1e-9, self.config.sa_final_temperature)
        base = t0 * ((tf / t0) ** min(max(progress, 0.0), 1.0))
        if no_improve_iters >= self.config.reheat_after_no_improve:
            base *= self.config.reheat_multiplier
        return min(base, self.config.max_reheated_temperature)

    def _accept(self, current_obj: float, cand_obj: float, progress: float, no_improve_iters: int) -> bool:
        delta = cand_obj - current_obj
        if delta < -1e-9:
            return True
        if abs(delta) <= 1e-9:
            return self.random.random() < self.config.equal_move_accept_prob
        if not self.config.accept_worse:
            return False
        scale = max(self.config.sa_scale_floor, self.config.sa_scale_fraction * max(1.0, current_obj))
        scaled = delta / scale
        if scaled <= self.config.scaled_delta_small_threshold:
            return self.random.random() < 0.55
        if scaled > self.config.scaled_delta_hard_reject:
            return False
        temp = self._temperature(progress, no_improve_iters)
        return self.random.random() < math.exp(-scaled / max(temp, 1e-9))

    def run_alns(self, max_seconds: float) -> dict[str, float | int]:
        valid, errors = self.validate()
        if not valid:
            raise RuntimeError(f"Phase-2 heuristic needs a complete initial assignment: {errors[:5]}")

        start = time.time()
        best_state = self._snapshot_state()
        best_obj = self.objective_value()
        current_state = self._snapshot_state()
        current_obj = best_obj
        no_improve_iters = 0
        stats: dict[str, float | int] = {
            "iterations": 0,
            "accepted": 0,
            "improved": 0,
            "rejected_infeasible": 0,
            "rejected_objective": 0,
            "resets_to_best": 0,
            "best_obj": best_obj,
            "macro_used": 0,
            "micro_used": 0,
        }
        destroy_ops = [self.destroy_random, self.destroy_overloaded, self.destroy_continuity_conflict]
        repair_ops = [
            lambda removed: self.repair_greedy(removed, continuity_bias=False),
            lambda removed: self.repair_greedy(removed, continuity_bias=True),
        ]
        micro_ops = [self.micro_reassign, self.micro_swap]

        while time.time() - start < max_seconds:
            self._restore_state(current_state)
            family = self.random.choices(["macro", "micro"], weights=self.config.family_weights, k=1)[0]
            changed = False
            if family == "macro":
                stats["macro_used"] = int(stats["macro_used"]) + 1
                k = self._draw_destroy_size()
                destroy_fun = self.random.choices(destroy_ops, weights=self.config.destroy_weights, k=1)[0]
                repair_fun = self.random.choice(repair_ops)
                removed = destroy_fun(k)
                if removed:
                    repair_fun(removed)
                    changed = True
            else:
                stats["micro_used"] = int(stats["micro_used"]) + 1
                changed = self.random.choice(micro_ops)()
            if not changed:
                stats["iterations"] = int(stats["iterations"]) + 1
                no_improve_iters += 1
                continue
            valid, _ = self.validate()
            if not valid:
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
                    best_state = self._snapshot_state()
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
        return stats

    def apply_mip_start(self) -> None:
        self._rebuild_from_assignment()
        v = self.builder.variables
        z_values: dict[tuple[int, int, int], float] = {}
        x_values: dict[tuple[int, int, int], float] = {}
        eta_values: dict[tuple[int, int], float] = {}
        vio_skill_values: dict[tuple[int, int], float] = {}
        vio_wl_values: dict[tuple[int, int], float] = {}

        for room_shift, nurse in self.assignment.items():
            r, s = room_shift
            z_values[nurse, r, s] = 1.0
            for p in self.room_patients_by_shift.get(room_shift, []):
                x_values[p, nurse, s] = 1.0
                eta_values[p, nurse] = 1.0
                vio_skill_values[p, s] = max(
                    vio_skill_values.get((p, s), 0.0),
                    max(0.0, self.skill_requirement.get((p, s), 0.0) - self.instance.skill_level_nurse[nurse]),
                )
        for n in self.instance.N:
            for s in self.instance.S:
                value = max(0.0, self.nurse_load[n][s] - self.instance.max_load[n][s])
                if value > 0.0:
                    vio_wl_values[n, s] = value

        starts = {
            "z": z_values,
            "x": x_values,
            "eta": eta_values,
            "vio_skill": vio_skill_values,
            "vio_wl": vio_wl_values,
        }
        for name, values in starts.items():
            if name not in v:
                continue
            vardict = v[name]
            for key, value in values.items():
                if key in vardict:
                    vardict[key].Start = value
        self.builder.model.update()


def run_phase2_heuristic(builder: Phase2Builder, config: Phase2HeuristicConfig) -> Phase2HeuristicSolution | None:
    solution = Phase2HeuristicSolution(builder, config)
    if not solution.build_greedy():
        logger.warning("Phase-2 heuristic could not build an initial assignment")
        return None
    valid, errors = solution.validate()
    if not valid:
        logger.warning("Phase-2 greedy incumbent invalid: %s", errors[:5])
        return None
    if config.time_limit > 0.0:
        try:
            solution.run_alns(config.time_limit)
        except RuntimeError as exc:
            logger.warning("Phase-2 ALNS skipped: %s", exc)
    return solution