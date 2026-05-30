"""Phase 2 — *Full* model.

Takes the admission / room / surgery decisions fixed by Phase 1 and
optimises nurse-to-room assignment together with skill-level and
workload-violation penalties.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from hippo.instance import Instance
from hippo.models.base import BaseModelBuilder

logger = logging.getLogger(__name__)

FIXED_VAR_TOL = 1e-6


class Phase2Builder(BaseModelBuilder):
    """Builder for the Phase-2 (full) MIP."""

    def __init__(
        self,
        instance: Instance,
        fixed_vars: dict[str, Any],
        *,
        relaxation: bool = False,
    ) -> None:
        super().__init__(instance, relaxation=relaxation, name="HIPPO_Phase2")
        self.fixed_vars = fixed_vars
        self._active_nurses_by_shift: dict[int, list[int]] = {}
        self._room_rhs_by_shift: dict[tuple[int, int], float] = {}
        self._room_shifts: list[tuple[int, int]] = []
        self._patient_rooms_by_shift: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self._patients_by_shift: dict[int, list[int]] = {}
        self._z_keys: list[tuple[int, int, int]] = []
        self._x_keys: list[tuple[int, int, int]] = []
        self._eta_keys: list[tuple[int, int]] = []
        self._vio_skill_keys: list[tuple[int, int]] = []
        self._skill_terms_by_patient_shift: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self._workload_coeff_by_patient_shift: dict[tuple[int, int], float] = {}
        self._workload_patients_by_shift: dict[int, list[int]] = {}

    # ================================================================== #
    #  Public API                                                        #
    # ================================================================== #

    def build(self) -> None:
        """Construct the full Phase-2 model."""
        self._prepare_fixed_data()
        self._add_variables()
        self._apply_preprocessing()
        self._set_objective()
        self._add_constraints()
        self.model.update()
        logger.info(
            "Phase-2 model built (%d vars, %d constrs).",
            self.model.NumVars, self.model.NumConstrs,
        )

    # ================================================================== #
    #  Variables                                                         #
    # ================================================================== #

    def _prepare_fixed_data(self) -> None:
        inst = self.instance
        fv = self.fixed_vars

        active_nurses_by_shift = {
            s: [n for n in inst.N if inst.availability[n][s] == 1]
            for s in inst.S
        }

        room_rhs_by_shift: dict[tuple[int, int], float] = {}
        room_shifts: list[tuple[int, int]] = []
        z_keys: list[tuple[int, int, int]] = []
        for r in inst.R:
            for d in inst.D:
                rhs = fv["phi"][r, d].X + fv["mu"][r, d].X
                if rhs <= FIXED_VAR_TOL:
                    continue
                for s in range(3 * d, min(3 * d + 3, inst.n_shifts)):
                    room_rhs_by_shift[r, s] = rhs
                    room_shifts.append((r, s))
                    for n in active_nurses_by_shift[s]:
                        z_keys.append((n, r, s))

        patient_rooms_by_shift: defaultdict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        patients_by_shift_sets: defaultdict[int, set[int]] = defaultdict(set)
        for (p, r, d), var in fv["y"].items():
            y_val = var.X
            if y_val <= FIXED_VAR_TOL:
                continue
            for s in range(3 * d, min(3 * d + 3, inst.n_shifts)):
                patient_rooms_by_shift[p, s].append((r, y_val))
                patients_by_shift_sets[s].add(p)

        x_keys: list[tuple[int, int, int]] = []
        eta_pairs: set[tuple[int, int]] = set()
        vio_skill_keys: list[tuple[int, int]] = []
        for (p, s), rooms in patient_rooms_by_shift.items():
            del rooms
            vio_skill_keys.append((p, s))
            for n in active_nurses_by_shift[s]:
                x_keys.append((p, n, s))
                eta_pairs.add((p, n))

        alpha_by_patient_day: defaultdict[tuple[int, int], float] = defaultdict(float)
        for (p, r, d), var in fv["alpha"].items():
            alpha_val = var.X
            if alpha_val > FIXED_VAR_TOL:
                alpha_by_patient_day[p, d] += alpha_val

        skill_terms_by_patient_shift: defaultdict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        workload_coeff_by_patient_shift: defaultdict[tuple[int, int], float] = defaultdict(float)
        for (p, d), alpha_val in alpha_by_patient_day.items():
            skill_req = inst.skill_level_required[p]
            for rel, req in enumerate(skill_req):
                s = 3 * d + rel
                if s >= inst.n_shifts:
                    break
                skill_terms_by_patient_shift[p, s].append((alpha_val, float(req)))

            workload = inst.workload_produced[p]
            for rel, load in enumerate(workload):
                s = 3 * d + rel
                if s >= inst.n_shifts:
                    break
                workload_coeff_by_patient_shift[p, s] += alpha_val * load

        workload_patients_by_shift: defaultdict[int, list[int]] = defaultdict(list)
        for s, patients in patients_by_shift_sets.items():
            for p in sorted(patients):
                if workload_coeff_by_patient_shift.get((p, s), 0.0) > FIXED_VAR_TOL:
                    workload_patients_by_shift[s].append(p)

        self._active_nurses_by_shift = active_nurses_by_shift
        self._room_rhs_by_shift = room_rhs_by_shift
        self._room_shifts = room_shifts
        self._patient_rooms_by_shift = dict(patient_rooms_by_shift)
        self._patients_by_shift = {s: sorted(patients) for s, patients in patients_by_shift_sets.items()}
        self._z_keys = z_keys
        self._x_keys = x_keys
        self._eta_keys = sorted(eta_pairs)
        self._vio_skill_keys = vio_skill_keys
        self._skill_terms_by_patient_shift = dict(skill_terms_by_patient_shift)
        self._workload_coeff_by_patient_shift = dict(workload_coeff_by_patient_shift)
        self._workload_patients_by_shift = dict(workload_patients_by_shift)

    def _add_variables(self) -> None:
        inst = self.instance
        v = self.variables

        # Nurse-to-room assignment
        v["z"] = self.model.addVars(
            self._z_keys,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="z",
        )
        # Nurse-patient indicator
        v["x"] = self.model.addVars(
            self._x_keys,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, ub=1, name="x",
        )
        # Continuity-of-care indicator
        v["eta"] = self.model.addVars(
            self._eta_keys,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, ub=1, name="eta",
        )
        # Skill-level violation
        v["vio_skill"] = self.model.addVars(
            self._vio_skill_keys,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, name="vio_skill",
        )
        # Workload violation
        v["vio_wl"] = self.model.addVars(
            inst.N, inst.S,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, name="vio_wl",
        )

    # ================================================================== #
    #  Preprocessing                                                     #
    # ================================================================== #

    def _apply_preprocessing(self) -> None:
        """Bounds are already tightened by sparse variable creation."""

    # ================================================================== #
    #  Objective                                                         #
    # ================================================================== #

    def _set_objective(self) -> None:
        inst = self.instance
        W = inst.weights
        v = self.variables

        obj_skill = gp.quicksum(
            v["vio_skill"][p, s] for p, s in self._vio_skill_keys
        )
        obj_continuity = gp.quicksum(
            v["eta"][p, n] for p, n in self._eta_keys
        )
        obj_workload = gp.quicksum(
            v["vio_wl"][n, s] for n in inst.N for s in inst.S
        )

        self.model.setObjective(
            W["W2"] * obj_skill
            + W["W3"] * obj_continuity
            + W["W4"] * obj_workload,
            GRB.MINIMIZE,
        )

    # ================================================================== #
    #  Constraints                                                       #
    # ================================================================== #

    def _add_constraints(self) -> None:
        self._cstr_nurse_to_room()
        self._cstr_nurse_patient_consistency()
        self._cstr_continuity_of_care()
        self._cstr_skill_violation()
        self._cstr_workload_violation()

    def _cstr_nurse_to_room(self) -> None:
        """Each occupied room must have exactly one nurse per shift."""
        v = self.variables

        self.model.addConstrs(
            (
                gp.quicksum(v["z"][n, r, s] for n in self._active_nurses_by_shift[s])
                == self._room_rhs_by_shift[r, s]
                for r, s in self._room_shifts
            ),
            name="nurse_to_room",
        )

    def _cstr_nurse_patient_consistency(self) -> None:
        """Link x[p,n,s] to z[n,r,s] and fixed y[p,r,d]."""
        v = self.variables

        self.model.addConstrs(
            (
                v["x"][p, n, s] >= y_val * v["z"][n, r, s]
                for p, s in self._patient_rooms_by_shift
                for r, y_val in self._patient_rooms_by_shift[p, s]
                for n in self._active_nurses_by_shift[s]
            ),
            name="x_consistency",
        )

    def _cstr_continuity_of_care(self) -> None:
        """eta[p,n] >= x[p,n,s] for all s."""
        v = self.variables

        self.model.addConstrs(
            (v["eta"][p, n] >= v["x"][p, n, s] for p, n, s in self._x_keys),
            name="eta_lb",
        )

    def _cstr_skill_violation(self) -> None:
        """Skill-level violation linearisation using fixed alpha."""
        inst = self.instance
        v = self.variables

        self.model.addConstrs(
            (
                v["vio_skill"][p, s]
                >= (skill_req * alpha_val - inst.skill_level_nurse[n]) * v["x"][p, n, s]
                for p, n, s in self._x_keys
                for alpha_val, skill_req in self._skill_terms_by_patient_shift.get((p, s), ())
            ),
            name="skill_vio",
        )

    def _cstr_workload_violation(self) -> None:
        """Workload violation: excess over max_load."""
        inst = self.instance
        v = self.variables

        self.model.addConstrs(
            (
                v["vio_wl"][n, s]
                >= gp.quicksum(
                    self._workload_coeff_by_patient_shift[p, s] * v["x"][p, n, s]
                    for p in self._workload_patients_by_shift.get(s, ())
                    if (p, n, s) in v["x"]
                )
                - inst.max_load[n][s]
                for n in inst.N
                for s in inst.S
            ),
            name="workload_vio",
        )
