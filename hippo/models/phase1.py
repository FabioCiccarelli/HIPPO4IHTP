"""Phase 1 — Light model.

Decides patient admission days, room assignments, and surgery scheduling.
Nurse assignment is deferred to Phase 2.
"""

from __future__ import annotations

from collections import defaultdict
import logging

import gurobipy as gp
from gurobipy import GRB

from hippo.config import ProxyStrategy
from hippo.instance import Instance
from hippo.models.base import BaseModelBuilder

logger = logging.getLogger(__name__)


class Phase1Builder(BaseModelBuilder):

    def __init__(
        self,
        instance: Instance,
        *,
        proxy_strategy: ProxyStrategy = ProxyStrategy.NONE,
        proxy_weight: float = 0.0,
        relaxation: bool = False,
    ) -> None:
        super().__init__(instance, relaxation=relaxation, name="HIPPO_Phase1")
        self.proxy_strategy = proxy_strategy
        self.proxy_weight = proxy_weight
        self._fixed_room: dict[int, int] = {}
        self._compatible_rooms: dict[int, list[int]] = {}
        self._feasible_days: dict[int, list[int]] = {}
        self._alpha_keys: list[tuple[int, int, int]] = []
        self._alpha_keys_by_patient: dict[int, list[tuple[int, int]]] = {}
        self._y_keys: list[tuple[int, int, int]] = []
        self._incoming_y_keys: list[tuple[int, int, int]] = []
        self._patients_by_room_day: dict[tuple[int, int], list[int]] = {}
        self._female_by_room_day: dict[tuple[int, int], list[int]] = {}
        self._male_by_room_day: dict[tuple[int, int], list[int]] = {}
        self._y_pairs_by_day: dict[int, list[tuple[int, int]]] = {}
        self._gamma_keys: list[tuple[int, int, int]] = []
        self._incoming_patients_by_day: dict[int, list[int]] = {}
        self._patient_surgeon: dict[int, int] = {}
        self._surgeon_patients_by_day: dict[tuple[int, int], list[int]] = {}

    def build(self) -> None:
        self._prepare_sparse_indices()
        self._add_variables()
        self._apply_preprocessing()
        self._set_objective()
        self._add_constraints()
        self.model.update()
        logger.info("Phase-1 model built (%d vars, %d constrs).",
                     self.model.NumVars, self.model.NumConstrs)

    def _prepare_sparse_indices(self) -> None:
        inst = self.instance

        alpha_keys: list[tuple[int, int, int]] = []
        alpha_keys_by_patient: dict[int, list[tuple[int, int]]] = {p: [] for p in inst.P}
        y_keys: list[tuple[int, int, int]] = []
        incoming_y_keys: list[tuple[int, int, int]] = []
        patients_by_room_day: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        female_by_room_day: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        male_by_room_day: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        y_pairs_by_day: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        gamma_keys: list[tuple[int, int, int]] = []
        incoming_patients_by_day: defaultdict[int, list[int]] = defaultdict(list)
        patient_surgeon: dict[int, int] = {}
        surgeon_patients_by_day: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        compatible_rooms: dict[int, list[int]] = {}
        feasible_days: dict[int, list[int]] = {}
        fixed_room: dict[int, int] = {}

        female_patients = set(inst.Pfemale)

        for p in inst.PA:
            r_fixed = next(r for r, val in inst.room_occupant[p].items() if val == 1)
            fixed_room[p] = r_fixed
            compatible_rooms[p] = [r_fixed]
            feasible_days[p] = [0]

            alpha_keys.append((p, r_fixed, 0))
            alpha_keys_by_patient[p].append((r_fixed, 0))

            for d in range(min(inst.length_of_stay[p], inst.n_days)):
                y_keys.append((p, r_fixed, d))
                patients_by_room_day[r_fixed, d].append(p)
                y_pairs_by_day[d].append((p, r_fixed))
                if p in female_patients:
                    female_by_room_day[r_fixed, d].append(p)
                else:
                    male_by_room_day[r_fixed, d].append(p)

        for p in inst.incoming_patients:
            rooms = [r for r in inst.R if inst.room_compatible[p][r] == 1]
            compatible_rooms[p] = rooms

            days = list(range(inst.surgery_release_day[p], inst.surgery_due_day[p] + 1))
            feasible_days[p] = days
            if not days:
                continue

            surgeon = next((c for c, val in inst.surgeon_assignment[p].items() if val == 1), None)
            if surgeon is not None:
                patient_surgeon[p] = surgeon

            if rooms:
                for d in days:
                    incoming_patients_by_day[d].append(p)
                    if surgeon is not None:
                        surgeon_patients_by_day[surgeon, d].append(p)
                    for o in inst.OT:
                        gamma_keys.append((p, o, d))

            los = inst.length_of_stay[p]
            first_day = days[0]
            last_day = days[-1]

            for r in rooms:
                for d in days:
                    alpha_keys.append((p, r, d))
                    alpha_keys_by_patient[p].append((r, d))

                for d in inst.D:
                    first_possible = max(first_day, d - los + 1)
                    last_possible = min(last_day, d)
                    if first_possible > last_possible:
                        continue

                    y_keys.append((p, r, d))
                    incoming_y_keys.append((p, r, d))
                    patients_by_room_day[r, d].append(p)
                    y_pairs_by_day[d].append((p, r))
                    if p in female_patients:
                        female_by_room_day[r, d].append(p)
                    else:
                        male_by_room_day[r, d].append(p)

        self._fixed_room = fixed_room
        self._compatible_rooms = compatible_rooms
        self._feasible_days = feasible_days
        self._alpha_keys = alpha_keys
        self._alpha_keys_by_patient = alpha_keys_by_patient
        self._y_keys = y_keys
        self._incoming_y_keys = incoming_y_keys
        self._patients_by_room_day = dict(patients_by_room_day)
        self._female_by_room_day = dict(female_by_room_day)
        self._male_by_room_day = dict(male_by_room_day)
        self._y_pairs_by_day = dict(y_pairs_by_day)
        self._gamma_keys = gamma_keys
        self._incoming_patients_by_day = dict(incoming_patients_by_day)
        self._patient_surgeon = patient_surgeon
        self._surgeon_patients_by_day = dict(surgeon_patients_by_day)

    def _add_variables(self) -> None:
        inst = self.instance
        v = self.variables

        # Admission
        v["alpha"] = self.model.addVars(
            self._alpha_keys,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="alpha",
        )
        v["delta"] = self.model.addVars(
            inst.incoming_patients,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, name="delta",
        )

        # Room occupancy
        v["y"] = self.model.addVars(
            self._y_keys,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, ub=1, name="y",
        )

        # Room characteristics
        v["pi_max"] = self.model.addVars(
            inst.R, inst.D,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, ub=inst.max_age_group, name="pi_max",
        )
        v["pi_min"] = self.model.addVars(
            inst.R, inst.D,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, ub=inst.max_age_group, name="pi_min",
        )
        v["phi"] = self.model.addVars(
            inst.R, inst.D,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="phi",
        )
        v["mu"] = self.model.addVars(
            inst.R, inst.D,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="mu",
        )

        # Surgery scheduling
        v["gamma"] = self.model.addVars(
            self._gamma_keys,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="gamma",
        )
        v["j"] = self.model.addVars(
            inst.OT, inst.D,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="j",
        )
        v["w"] = self.model.addVars(
            inst.C, inst.OT, inst.D,
            vtype=self._vtype(GRB.BINARY), lb=0, ub=1, name="w",
        )
        v["t"] = self.model.addVars(
            inst.C, inst.D,
            vtype=self._vtype(GRB.CONTINUOUS), lb=0, name="t",
        )

    def _apply_preprocessing(self) -> None:
        self._fix_occupants()

    def _fix_occupants(self) -> None:
        # pazienti già ricoverati: stanza e giorno sono noti, fissiamo tutto
        inst = self.instance
        v = self.variables
        for p in inst.PA:
            r_fixed = self._fixed_room[p]
            v["alpha"][p, r_fixed, 0].lb = 1
            for d in range(min(inst.length_of_stay[p], inst.n_days)):
                v["y"][p, r_fixed, d].lb = 1

    def _set_objective(self) -> None:
        inst = self.instance
        W = inst.weights
        v = self.variables

        # W1: age mix
        obj_age_mix = gp.quicksum(
            v["pi_max"][r, d] - v["pi_min"][r, d]
            for r in inst.R for d in inst.D
        )
        # W5: open OTs
        obj_open_ot = gp.quicksum(
            v["j"][o, d] for o in inst.OT for d in inst.D
        )
        # W6: surgeon transfers
        obj_transfer = gp.quicksum(
            v["t"][c, d] for c in inst.C for d in inst.D
        )
        # W7: admission delay
        obj_delay = gp.quicksum(v["delta"][p] for p in inst.incoming_patients)
        # W8: unscheduled optional
        obj_unscheduled = len(inst.PO) - gp.quicksum(
            v["alpha"][p, r, d]
            for p in inst.PO
            for r, d in self._alpha_keys_by_patient[p]
        )

        # --- Base objective (without proxy) ---
        base_obj = (
            W["W1"] * obj_age_mix
            + W["W5"] * obj_open_ot
            + W["W6"] * obj_transfer
            + W["W7"] * obj_delay
            + W["W8"] * obj_unscheduled
        )

        # --- Proxy term ---
        proxy_expr = self._build_proxy_term()

        self.model.setObjective(base_obj + proxy_expr, GRB.MINIMIZE)

    def _build_proxy_term(self) -> gp.LinExpr:
        """Costruisce il termine proxy da aggiungere alla funzione obiettivo."""
        strategy = self.proxy_strategy
        pw = self.proxy_weight
        inst = self.instance
        v = self.variables

        if strategy == ProxyStrategy.NONE:
            return 0

        # Helper: total occupied rooms per day  r_d = phi[r,d] + mu[r,d]
        def _rooms_day(d: int) -> gp.LinExpr:
            return gp.quicksum(v["phi"][r, d] + v["mu"][r, d] for r in inst.R)

        if strategy == ProxyStrategy.MAXIMIZE_ROOMS:
            # segno meno perché stiamo minimizzando
            return -pw * gp.quicksum(
                v["mu"][r, d] + v["phi"][r, d]
                for r in inst.R for d in inst.D
            )

        if strategy == ProxyStrategy.MINIMIZE_ROOMS:
            return pw * gp.quicksum(
                v["mu"][r, d] + v["phi"][r, d]
                for r in inst.R for d in inst.D
            )

        if strategy == ProxyStrategy.STABLE_ROOMS:
            # linearizzazione del valore assoluto -- FATTO
            days_pairs = list(zip(inst.D[:-1], inst.D[1:]))
            v["_proxy_var_plus"] = self.model.addVars(
                len(days_pairs), vtype=GRB.CONTINUOUS, lb=0, name="proxy_var_plus",
            )
            v["_proxy_var_minus"] = self.model.addVars(
                len(days_pairs), vtype=GRB.CONTINUOUS, lb=0, name="proxy_var_minus",
            )
            for idx, (d1, d2) in enumerate(days_pairs):
                self.model.addConstr(
                    _rooms_day(d2) - _rooms_day(d1)
                    == v["_proxy_var_plus"][idx] - v["_proxy_var_minus"][idx],
                    name=f"proxy_stable_{d1}_{d2}",
                )
            
            return pw * gp.quicksum(
                v["_proxy_var_plus"][i] + v["_proxy_var_minus"][i]
                for i in range(len(days_pairs))
            )

        if strategy == ProxyStrategy.BALANCE_WORKLOAD:
            return self._build_balance_proxy(
                self._nurse_balance_proxy_weight(attr="wl_daily"),
                attr="wl_daily",
                tag="wl",
            )

        if strategy == ProxyStrategy.BALANCE_SKILL:
            return self._build_balance_proxy(
                self._nurse_balance_proxy_weight(attr="skill_daily"),
                attr="skill_daily",
                tag="sk",
            )

        if strategy == ProxyStrategy.HYBRID:
            return (
                self._build_balance_proxy(
                    self._nurse_balance_proxy_weight(attr="skill_daily"),
                    attr="skill_daily",
                    tag="sk",
                )
                + self._build_balance_proxy(
                    self._nurse_balance_proxy_weight(attr="wl_daily"),
                    attr="wl_daily",
                    tag="wl",
                )
            )

        raise ValueError(f"Unknown proxy strategy: {strategy}")

    def _nurse_balance_proxy_weight(self, *, attr: str) -> float:

        if attr == "skill_daily":
            return self.proxy_weight * self.instance.weights["W2"]
        if attr == "wl_daily":
            return self.proxy_weight * self.instance.weights["W4"]
        raise ValueError(f"Unsupported nurse-balance attribute: {attr}")

    def _build_balance_proxy(
        self, pw: float, *, attr: str, tag: str,
    ) -> gp.LinExpr:
        """Proxy di bilanciamento: penalizza deviazioni dalla media per stanza/giorno."""
        inst = self.instance
        v = self.variables
        nR = len(inst.R)
        patient_val: dict[int, float] = getattr(inst, attr)

        v[f"_proxy_{tag}_dev_p"] = self.model.addVars(
            inst.R, inst.D, vtype=GRB.CONTINUOUS, lb=0,
            name=f"proxy_{tag}_dev_p",
        )
        v[f"_proxy_{tag}_dev_m"] = self.model.addVars(
            inst.R, inst.D, vtype=GRB.CONTINUOUS, lb=0,
            name=f"proxy_{tag}_dev_m",
        )

        for d in inst.D:
            total_d = gp.quicksum(
                patient_val[p] * v["y"][p, r, d]
                for p, r in self._y_pairs_by_day.get(d, ())
            )
            mean_d = total_d / nR

            for r in inst.R:
                room_val = gp.quicksum(
                    patient_val[p] * v["y"][p, r, d]
                    for p in self._patients_by_room_day.get((r, d), ())
                )
                self.model.addConstr(
                    room_val - mean_d
                    == v[f"_proxy_{tag}_dev_p"][r, d]
                    - v[f"_proxy_{tag}_dev_m"][r, d],
                    name=f"proxy_{tag}_bal_{r}_{d}",
                )

        return pw * gp.quicksum(
            v[f"_proxy_{tag}_dev_p"][r, d] + v[f"_proxy_{tag}_dev_m"][r, d]
            for r in inst.R for d in inst.D
        )

    def _add_constraints(self) -> None:
        self._cstr_admission_control()
        self._cstr_admission_occupancy_consistency()
        self._cstr_room_patient_compatibility()
        self._cstr_surgery_scheduling()

    def _cstr_admission_control(self) -> None:
        inst = self.instance
        v = self.variables
        m = self.model

        # obbligatori: esattamente una ammissione
        m.addConstrs(
            (
                gp.quicksum(v["alpha"][p, r, d] for r, d in self._alpha_keys_by_patient[p]) == 1
                for p in inst.PM
            ),
            name="mandatory_admission",
        )

        # facoltativi: al più una ammissione
        m.addConstrs(
            (
                gp.quicksum(v["alpha"][p, r, d] for r, d in self._alpha_keys_by_patient[p]) <= 1
                for p in inst.PO
            ),
            name="optional_admission",
        )

        m.addConstrs(
            (
                v["delta"][p]
                >= gp.quicksum(
                    d * v["alpha"][p, r, d] for r, d in self._alpha_keys_by_patient[p]
                )
                - inst.surgery_release_day[p]
                for p in inst.incoming_patients
            ),
            name="admission_delay",
        )

    def _cstr_admission_occupancy_consistency(self) -> None:
        inst = self.instance
        v = self.variables
        m = self.model

        m.addConstrs(
            (
                v["y"][p, r, d]
                == gp.quicksum(
                    v["alpha"][p, r, d2]
                    for d2 in range(
                        max(self._feasible_days[p][0], d - inst.length_of_stay[p] + 1),
                        min(self._feasible_days[p][-1], d) + 1,
                    )
                )
                for p, r, d in self._incoming_y_keys
            ),
            name="adm_occ",
        )

    def _cstr_room_patient_compatibility(self) -> None:
        inst = self.instance
        v = self.variables
        m = self.model
        M = inst.max_age_group

        m.addConstrs(
            (v["phi"][r, d] + v["mu"][r, d] <= 1 for r in inst.R for d in inst.D),
            name="gender_exclusion",
        )

        m.addConstrs(
            (
                gp.quicksum(v["y"][p, r, d] for p in self._female_by_room_day.get((r, d), ()))
                <= inst.room_capacity[r] * v["phi"][r, d]
                for r in inst.R
                for d in inst.D
            ),
            name="capacity_female",
        )
        m.addConstrs(
            (
                gp.quicksum(v["y"][p, r, d] for p in self._male_by_room_day.get((r, d), ()))
                <= inst.room_capacity[r] * v["mu"][r, d]
                for r in inst.R
                for d in inst.D
            ),
            name="capacity_male",
        )

        m.addConstrs(
            (
                v["pi_max"][r, d] >= inst.age_group[p] * v["y"][p, r, d]
                for p, r, d in self._y_keys
            ),
            name="age_max_lb",
        )
        m.addConstrs(
            (
                v["pi_min"][r, d] <= inst.age_group[p] + M * (1 - v["y"][p, r, d])
                for p, r, d in self._y_keys
            ),
            name="age_min_ub",
        )
        m.addConstrs(
            (
                v["pi_min"][r, d]
                <= M * gp.quicksum(v["y"][p, r, d] for p in self._patients_by_room_day.get((r, d), ()))
                for r in inst.R
                for d in inst.D
            ),
            name="age_min_empty",
        )

    def _cstr_surgery_scheduling(self) -> None:
        inst = self.instance
        v = self.variables
        m = self.model

        # l'intervento avviene lo stesso giorno dell'ammissione
        m.addConstrs(
            (
                gp.quicksum(v["gamma"][p, o, d] for o in inst.OT)
                == gp.quicksum(v["alpha"][p, r, d] for r in self._compatible_rooms[p])
                for p in inst.incoming_patients
                for d in self._feasible_days[p]
                if self._compatible_rooms[p]
            ),
            name="surgery_on_admission",
        )

        m.addConstrs(
            (
                gp.quicksum(
                    inst.surgery_duration[p] * v["gamma"][p, o, d]
                    for p in self._surgeon_patients_by_day.get((c, d), ())
                    for o in inst.OT
                )
                <= inst.max_surgery_time[c][d]
                for c in inst.C
                for d in inst.D
            ),
            name="max_surgeon_time",
        )

        m.addConstrs(
            (
                v["t"][c, d] >= gp.quicksum(v["w"][c, o, d] for o in inst.OT) - 1
                for c in inst.C
                for d in inst.D
            ),
            name="surgeon_transfer_ub",
        )
        m.addConstrs(
            (
                v["w"][self._patient_surgeon[p], o, d] >= v["gamma"][p, o, d]
                for p, o, d in self._gamma_keys
            ),
            name="surgeon_transfer_lb",
        )

        # disponibilità delle sale operatorie -- FATTO
        m.addConstrs(
            (
                gp.quicksum(
                    inst.surgery_duration[p] * v["gamma"][p, o, d]
                    for p in self._incoming_patients_by_day.get(d, ())
                )
                <= inst.max_ot_availability[o][d] * v["j"][o, d]
                for o in inst.OT
                for d in inst.D
            ),
            name="ot_availability",
        )

    # ================================================================== #
    def compute_phase1_objective(self) -> float:
        # il proxy non va nel costo reale, lo togliamo prima di riportare il valore
        raw_obj = self.model.ObjVal
        proxy_val = self._evaluate_proxy_value()
        return raw_obj - proxy_val

    def _evaluate_proxy_value(self) -> float:
        strategy = self.proxy_strategy
        pw = self.proxy_weight
        inst = self.instance
        v = self.variables

        if strategy == ProxyStrategy.NONE:
            return 0.0

        if strategy == ProxyStrategy.MAXIMIZE_ROOMS:
            return -pw * sum(
                v["mu"][r, d].X + v["phi"][r, d].X
                for r in inst.R for d in inst.D
            )

        if strategy == ProxyStrategy.MINIMIZE_ROOMS:
            return pw * sum(
                v["mu"][r, d].X + v["phi"][r, d].X
                for r in inst.R for d in inst.D
            )

        if strategy == ProxyStrategy.STABLE_ROOMS:
            n_pairs = len(inst.D) - 1
            return pw * sum(
                v["_proxy_var_plus"][i].X + v["_proxy_var_minus"][i].X
                for i in range(n_pairs)
            )

        if strategy == ProxyStrategy.BALANCE_WORKLOAD:
            return self._evaluate_balance_proxy(
                self._nurse_balance_proxy_weight(attr="wl_daily"),
                tag="wl",
            )

        if strategy == ProxyStrategy.BALANCE_SKILL:
            return self._evaluate_balance_proxy(
                self._nurse_balance_proxy_weight(attr="skill_daily"),
                tag="sk",
            )

        if strategy == ProxyStrategy.HYBRID:
            return (
                self._evaluate_balance_proxy(
                    self._nurse_balance_proxy_weight(attr="skill_daily"),
                    tag="sk",
                )
                + self._evaluate_balance_proxy(
                    self._nurse_balance_proxy_weight(attr="wl_daily"),
                    tag="wl",
                )
            )

        raise ValueError(f"Unknown proxy strategy: {strategy}")

    def _evaluate_balance_proxy(self, pw: float, *, tag: str) -> float:
        inst = self.instance
        v = self.variables
        return pw * sum(
            v[f"_proxy_{tag}_dev_p"][r, d].X + v[f"_proxy_{tag}_dev_m"][r, d].X
            for r in inst.R for d in inst.D
        )

    def get_fixed_vars(self) -> dict[str, dict]:
        """Return the variable groups that Phase 2 needs as fixed input."""
        keys = ("alpha", "y", "phi", "mu", "pi_max", "pi_min", "gamma", "t")
        return {k: self.variables[k] for k in keys}
