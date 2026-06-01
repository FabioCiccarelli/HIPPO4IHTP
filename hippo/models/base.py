"""Classe base con helper Gurobi condivisi tra le due fasi."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from hippo.instance import Instance

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Lightweight container for solver outcome."""

    status: int
    runtime: float
    obj_value: float | None = None
    mip_gap: float | None = None

    @property
    def is_feasible(self) -> bool:
        return self.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.obj_value is not None


class BaseModelBuilder:
    """Abstract base for HIPPO MIP phases."""

    def __init__(self, instance: Instance, *, relaxation: bool = False, name: str = "IHTP") -> None:
        self.instance = instance
        self.relaxation = relaxation
        self.model: gp.Model = gp.Model(name)
        self.variables: dict[str, Any] = {}

    def _vtype(self, default: str = GRB.BINARY) -> str:
        # in modalità rilassamento tutto diventa continuo
        return GRB.CONTINUOUS if self.relaxation else default

    def build(self) -> None:
        raise NotImplementedError

    def solve(
        self,
        *,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        threads: int | None = None,
        verbose: bool = True,
        extra_params: dict[str, Any] | None = None,
    ) -> SolveResult:

        if time_limit is not None:
            self.model.setParam("TimeLimit", time_limit)
        if mip_gap is not None:
            self.model.setParam("MIPGap", mip_gap)
        if threads is not None:
            self.model.setParam("Threads", threads)
        if not verbose:
            self.model.setParam("OutputFlag", 0)
        if extra_params:
            for key, val in extra_params.items():
                self.model.setParam(key, val)

        self.model.optimize()

        status = self.model.status
        runtime = self.model.Runtime
        obj_value = self.model.ObjVal if self.model.SolCount > 0 else None
        gap = self.model.MIPGap if self.model.SolCount > 0 else None

        result = SolveResult(status=status, runtime=runtime, obj_value=obj_value, mip_gap=gap)

        if result.is_feasible:
            logger.info(
                "Objective: %.2f | Time: %.2fs | Gap: %.4f",
                obj_value, runtime, gap,
            )
        elif status == GRB.INFEASIBLE:
            logger.warning("Model is infeasible — computing IIS.")
            self.model.computeIIS()
            self.model.write("infeasible.ilp")  # utile per il debug, da guardare dopo
        else:
            logger.warning("No feasible solution found (status=%d).", status)

        return result

    def write_active_vars(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for v in self.model.getVars():
                if abs(v.X) > 1e-6:
                    fh.write(f"{v.VarName} {v.X}\n")
        logger.info("Active variables written to %s", path)

    def extract_nonzero(self) -> dict[str, dict[tuple, float]]:
        if self.model.SolCount == 0:
            logger.warning("No solution available to extract.")
            return {}
        return {
            name: {k: var[k].X for k in var if abs(var[k].X) > 1e-6}
            for name, var in self.variables.items()
        }
