"""HIPPO solver — two-phase matheuristic orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hippo.config import SolverConfig, load_bks
from hippo.heuristics.phase1 import Phase1HeuristicConfig, run_phase1_heuristic
from hippo.heuristics.phase2 import Phase2HeuristicConfig, run_phase2_heuristic
from hippo.instance import Instance, load_instance
from hippo.models.base import SolveResult
from hippo.models.phase1 import Phase1Builder
from hippo.models.phase2 import Phase2Builder
from hippo.solution.exporter import export_full_solution, export_phase1_solution

logger = logging.getLogger(__name__)


def _remaining_mip_time(total_phase_limit: float, heuristic_runtime: float) -> float:
    return max(0.0, total_phase_limit - heuristic_runtime)


@dataclass
class HippoResult:
    """Summary of a complete HIPPO run."""

    instance_name: str
    build_time_phase1: float
    solve_result_phase1: SolveResult
    phase1_objective: float | None
    build_time_phase2: float
    solve_result_phase2: SolveResult
    phase2_objective: float | None
    heuristic_obj_phase1: float | None = None
    heuristic_obj_phase2: float | None = None
    bks: int | None = None

    @property
    def overall_objective(self) -> float | None:
        if self.phase1_objective is not None and self.phase2_objective is not None:
            return self.phase1_objective + self.phase2_objective
        return None

    def summary(self) -> str:
        obj = self.overall_objective or 0
        lines = [
            "",
            "=" * 50,
            f"  HIPPO Results — {self.instance_name}",
            "=" * 50,
            f"  Phase 1  build : {self.build_time_phase1:>8.3f} s",
            f"  Phase 1  solve : {self.solve_result_phase1.runtime:>8.3f} s",
            f"  Phase 1  gap   : {self.solve_result_phase1.mip_gap or 0:>8.4f}",
            f"  Phase 1  obj   : {self.phase1_objective or 0:>10.1f}",
            "-" * 50,
            f"  Phase 2  build : {self.build_time_phase2:>8.3f} s",
            f"  Phase 2  solve : {self.solve_result_phase2.runtime:>8.3f} s",
            f"  Phase 2  gap   : {self.solve_result_phase2.mip_gap or 0:>8.4f}",
            f"  Phase 2  obj   : {self.phase2_objective or 0:>10.1f}",
            "-" * 50,
            f"  Overall  obj   : {obj:>10.1f}",
        ]
        if self.bks is not None and self.overall_objective is not None:
            abs_gap = obj - self.bks
            denom = max(obj, self.bks)
            pct_gap = (abs_gap / denom * 100) if denom > 0 else 0.0
            lines.append(f"  BKS            : {self.bks:>10d}")
            lines.append(f"  Gap (abs)      : {abs_gap:>+10.1f}")
            lines.append(f"  Gap (%)        : {pct_gap:>+9.2f} %")
        lines += ["=" * 50, ""]
        return "\n".join(lines)


def run(config: SolverConfig) -> HippoResult:
    """Execute the full HIPPO two-phase pipeline."""

    logger.info("Loading instance %s from %s", config.instance_name, config.instance_path)
    instance: Instance = load_instance(config.instance_path)

    dataset_for_bks = "" if config.instance_file is not None else config.dataset
    bks_map = load_bks(config.data_dir, dataset_for_bks)
    bks_value = bks_map.get(config.instance_name)

    # --- Fase 1: ammissione pazienti, stanze, chirurgia ---
    logger.info("=== Phase 1: Building light model ===\n  proxy_strategy=%s  proxy_weight=%.2f",
                config.proxy_strategy.value, config.proxy_weight)
    t0 = time.perf_counter()
    builder1 = Phase1Builder(
        instance,
        proxy_strategy=config.proxy_strategy,
        proxy_weight=config.proxy_weight,
        relaxation=config.relaxation,
    )
    builder1.build()
    build_time1 = time.perf_counter() - t0
    logger.info("Phase-1 model built in %.3f s", build_time1)

    phase1_mip_time_limit = config.time_limit_phase1
    heuristic_obj1: float | None = None
    if (not config.relaxation) and config.use_phase1_heuristic and config.heuristic_time_limit_phase1 > 0.0:
        heuristic_t0 = time.perf_counter()
        heuristic1 = run_phase1_heuristic(
            instance,
            Phase1HeuristicConfig(
                time_limit=min(config.heuristic_time_limit_phase1, config.time_limit_phase1),
                seed=config.heuristic_seed,
                proxy_strategy=config.proxy_strategy,
                proxy_weight=config.proxy_weight,
            ),
        )
        heuristic_time1 = time.perf_counter() - heuristic_t0
        phase1_mip_time_limit = _remaining_mip_time(config.time_limit_phase1, heuristic_time1)
        logger.info(
            "Phase-1 heuristic consumed %.3f s; residual MIP time limit %.3f s",
            heuristic_time1,
            phase1_mip_time_limit,
        )
        if heuristic1 is not None:
            heuristic_obj1 = heuristic1.objective_value()
            heuristic1.apply_mip_start(builder1)
            logger.info("Applied Phase-1 heuristic MIP start (objective %.3f)", heuristic_obj1)
        else:
            logger.warning("Phase-1 heuristic did not produce a feasible warm start")

    result1 = builder1.solve(
        time_limit=phase1_mip_time_limit,
        mip_gap=config.mip_gap_phase1,
        threads=config.gurobi_threads,
        verbose=config.verbose,
        extra_params=config.gurobi_extra_params or None,
    )

    phase1_obj: float | None = None
    if result1.is_feasible:
        phase1_obj = builder1.compute_phase1_objective()
        sol_light = config.solution_path("light")
        export_phase1_solution(instance, builder1.variables, sol_light)
        builder1.write_active_vars(config.result_path("light"))
    else:
        logger.error("Phase 1 did not find a feasible solution — aborting.")
        return HippoResult(
            instance_name=config.instance_name,
            build_time_phase1=build_time1,
            solve_result_phase1=result1,
            phase1_objective=None,
            build_time_phase2=0.0,
            solve_result_phase2=SolveResult(status=-1, runtime=0.0),
            phase2_objective=None,
            heuristic_obj_phase1=heuristic_obj1,
            heuristic_obj_phase2=None,
            bks=bks_value,
        )

    # --- Fase 2: assegnazione infermieri (variabili di fase 1 fissate) ---
    logger.info("=== Phase 2: Building full model ===")
    fixed_vars = builder1.get_fixed_vars()

    t0 = time.perf_counter()
    builder2 = Phase2Builder(instance, fixed_vars, relaxation=config.relaxation)
    builder2.build()
    build_time2 = time.perf_counter() - t0
    logger.info("Phase-2 model built in %.3f s", build_time2)

    phase2_mip_time_limit = config.time_limit_phase2
    heuristic_obj2: float | None = None
    if (not config.relaxation) and config.use_phase2_heuristic and config.heuristic_time_limit_phase2 > 0.0:
        heuristic_t0 = time.perf_counter()
        heuristic2 = run_phase2_heuristic(
            builder2,
            Phase2HeuristicConfig(
                time_limit=min(config.heuristic_time_limit_phase2, config.time_limit_phase2),
                seed=config.heuristic_seed,
            ),
        )
        heuristic_time2 = time.perf_counter() - heuristic_t0
        phase2_mip_time_limit = _remaining_mip_time(config.time_limit_phase2, heuristic_time2)
        logger.info(
            "Phase-2 heuristic consumed %.3f s; residual MIP time limit %.3f s",
            heuristic_time2,
            phase2_mip_time_limit,
        )
        if heuristic2 is not None:
            heuristic_obj2 = heuristic2.objective_value()
            heuristic2.apply_mip_start()
            logger.info("Applied Phase-2 heuristic MIP start (objective %.3f)", heuristic_obj2)
        else:
            logger.warning("Phase-2 heuristic did not produce a complete warm start")

    result2 = builder2.solve(
        time_limit=phase2_mip_time_limit,
        mip_gap=config.mip_gap_phase2,
        threads=config.gurobi_threads,
        verbose=config.verbose,
        extra_params=config.gurobi_extra_params or None,
    )

    phase2_obj: float | None = None
    if result2.is_feasible:
        phase2_obj = result2.obj_value
        sol_full = config.solution_path("full")
        export_full_solution(instance, builder1.variables, builder2.variables, sol_full)
        builder2.write_active_vars(config.result_path("full"))
    else:
        logger.warning("Phase 2 did not find a feasible solution.")

    return HippoResult(
        instance_name=config.instance_name,
        build_time_phase1=build_time1,
        solve_result_phase1=result1,
        phase1_objective=phase1_obj,
        build_time_phase2=build_time2,
        solve_result_phase2=result2,
        phase2_objective=phase2_obj,
        heuristic_obj_phase1=heuristic_obj1,
        heuristic_obj_phase2=heuristic_obj2,
        bks=bks_value,
    )


def append_results(result: HippoResult, path: str) -> None:
    """Append a run summary to a text log file."""
    from pathlib import Path as P
    P(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(result.summary())
