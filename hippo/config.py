"""Configuration and CLI argument parsing for HIPPO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ProxyStrategy(str, Enum):
    """Proxy terms for the Phase-1 objective (λ in the paper).

    Each strategy steers room-occupancy to ease Phase-2 nurse scheduling.
    """

    MAXIMIZE_ROOMS = "maximize_rooms"
    MINIMIZE_ROOMS = "minimize_rooms"
    STABLE_ROOMS = "stable_rooms"
    BALANCE_WORKLOAD = "balance_workload"
    BALANCE_SKILL = "balance_skill"
    HYBRID = "hybrid"
    NONE = "none"


DATASET_CHOICES = ("public", "hidden", "test")

_FILE_PREFIX = {
    "public": "i",
    "hidden": "m",
    "test": "test",
}


def instance_filename(dataset: str, instance_id: int) -> str:
    """Return the JSON filename, e.g. public/7 → i07.json."""
    prefix = _FILE_PREFIX[dataset]
    return f"{prefix}{instance_id:02d}.json"


def load_bks(data_dir: Path, dataset: str) -> dict[str, int]:
    """Load best-known values from bestKnownValues.txt.

    Returns empty dict if the file doesn't exist (hidden instances or custom paths).
    """
    if not dataset:
        return {}
    bks_path = data_dir / dataset / "bestKnownValues.txt"
    if not bks_path.exists():
        return {}
    result: dict[str, int] = {}
    for line in bks_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            result[parts[0]] = int(parts[1])
    return result


@dataclass
class SolverConfig:
    """All parameters for a HIPPO run."""

    # instance — use dataset+instance_id or instance_file (mutually exclusive)
    dataset: str = "public"
    instance_id: int = 1
    instance_file: Path | None = None  # direct path to any JSON instance

    # time limits (seconds per phase)
    time_limit_phase1: float = 1800.0
    time_limit_phase2: float = 1800.0

    # MIP gaps
    mip_gap_phase1: float = 1e-6
    mip_gap_phase2: float = 1e-6

    # proxy
    proxy_strategy: ProxyStrategy = ProxyStrategy.NONE
    proxy_weight: float = 0.0

    # warm-start heuristics (disabled by default)
    use_phase1_heuristic: bool = False
    use_phase2_heuristic: bool = False
    heuristic_time_limit_phase1: float = 60.0
    heuristic_time_limit_phase2: float = 60.0
    heuristic_seed: int = 1234

    # misc
    relaxation: bool = False
    verbose: bool = True
    gurobi_threads: int | None = None

    # paths — resolved in __post_init__
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path | None = None
    results_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data"
        if self.results_dir is None:
            self.results_dir = self.root_dir / "results"
        if self.instance_file is None and self.dataset not in DATASET_CHOICES:
            raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got '{self.dataset}'")
        if self.gurobi_threads is not None and self.gurobi_threads < 1:
            raise ValueError("gurobi_threads must be >= 1 when provided")

    @property
    def instance_name(self) -> str:
        if self.instance_file is not None:
            return Path(self.instance_file).stem
        prefix = _FILE_PREFIX[self.dataset]
        return f"{prefix}{self.instance_id:02d}"

    @property
    def instance_path(self) -> Path:
        if self.instance_file is not None:
            return Path(self.instance_file)
        return self.data_dir / self.dataset / instance_filename(self.dataset, self.instance_id)

    @property
    def _output_dir(self) -> Path:
        if self.instance_file is not None:
            p = Path(self.instance_file)
            d = self.results_dir / p.parent.name / p.stem
        else:
            d = self.results_dir / self.dataset / f"{self.instance_id:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def result_path(self, phase: str) -> Path:
        return self._output_dir / f"vars_{phase}.txt"

    def solution_path(self, phase: str) -> Path:
        return self._output_dir / f"sol_{phase}.json"


def parse_args(argv: list[str] | None = None) -> SolverConfig:
    """Parse CLI arguments and return a SolverConfig."""

    parser = argparse.ArgumentParser(
        prog="hippo",
        description="HIPPO — Two-phase matheuristic for the IHTC 2024 problem.",
    )

    # Instance selection: either dataset+id (named datasets) or --file (arbitrary path)
    instance_group = parser.add_mutually_exclusive_group(required=True)
    instance_group.add_argument(
        "--file", type=Path, dest="instance_file", metavar="PATH",
        help="Path to any JSON instance file (overrides dataset/instance_id)",
    )
    instance_group.add_argument(
        "dataset", nargs="?", choices=DATASET_CHOICES,
        help="Dataset name: public, hidden, or test",
    )

    parser.add_argument("instance_id", type=int, nargs="?",
                        help="Instance number within the dataset (e.g. 7 for public/i07)")

    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Time limit per phase in seconds (default: 1800)")
    parser.add_argument("--time-limit-phase1", type=float, default=None)
    parser.add_argument("--time-limit-phase2", type=float, default=None)

    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--mip-gap-phase1", type=float, default=None)
    parser.add_argument("--mip-gap-phase2", type=float, default=None)

    parser.add_argument("--proxy-strategy", type=str, default="none",
                        choices=[s.value for s in ProxyStrategy],
                        help="Phase-1 proxy strategy (default: none)")
    parser.add_argument("--proxy-weight", type=float, default=0.0,
                        help="λ — weight of the proxy term (default: 0)")

    parser.add_argument("--heuristic-time-limit", type=float, default=60.0,
                        help="Heuristic budget per phase in seconds (default: 60)")
    parser.add_argument("--heuristic-time-limit-phase1", type=float, default=None)
    parser.add_argument("--heuristic-time-limit-phase2", type=float, default=None)
    parser.add_argument("--heuristic-seed", type=int, default=1234)
    parser.add_argument("--enable-phase1-heuristic", dest="use_phase1_heuristic",
                        action="store_true", default=False)
    parser.add_argument("--disable-phase1-heuristic", dest="use_phase1_heuristic",
                        action="store_false")
    parser.add_argument("--enable-phase2-heuristic", dest="use_phase2_heuristic",
                        action="store_true", default=False)
    parser.add_argument("--disable-phase2-heuristic", dest="use_phase2_heuristic",
                        action="store_false")

    parser.add_argument("--relaxation", action="store_true",
                        help="Solve LP relaxation instead of MIP")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--threads", type=int, default=None)

    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)

    args = parser.parse_args(argv)

    # When using named dataset, instance_id is also required
    if args.instance_file is None:
        if args.dataset is None:
            parser.error("positional argument 'dataset' is required unless --file is used")
        if args.instance_id is None:
            parser.error("positional argument 'instance_id' is required unless --file is used")

    return SolverConfig(
        dataset=args.dataset or "public",
        instance_id=args.instance_id or 1,
        instance_file=args.instance_file,
        time_limit_phase1=args.time_limit_phase1 if args.time_limit_phase1 is not None else args.time_limit,
        time_limit_phase2=args.time_limit_phase2 if args.time_limit_phase2 is not None else args.time_limit,
        mip_gap_phase1=args.mip_gap_phase1 if args.mip_gap_phase1 is not None else args.mip_gap,
        mip_gap_phase2=args.mip_gap_phase2 if args.mip_gap_phase2 is not None else args.mip_gap,
        proxy_strategy=ProxyStrategy(args.proxy_strategy),
        proxy_weight=args.proxy_weight,
        use_phase1_heuristic=args.use_phase1_heuristic,
        use_phase2_heuristic=args.use_phase2_heuristic,
        heuristic_time_limit_phase1=(
            args.heuristic_time_limit_phase1
            if args.heuristic_time_limit_phase1 is not None else args.heuristic_time_limit
        ),
        heuristic_time_limit_phase2=(
            args.heuristic_time_limit_phase2
            if args.heuristic_time_limit_phase2 is not None else args.heuristic_time_limit
        ),
        heuristic_seed=args.heuristic_seed,
        relaxation=args.relaxation,
        verbose=not args.quiet,
        gurobi_threads=args.threads,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
