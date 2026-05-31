# HIPPO

**A two-phase matheuristic for the Integrated Healthcare Timetabling Problem (IHTC 2024)**

This repository contains the source code and computational results associated with the paper:

> Ciccarelli, F., Di Biase, A., and Furini, F.  
> *A Two-Phase Matheuristic for the Integrated Healthcare Timetabling Problem* (2026)


The algorithm targets the Integrated Healthcare Timetabling Problem (IHTP) defined in the [IHTC 2024 challenge](https://ihtc2024.github.io/), a combinatorial optimization competition focused on integrated scheduling of patient admissions, room assignments, surgical planning, and nurse rostering in a hospital setting.

HIPPO decomposes the IHTP problem into two sequential MIP phases solved with [Gurobi](https://www.gurobi.com()):

| Phase | Name   | Decisions                                            |
|-------|--------|------------------------------------------------------|
| 1     | Light  | Patient admission, room assignment, surgery schedule |
| 2     | Full   | Nurse-to-room assignment, skill/workload violations  | 

Phase 1 fixes the admission/room/surgery variables; Phase 2 uses those fixed values
to optimize nurse scheduling and soft-constraint violations.

---

## Repository structure

```
hippo/                         # Main Python package
├── __init__.py
├── __main__.py                # Entry point: python -m hippo
├── config.py                  # CLI arguments, ProxyStrategy enum, SolverConfig
├── heuristics/                # Warm-start heuristics for Phase 1 and Phase 2
├── instance.py                # Instance parser (JSON → dataclass)
├── solver.py                  # Two-phase orchestrator
├── models/
│   ├── base.py                # Abstract base with common Gurobi helpers
│   ├── phase1.py              # Phase 1: admission + rooms + surgery + proxy
│   └── phase2.py              # Phase 2: nurse assignment + violations
└── solution/
    └── exporter.py            # IHTC-format JSON exporter
data/
├── public/                    # i01.json … i30.json
├── hidden/                    # m01.json … m30.json
├── test/                      # test01.json … test10.json
└── longerHorizon/             # lHH_N.json (extended-horizon instances)
computationalResults/          # Instance-wise results reported in the paper
pyproject.toml
.gitignore
README.md
```

### Output layout

For named datasets outputs go to `results/{dataset}/{instance_id}/`; for arbitrary files they go to `results/{parent_dir}/{stem}/`:

```
results/
├── public/
│   └── 01/
│       ├── sol_light.json         # Phase-1 solution (patients only)
│       ├── sol_full.json          # Full solution (patients + nurses)
│       ├── vars_light.txt         # Active variables from Phase 1
│       └── vars_full.txt          # Active variables from Phase 2
└── longerHorizon/
    └── l35_1/
        ├── sol_light.json
        ├── sol_full.json
        ├── vars_light.txt
        └── vars_full.txt
```

---

## Installation

```bash
pip install -e .
# or with dev tools
pip install -e ".[dev]"
```

**Requires:** Python ≥ 3.9 and a valid Gurobi license.

---

## Quick start

Two ways to specify the instance:

**Named dataset** (public / hidden / test):
```bash
# Public instance i01, default 30 minutes per phase
python -m hippo public 1

# Hidden instance m03, custom time limits
python -m hippo hidden 3 --time-limit-phase1 1800 --time-limit-phase2 600
```

**Arbitrary file** (`--file`), for any JSON instance (including `longerHorizon`):
```bash
python -m hippo --file data/longerHorizon/l35_1.json
python -m hippo --file /path/to/my_custom_instance.json --time-limit 3600
```

Other options work the same in both modes:
```bash
# No proxy term
python -m hippo public 1 --proxy-strategy none

# Proxy with weight λ=200
python -m hippo public 5 --proxy-strategy stable_rooms --proxy-weight 200

# Hybrid proxy
python -m hippo public 1 --proxy-strategy hybrid --proxy-weight 100

# With warm-start heuristics
python -m hippo public 1 --enable-phase1-heuristic --enable-phase2-heuristic --heuristic-time-limit 60

# LP relaxation
python -m hippo public 1 --time-limit 60 --relaxation
```

## Warm-start heuristics

HIPPO includes two internal warm-start heuristics:

- **Phase 1**: constructive admission/room/OT heuristic followed by ALNS.
- **Phase 2**: greedy nurse-to-room-shift assignment followed by ALNS.

Both are disabled by default and skipped automatically in `--relaxation` mode.

CLI controls:

- `--heuristic-time-limit`: budget in seconds for both phases (default `60`).
- `--heuristic-time-limit-phase1` / `--heuristic-time-limit-phase2`: phase-specific overrides.
- `--heuristic-seed`: random seed (default `1234`).
- `--enable-phase1-heuristic` / `--enable-phase2-heuristic`: enable each warm start.

The heuristic budget is subtracted from the MIP time limit of the same phase.

---

## Proxy strategies

The Phase-1 objective can include a proxy term weighted by *λ* (`--proxy-weight`, default 0):

| Strategy             | CLI flag            | Effect                                                   |
|----------------------|---------------------|----------------------------------------------------------|
| **Maximize rooms**   | `maximize_rooms`    | Reward total occupied room-day slots                     |
| **Minimize rooms**   | `minimize_rooms`    | Penalize total occupied room-day slots                   |
| **Stable rooms**     | `stable_rooms`      | Penalize day-to-day variation in occupied rooms          |
| **Balance workload** | `balance_workload`  | Penalize workload imbalance across rooms per day         |
| **Balance skill**    | `balance_skill`     | Penalize skill-requirement imbalance across rooms per day|
| **Hybrid**           | `hybrid`            | Penalize skill and workload imbalances                   |
| **None**             | `none`              | No proxy — pure Phase-1 objective (default)              |

All strategies are mutually exclusive.

`balance_workload` and `balance_skill` use averaged per-patient parameters to
penalize deviations from the per-day mean across rooms.

`hybrid` combines both balance proxies: *λ* × (W2 × balance-skill + W4 × balance-workload).

---

## Configuration via Python API

```python
from pathlib import Path
from hippo.config import ProxyStrategy, SolverConfig
from hippo.solver import run

# Named dataset
config = SolverConfig(
    dataset="public",
    instance_id=1,
    time_limit_phase1=1800,
    time_limit_phase2=600,
    mip_gap_phase1=0.01,
    proxy_strategy=ProxyStrategy.STABLE_ROOMS,
    proxy_weight=200.0,
    verbose=False,
)
result = run(config)
print(result.summary())

# Arbitrary instance file
config2 = SolverConfig(
    instance_file=Path("data/longerHorizon/l35_1.json"),
    time_limit_phase1=1800,
    time_limit_phase2=600,
)
result2 = run(config2)
print(result2.summary())
```

---

## Computational results

The `computationalResults/` directory contains the full instance-wise results discussed in the paper. For each dataset (public, hidden, test) and each instance, the files report objective values, bound, gap, and runtimes for both phases, as well as soft-constraint breakdowns. These tables correspond directly to the experiments presented in the paper.

---

## License

This software is released under an academic and research license. Any publication that uses or builds upon this code must cite the paper listed above. See [LICENSE](LICENSE) for the full terms.


---
For questions or issues, please contact: [f.ciccarelli@uniroma1.it](mailto:f.ciccarelli@uniroma1.it)
