"""Entry point: ``python -m hippo``."""

from __future__ import annotations

import logging
import sys

from hippo.config import parse_args
from hippo.solver import append_results, run


def main() -> None:
    config = parse_args()

    # --- Logging setup ---
    log_level = logging.INFO if config.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Prevent gurobipy (≥11) from duplicating solver output
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    result = run(config)

    # Append to persistent log
    log_path = config.root_dir / "HIPPO_results_summary.txt"
    append_results(result, str(log_path))

    # Print summary to stdout
    print(result.summary())

    # Exit with non-zero if no feasible solution
    if result.overall_objective is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
