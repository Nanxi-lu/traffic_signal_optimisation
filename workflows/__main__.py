"""Command-line entry point: ``python -m workflows <phase> [<phase> ...]``."""

import argparse
import sys
import time

from workflows import phase_registry, run_phase


def main(argv: list[str] | None = None) -> int:
    registry = phase_registry()
    parser = argparse.ArgumentParser(
        prog="python -m workflows",
        description=(
            "Run phases of the synthetic traffic signal study. Each phase reads "
            "earlier outputs and overwrites its own files in outputs/."
        ),
    )
    parser.add_argument("phases", nargs="*", help="phase names, or 'all' for every phase in order")
    parser.add_argument("--list", action="store_true", help="list the available phases")
    args = parser.parse_args(argv)

    if args.list or not args.phases:
        width = max(map(len, registry))
        for phase, (module, description) in registry.items():
            print(f"{phase:<{width}}  {description}  [{module}]")
        if not args.phases:
            print("\nRun Phase 1 data generation first with: python -m traffic_signals.demo")
        return 0

    phases = list(registry) if args.phases == ["all"] else args.phases
    unknown = [p for p in phases if p not in registry]
    if unknown:
        parser.error(f"unknown phase(s): {', '.join(unknown)}")
    for phase in phases:
        start = time.perf_counter()
        print(f"\n=== {phase}: {registry[phase][1]} ===")
        run_phase(phase)
        print(f"=== {phase} finished in {time.perf_counter() - start:.1f} s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
