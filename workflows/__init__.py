"""Research workflow for the synthetic traffic signal optimisation study.

The original phase runner scripts are merged into four stage modules:

- ``workflows.stage1_2_timing``: Phase 1 checks and Phase 2 fixed timing plans
- ``workflows.stage3_coordination``: Phase 3 coordination and offset optimisation
- ``workflows.stage4a_4e_corridor``: Phases 4A to 4E corridor vehicle simulation
- ``workflows.stage4f_4g_full_junction``: Phases 4F and 4G full-junction simulation

Run a phase from the repository root with ``python -m workflows <phase>``,
list phases with ``python -m workflows --list``, or run the whole chain in
order with ``python -m workflows all``. Phase 1 data generation remains
``python -m traffic_signals.demo``.
"""

from importlib import import_module

STAGE_MODULES = (
    "workflows.stage1_2_timing",
    "workflows.stage3_coordination",
    "workflows.stage4a_4e_corridor",
    "workflows.stage4f_4g_full_junction",
)


def phase_registry() -> dict[str, tuple[str, str]]:
    """Map each phase name to (stage module, description), in run order."""

    registry: dict[str, tuple[str, str]] = {}
    for module_name in STAGE_MODULES:
        module = import_module(module_name)
        for phase, description in module.PHASE_DESCRIPTIONS.items():
            registry[phase] = (module_name, description)
    return registry


def run_phase(phase: str):
    """Run one phase and return whatever its runner returns."""

    registry = phase_registry()
    if phase not in registry:
        raise KeyError(f"Unknown phase {phase!r}. Use one of: {', '.join(registry)}.")
    module = import_module(registry[phase][0])
    return module.PHASES[phase]()
