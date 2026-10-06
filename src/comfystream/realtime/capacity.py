"""How many realtime pipelines a node may keep loaded at once.

Two streams fit a 24 GB card. A third or fourth often OOMs. Some models can
only run alone. The operator states those limits here; the supervisor loads
only a set that satisfies them and leaves the rest unloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class WarmCapacity:
    """Admission rules for pipelines that occupy GPU memory.

    ``max_loaded`` is the runner-wide cap. ``gpus`` overrides that cap per
    device id (the same string as a pipeline's ``gpu``). ``exclusive``
    pipelines load alone. ``combinations``, when set, require the loaded set
    to fit inside one listed group. Pipelines are considered in config order.
    """

    max_loaded: int = 2
    gpus: dict[str, int] = field(default_factory=dict)
    exclusive: frozenset[str] = frozenset()
    combinations: tuple[frozenset[str], ...] = ()

    def __post_init__(self) -> None:
        from comfystream.realtime.spec import RealtimeSpecError

        if self.max_loaded < 1:
            raise RealtimeSpecError("capacity.max_loaded must be >= 1")
        for gpu, limit in self.gpus.items():
            if limit < 1:
                raise RealtimeSpecError(f"capacity.gpus[{gpu!r}] must be >= 1")

    def refuse(self, loaded: list[RealtimePipelineSpec], spec: RealtimePipelineSpec) -> str | None:
        """Why ``spec`` cannot load beside ``loaded``, or None if it can."""
        names = {item.name for item in loaded}
        if spec.name in names:
            return None
        blockers = names & self.exclusive
        if blockers:
            return f"{sorted(blockers)[0]} is exclusive and is already loaded"
        if spec.name in self.exclusive and names:
            return f"{spec.name} is exclusive and cannot load beside {sorted(names)}"
        proposed = names | {spec.name}
        if self.combinations and not any(proposed <= group for group in self.combinations):
            return f"{spec.name} is outside every allowed combination with {sorted(names)}"
        if len(names) + 1 > self.max_loaded:
            return f"runner already has {len(names)} loaded pipelines (max_loaded {self.max_loaded})"
        if spec.gpu:
            on_gpu = sum(item.gpu == spec.gpu for item in loaded)
            limit = self.gpus.get(spec.gpu, self.max_loaded)
            if on_gpu + 1 > limit:
                return f"gpu {spec.gpu} already has {on_gpu} loaded pipelines (max {limit})"
        return None

    def select(
        self, specs: list["RealtimePipelineSpec"]
    ) -> tuple[set[str], list[tuple[str, str]]]:
        """Pipelines to load, and the ones held back with the reason."""
        chosen: list[RealtimePipelineSpec] = []
        held: list[tuple[str, str]] = []
        for spec in specs:
            reason = self.refuse(chosen, spec)
            if reason:
                held.append((spec.name, reason))
                continue
            chosen.append(spec)
        return {spec.name for spec in chosen}, held


def parse_capacity(raw: object, names: set[str]) -> WarmCapacity:
    """Build a :class:`WarmCapacity` from the ``capacity`` mapping in realtime.yaml."""
    from comfystream.realtime.spec import RealtimeSpecError

    if raw is None:
        return WarmCapacity()
    if not isinstance(raw, dict):
        raise RealtimeSpecError("capacity must be a mapping")
    unknown = set(raw) - {"max_loaded", "gpus", "exclusive", "combinations"}
    if unknown:
        raise RealtimeSpecError(f"capacity: unknown keys {sorted(unknown)}")
    gpus_raw = raw.get("gpus") or {}
    if not isinstance(gpus_raw, dict):
        raise RealtimeSpecError("capacity.gpus must map a gpu id to a max loaded count")
    exclusive = _name_list(raw.get("exclusive") or [], "capacity.exclusive", names)
    combinations_raw = raw.get("combinations") or []
    if not isinstance(combinations_raw, list):
        raise RealtimeSpecError("capacity.combinations must be a list of pipeline groups")
    combinations = tuple(
        frozenset(_name_list(group, "capacity.combinations", names)) for group in combinations_raw
    )
    if any(not group for group in combinations):
        raise RealtimeSpecError("capacity.combinations entries must name at least one pipeline")
    try:
        return WarmCapacity(
            max_loaded=int(raw.get("max_loaded", 2)),
            gpus={str(gpu): int(limit) for gpu, limit in gpus_raw.items()},
            exclusive=frozenset(exclusive),
            combinations=combinations,
        )
    except RealtimeSpecError:
        raise
    except (TypeError, ValueError) as exc:
        raise RealtimeSpecError(f"capacity: {exc}") from exc


def _name_list(raw: object, label: str, names: set[str]) -> list[str]:
    from comfystream.realtime.spec import RealtimeSpecError

    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise RealtimeSpecError(f"{label} must be a list of pipeline names")
    unknown = sorted(set(raw) - names)
    if unknown:
        raise RealtimeSpecError(f"{label}: unknown pipelines {unknown}")
    return list(raw)
