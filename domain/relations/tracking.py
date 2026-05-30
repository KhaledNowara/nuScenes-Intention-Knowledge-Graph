"""Iterate instance tracks across consecutive keyframes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Tuple

from domain.model import Dataset, Scene, SceneParticipant, Trip


def iter_consecutive_appearances(
    dataset: Dataset,
) -> Iterator[Tuple[Trip, Scene, SceneParticipant, Scene, SceneParticipant]]:
    """Yield (trip, scene_prev, sp_prev, scene_curr, sp_curr) for every instance
    across consecutive keyframes. Skips the first appearance of each instance.
    """
    for trip in dataset.trips:
        track: dict[str, list[tuple[int, Scene, SceneParticipant]]] = defaultdict(list)
        for scene in trip.sequence.scenes:
            for sp in scene.participants:
                track[sp.participant.instance_token].append((scene.index, scene, sp))

        for appearances in track.values():
            appearances.sort(key=lambda a: a[0])
            for i in range(1, len(appearances)):
                _, scene_prev, sp_prev = appearances[i - 1]
                _, scene_curr, sp_curr = appearances[i]
                yield trip, scene_prev, sp_prev, scene_curr, sp_curr
