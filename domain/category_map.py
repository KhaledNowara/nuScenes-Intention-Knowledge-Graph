"""Map nuScenes category names to coarse ontology labels."""

from __future__ import annotations

# Explicit prefixes and exact names (nuScenes uses e.g. vehicle.car, human.pedestrian.adult)
PREFIX_TO_CATEGORY: list[tuple[str, str]] = [
    ("vehicle.bicycle", "Vehicle"),
    ("vehicle.motorcycle", "Vehicle"),
    ("vehicle.car", "Vehicle"),
    ("vehicle.bus", "Vehicle"),
    ("vehicle.construction", "Vehicle"),
    ("vehicle.trailer", "Vehicle"),
    ("vehicle.truck", "Vehicle"),
    ("human.pedestrian", "Pedestrian"),
    ("human.pedestrian.adult", "Pedestrian"),
    ("human.pedestrian.child", "Pedestrian"),
    ("human.pedestrian.construction_worker", "Pedestrian"),
    ("human.pedestrian.police_officer", "Pedestrian"),
    ("movable_object", "MovableObject"),
    ("static_object", "Infrastructure"),
    ("traffic_cone", "MovableObject"),
    ("barrier", "Infrastructure"),
    ("debris", "MovableObject"),
]

DEFAULT_CATEGORY = "MovableObject"


def ontology_type_for_category(nuscenes_category_name: str) -> str:
    name = (nuscenes_category_name or "").strip()
    if not name:
        return DEFAULT_CATEGORY
    for prefix, category in PREFIX_TO_CATEGORY:
        if name == prefix or name.startswith(prefix + "."):
            return category
    return DEFAULT_CATEGORY
