"""Ego speed from the OSDaR23 ``novatel_oem7_inspva`` stream.

The CSVs hold one row per frame with geodetic position, NED velocities,
attitude and azimuth.  ``north_velocity``/``east_velocity`` are in the
local level frame, so the speed towards the direction the train actually
faces has to be obtained by rotating them by the azimuth:

    forward = north * cos(azimuth) + east * sin(azimuth)

with ``azimuth`` measured clockwise from north.  Using the raw magnitude
``hypot(north, east)`` would overstate the closing speed whenever the
vehicle is not pointing along its own velocity vector (sideslip, curves),
which is exactly what the TTC of a target straight ahead depends on.
"""

import csv
import math
from pathlib import Path

_REQUIRED = ("frame_idx", "north_velocity", "east_velocity", "azimuth")


def forward_speed_ms(north_velocity: float, east_velocity: float, azimuth_deg: float) -> float:
    """Speed along the vehicle heading, in m/s."""
    a = math.radians(float(azimuth_deg))
    return float(north_velocity) * math.cos(a) + float(east_velocity) * math.sin(a)


def read_ins(path: Path) -> dict:
    """Read one INSPVA row into a dict of numbers (frame_idx stays an int)."""
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh), None)
    if row is None:
        return {}
    rec = {}
    for key, value in row.items():
        if key is None:
            continue
        try:
            rec[key] = float(value)
        except (TypeError, ValueError):
            rec[key] = value
    if "frame_idx" in rec:
        rec["frame_idx"] = int(round(float(rec["frame_idx"])))
    return rec


def read_ins_speeds(directory: Path) -> dict[int, float]:
    """Map frame number -> forward speed in km/h, skipping unparsable rows."""
    table: dict[int, float] = {}
    for path in sorted(Path(directory).glob("*.csv")):
        rec = read_ins(path)
        if any(key not in rec for key in _REQUIRED):
            continue
        speed = forward_speed_ms(rec["north_velocity"], rec["east_velocity"], rec["azimuth"]) * 3.6
        table[int(rec["frame_idx"])] = speed
    return table


def speed_for_frame(table: dict[int, float], frame: int) -> float | None:
    """Speed for ``frame``; nearest available frame when there is no exact hit."""
    if not table:
        return None
    if frame in table:
        return table[frame]
    nearest = min(table, key=lambda k: abs(k - frame))
    return table[nearest]
