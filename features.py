from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
HORIZON = 30
CATEGORICAL = ["airframe_family", "part_family", "fault_code"]
NUMERIC = [
    "cycles_since_install", "cycles_since_service", "cycles_today",
    "vibration", "vibration_mean3", "temperature_c", "temperature_mean3",
    "faults_last5", "history_days", "vibration_missing", "temperature_missing",
]
INPUTS = NUMERIC + CATEGORICAL
IDS = ["date", "aircraft_id", "position", "serial", "install_id", "aircraft_cycle"]


def derive_labels(observations, events):
    """A planned exit ends the current installation. An unresolved horizon stays unknown."""
    exits = events.set_index("install_id", verify_integrity=True)
    delta = observations.install_id.map(exits.removal_cycle) - observations.aircraft_cycle
    reason = observations.install_id.map(exits.reason)
    if delta.dropna().le(0).any():
        raise ValueError("An observation is at or after its own installation exit")
    within = delta.between(1, HORIZON)
    last_cycle = observations.groupby("aircraft_id").aircraft_cycle.transform("max")
    complete = last_cycle - observations.aircraft_cycle >= HORIZON
    labels = (within & reason.eq("unscheduled")).astype("Int8")
    return labels.where(complete | within)


def build_features(raw):
    """Each rolling window ends at the current observation and resets at installation."""
    required = IDS + ["airframe_family", "part_family", "fault_code", "cycles_today",
                      "cycles_since_install", "cycles_since_service", "vibration", "temperature_c"]
    frame = raw[required].copy()
    if frame.duplicated(["date", "serial"]).any():
        raise ValueError("A serial occurs more than once on a scoring date")
    frame = frame.sort_values(["install_id", "aircraft_cycle"], kind="stable")
    for column, low, high in [("vibration", 0, 20), ("temperature_c", -40, 160)]:
        frame[column] = frame[column].where(frame[column].between(low, high))
    frame["vibration_missing"] = frame.vibration.isna().astype(int)
    frame["temperature_missing"] = frame.temperature_c.isna().astype(int)
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)
    frame["_fault"] = (~frame.fault_code.isin(["NONE", "UNKNOWN"])).astype(int)
    groups = frame.groupby("install_id", sort=False)
    for source, output, width, operation in [
        ("vibration", "vibration_mean3", 3, "mean"),
        ("temperature_c", "temperature_mean3", 3, "mean"),
        ("_fault", "faults_last5", 5, "sum"),
    ]:
        window = groups[source].rolling(width, min_periods=1)
        frame[output] = getattr(window, operation)().droplevel(0)
    frame["history_days"] = (groups.cumcount() + 1).clip(upper=5)
    return frame[IDS + INPUTS].reindex(raw.index)
