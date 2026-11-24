"""Generate observations from evolving component states, then derive labels from exits."""
import numpy as np
import pandas as pd

from features import DATA, derive_labels

SEED = 11703
N_TAILS, SLOTS, DAYS = 160, 8, 480


def simulate(seed=SEED, n_tails=N_TAILS, days=DAYS, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    n = n_tails * SLOTS
    tail = np.repeat(np.arange(n_tails), SLOTS)
    slot = np.tile(np.arange(1, SLOTS + 1), n_tails)
    tail_names = np.array([f"AC-{i:03d}" for i in range(n_tails)])
    airframes = np.array(["AF-A", "AF-B"])[np.arange(n_tails) % 2]
    types = (tail + slot) % 3
    part_names = np.array(["ROT-A", "ROT-B", "ROT-C"])[types]
    tail_bias = rng.normal(0, .1, n_tails)[tail]
    tail_cycle = rng.integers(5000, 22000, n_tails)
    age = rng.integers(10, 1000, n)
    since_service = np.minimum(age, rng.integers(0, 170, n))
    install = np.arange(1, n + 1)
    next_id = n + 1
    wear = np.zeros(n)
    growth = rng.uniform(.024, .047, n)
    transient = np.zeros(n)
    observations, exits = [], []
    calendar = pd.date_range(start, periods=days)

    for date in calendar:
        day = date.strftime("%Y-%m-%d")
        flown = rng.choice([1, 2, 3, 4], n_tails, p=[.10, .25, .45, .20])
        tail_cycle += flown
        cycles = flown[tail]
        age += cycles
        since_service += cycles
        since_service[rng.random(n) < .009] = 0

        onset_rate = .00058 * (1 + .22 * np.minimum(age / 1600, 1))
        onset = (wear == 0) & (rng.random(n) < 1 - (1 - onset_rate) ** cycles)
        wear[onset] = .025
        wear += (wear > 0) * cycles * growth
        sudden = rng.random(n) < 1 - (1 - .000045) ** cycles
        unscheduled = (wear >= 1) | sudden
        planned = (age > 1500) & (rng.random(n) < .004)
        removed = unscheduled | planned
        if removed.any():
            count = int(removed.sum())
            exits.append(pd.DataFrame({
                "date": day, "aircraft_id": tail_names[tail[removed]],
                "position": slot[removed], "serial": [f"SN-{v:06d}" for v in install[removed]],
                "install_id": install[removed], "removal_cycle": tail_cycle[tail[removed]],
                "reason": np.where(unscheduled[removed], "unscheduled", "planned"),
            }))
            install[removed] = np.arange(next_id, next_id + count)
            next_id += count
            age[removed] = since_service[removed] = 0
            wear[removed] = transient[removed] = 0
            growth[removed] = rng.uniform(.024, .047, count)

        # Temporary load excursions can resemble a degrading component.
        transient *= .74
        bursts = rng.random(n) < .012
        transient[bursts] += rng.uniform(.35, 1.05, int(bursts.sum()))
        load = rng.normal(0, 1, n_tails)[tail]
        vibration = (1.6 + .22 * types + tail_bias + .11 * load
                     + 2.25 * wear + 1.15 * transient + rng.normal(0, .24, n))
        temp = (59 + 2.2 * types + 1.2 * load + 9.2 * wear
                + 5.0 * transient + rng.normal(0, 1.65, n))
        fault = np.full(n, "NONE", dtype=object)
        faulted = rng.random(n) < (.012 + .44 * wear + .18 * transient)
        fault[faulted] = rng.choice(["VIB", "THERM"], int(faulted.sum()))
        fault[rng.random(n) < .018] = None
        vibration[rng.random(n) < .055] = np.nan
        temp[rng.random(n) < .065] = np.nan
        vibration[rng.random(n) < .0012] = -99
        temp[rng.random(n) < .0012] = 999
        observations.append(pd.DataFrame({
            "date": day, "aircraft_id": tail_names[tail], "airframe_family": airframes[tail],
            "position": slot, "serial": [f"SN-{v:06d}" for v in install],
            "install_id": install.copy(), "part_family": part_names, "aircraft_cycle": tail_cycle[tail],
            "cycles_today": cycles, "cycles_since_install": age.copy(),
            "cycles_since_service": since_service.copy(), "vibration": vibration,
            "temperature_c": temp, "fault_code": fault,
        }))
    return pd.concat(observations, ignore_index=True), pd.concat(exits, ignore_index=True)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    snapshots, events = simulate()
    snapshots["label_30fc"] = derive_labels(snapshots, events)
    snapshots.to_csv(DATA / "observations.csv.gz", index=False, float_format="%.3f",
                     compression={"method": "gzip", "compresslevel": 5, "mtime": 0})
    events.to_csv(DATA / "removals.csv", index=False)
    print(f"{len(snapshots):,} observations, {snapshots.label_30fc.mean():.2%} positive among known labels")
    print(events.reason.value_counts().to_string())
