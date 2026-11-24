import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import HORIZON


def split_cohorts(frame, train_end="2024-07-01", validation_end="2024-11-15"):
    tail_number = frame.aircraft_id.str.removeprefix("AC-").astype(int)
    train = frame[(tail_number < 80) & (frame.date < train_end)].copy()
    validation = frame[tail_number.between(80, 119) & frame.date.ge(train_end)
                       & frame.date.lt(validation_end)].copy()
    test = frame[(tail_number >= 120) & (frame.date >= validation_end)].copy()
    last = train.groupby("aircraft_id").aircraft_cycle.transform("max")
    train = train[last - train.aircraft_cycle >= HORIZON]
    parts = [train, validation, test]
    for i, left in enumerate(parts):
        for right in parts[i + 1:]:
            if left.date.max() >= right.date.min():
                raise ValueError("Cohort dates overlap")
            for key in ["aircraft_id", "serial", "install_id"]:
                if set(left[key].unique()) & set(right[key].unique()):
                    raise ValueError(f"Cohorts share {key}")
    return train, validation, test


def evaluation_period(block, events):
    cycles = block.groupby(["date", "aircraft_id"]).aircraft_cycle.max().unstack()
    if cycles.isna().any().any():
        raise ValueError("This mock benchmark requires a complete daily active-tail roster")
    warm = (cycles - cycles.iloc[0]).min(axis=1).ge(HORIZON)
    mature = (cycles.iloc[-1] - cycles).min(axis=1).ge(HORIZON)
    dates = cycles.index[warm & mature]
    if dates.empty:
        raise ValueError("The evaluation block has insufficient flight-cycle history")
    # Warm-up alerts carry forward. The end guard supplies outcomes for the last core scores.
    rows = block[block.date <= dates[-1]].sort_values(["date", "install_id"]).reset_index(drop=True)
    rows["core"] = rows.date >= dates[0]
    core_events = events[events.date.between(dates[0], dates[-1]) &
                         events.aircraft_id.isin(block.aircraft_id.unique()) &
                         events.reason.eq("unscheduled")].copy()
    if rows.loc[rows.core, "label_30fc"].isna().any():
        raise ValueError("An evaluation score has an unresolved outcome")
    return rows, core_events


def issue_alerts(rows, scores, threshold, daily_quota=None):
    """One new ticket per installation per 30 cycles, under each day's integer fleet cap."""
    scores = np.asarray(scores)
    ids = rows.install_id.to_numpy()
    cycles = rows.aircraft_cycle.to_numpy()
    fired = np.zeros(len(rows), dtype=bool)
    expiry = {}
    for day, positions in rows.groupby("date", sort=True).indices.items():
        cap = len(positions) // 20
        if daily_quota is not None:
            cap = min(cap, int(daily_quota[day]))
        if cap == 0:
            continue
        eligible = positions[scores[positions] >= threshold]
        ranked = eligible[np.lexsort((ids[eligible], -scores[eligible]))]
        issued = 0
        for i in ranked:
            if cycles[i] < expiry.get(ids[i], -1):
                continue
            fired[i] = True
            expiry[ids[i]] = cycles[i] + HORIZON
            issued += 1
            if issued == cap:
                break
    return fired


def event_leads(rows, fired, events):
    tickets = rows.loc[fired, ["install_id", "aircraft_cycle"]]
    pairs = tickets.merge(events[["install_id", "removal_cycle"]], on="install_id")
    pairs["lead"] = pairs.removal_cycle - pairs.aircraft_cycle
    matches = pairs[pairs.lead.between(1, HORIZON)]
    return matches.groupby("install_id").lead.max()


def evaluate(rows, events, scores, threshold, quota=None):
    fired = issue_alerts(rows, scores, threshold, quota)
    core = rows.core.to_numpy()
    leads = event_leads(rows, fired, events)
    daily = rows.loc[core, ["date"]].assign(alert=fired[core]).groupby("date").alert
    daily_count = daily.sum()
    if not daily_count.le(daily.size() // 20).all():
        raise AssertionError("Daily ticket limit exceeded")
    labels = rows.loc[core, "label_30fc"].astype(int)
    metrics = {
        "threshold": threshold, "events": len(events), "hits": len(leads),
        "event_recall": len(leads) / len(events) if len(events) else np.nan,
        "alerts": int(daily_count.sum()), "active_component_days": int(core.sum()),
        "alerts_per_100": 100 * fired[core].mean(),
        "peak_daily_per_100": 100 * daily.mean().max(),
        "alert_precision": float(labels[fired[core]].mean()) if fired[core].any() else 0.,
        "snapshot_AP": average_precision_score(labels, np.asarray(scores)[core]),
        "median_lead_cycles": float(leads.median()) if len(leads) else np.nan,
    }
    return metrics, fired


def recall_interval(events, leads, tails):
    """Percentile interval from resampling held-out aircraft, conditional on the fitted model."""
    by_tail = events.assign(hit=events.install_id.isin(leads.index)).groupby("aircraft_id").hit
    counts = pd.concat([by_tail.sum(), by_tail.size()], axis=1).reindex(tails, fill_value=0).to_numpy()
    rng = np.random.default_rng(107)
    draws = counts[rng.integers(0, len(counts), (2000, len(counts)))].sum(axis=1)
    draws = draws[draws[:, 1] > 0]
    return np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])
