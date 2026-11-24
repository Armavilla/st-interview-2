"""Freeze two validation policies per candidate, then replay six independent mock fleets."""
import joblib
import numpy as np
import pandas as pd

from assess import THRESHOLDS, fit_candidates
from competing_risks import ENDS, CompetingRiskModel
from features import DATA, HORIZON, INPUTS, RESULTS, build_features, derive_labels
from policy import evaluate, evaluation_period, event_leads, split_cohorts
from score import predict
from simulate import simulate
from study_report import publish_study

SEEDS = (93401, 93402, 93403, 93404, 93405, 93406)
RECALL = "Maximize recall"
WORK = "Minimize tickets"


def select_policies(sweep):
    recall = sweep.sort_values(["event_recall", "alerts", "threshold"], ascending=[False, True, False])
    feasible = sweep[(sweep.event_recall >= .8) & (sweep.peak_daily_per_100 <= 5)]
    work = feasible.sort_values(["alerts", "event_recall", "threshold"], ascending=[True, False, False])
    selections = [recall.iloc[0].to_dict() | {"policy": RECALL}]
    if not work.empty:
        selections.append(work.iloc[0].to_dict() | {"policy": WORK})
    return selections


def measured_run(rows, events, scores, threshold):
    metrics, issued = evaluate(rows, events, scores, threshold)
    leads = event_leads(rows, issued, events)
    ticket_cycle = rows.aircraft_cycle.where(issued).groupby(rows.install_id).ffill()
    open_today = (rows.aircraft_cycle - ticket_cycle).between(0, HORIZON - 1)
    daily_open = open_today[rows.core].groupby(rows.loc[rows.core, "date"]).mean()
    return metrics | {
        "confirmed_tickets": int(rows.loc[rows.core & issued, "label_30fc"].sum()),
        "hits_at_least_5_cycles": int(leads.ge(5).sum()),
        "peak_open_per_100": 100 * daily_open.max(),
    }


def main():
    raw = pd.read_csv(DATA / "observations.csv.gz")
    ledger = pd.read_csv(DATA / "removals.csv")
    features = build_features(raw)
    features["label_30fc"] = derive_labels(raw, ledger)
    train, val_block, _ = split_cohorts(features, train_end="2024-11-15", validation_end="2025-04-15")
    val, events = evaluation_period(val_block, ledger)
    bundles = fit_candidates(train)
    hazard = CompetingRiskModel().fit(train, ledger)
    hazard_name = "Competing risks"
    bundles[hazard_name] = {"name": hazard_name, "model": hazard,
                            "reference": bundles["Logistic regression"]["reference"]}
    examples = pd.read_csv(RESULTS / "examples.csv")[["case", "date", "serial", "outcome"]]
    cases = examples.merge(features, on=["date", "serial"], validate="one_to_one")
    if len(cases) != len(examples):
        raise ValueError("Case missing from the original held-out observations")
    curves = pd.concat([cases[examples.columns], pd.DataFrame(hazard.risk_curve(cases[INPUTS]),
                       columns=[f"by_{end}_cycles" for end in ENDS])], axis=1)
    curves.to_csv(RESULTS / "horizon_examples.csv", index=False)
    validation, locked = [], []
    for model, bundle in bundles.items():
        scores = predict(bundle, val)
        sweep = pd.DataFrame([{"model": model, **measured_run(val, events, scores, cut)} for cut in THRESHOLDS])
        validation.append(sweep)
        locked.extend(select_policies(sweep))
    pd.concat(validation).to_csv(RESULTS / "study_validation.csv", index=False)
    locked = pd.DataFrame(locked)
    eligible = locked[locked.policy == WORK].sort_values(["alerts", "event_recall", "model"],
                                                          ascending=[True, False, True])
    if eligible.empty:
        raise ValueError("No validation policy meets the joint operating target")
    pilot = eligible.iloc[0]
    joblib.dump(bundles[pilot.model] | {"threshold": pilot.threshold}, RESULTS / "shadow_policy.joblib", compress=3)
    joblib.dump(bundles[hazard_name], RESULTS / "competing_risks.joblib", compress=3)
    # Replay the original two rules on the same fleets to isolate the threshold change.
    original = joblib.load(RESULTS / "model.joblib")
    historical_sweep = pd.read_csv(RESULTS / "validation_thresholds.csv")
    historical_rules = select_policies(historical_sweep[historical_sweep.model == original["name"]])
    if historical_rules[0]["threshold"] != original["threshold"]:
        raise ValueError("Original validation selection and deployed bundle disagree")
    bundles["Original logistic"] = original
    historical_rules = pd.DataFrame(historical_rules).assign(model="Original logistic")
    locked = pd.concat([locked, historical_rules], ignore_index=True)
    locked.to_csv(RESULTS / "locked_policies.csv", index=False)

    records = []
    for seed in SEEDS:
        raw, exits = simulate(seed=seed, n_tails=40, days=365, start="2025-05-01")
        # The seed namespaces physical identities. Development aircraft never recur here.
        for frame in (raw, exits):
            for key in ["aircraft_id", "serial"]:
                frame[key] = str(seed) + "-" + frame[key]
        raw[["temperature_c", "vibration"]] = raw[["temperature_c", "vibration"]].round(3)
        features = build_features(raw)
        features["label_30fc"] = derive_labels(raw, exits)
        block = features[features.date >= "2025-11-01"]
        rows, events = evaluation_period(block, exits)
        for model, bundle in bundles.items():
            scores = predict(bundle, rows)
            for rule in locked[locked.model == model].itertuples():
                result = measured_run(rows, events, scores, rule.threshold)
                records.append({"fleet_seed": seed, "start": rows.loc[rows.core, "date"].min(),
                                "end": rows.loc[rows.core, "date"].max(), "model": model,
                                "policy": rule.policy, **result})
        print(f"Finished fleet {seed} with {len(events)} removal events", flush=True)
    runs = pd.DataFrame(records)
    runs.to_csv(RESULTS / "fleet_policy_results.csv", index=False)
    publish_study(runs, pilot, curves)


if __name__ == "__main__":
    main()
