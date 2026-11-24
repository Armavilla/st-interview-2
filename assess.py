"""Fit two candidates, freeze the validation decision, then evaluate the future aircraft cohort."""
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, DATA, HORIZON, IDS, INPUTS, NUMERIC, RESULTS, build_features, derive_labels
from policy import evaluation_period, evaluate, event_leads, recall_interval, split_cohorts
from report import write_report
from score import explain, latest_scores, predict

THRESHOLDS = [0.01, 0.03, 0.06, 0.12, 0.25, 0.40, 0.60, 0.80, 0.95]


def fit_candidates(train):
    transforms = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("family", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    logistic = make_pipeline(transforms, LogisticRegression(class_weight={0: 1, 1: 4}, max_iter=400))
    boosted = CatBoostClassifier(iterations=160, depth=5, learning_rate=.085, random_seed=190,
                                 cat_features=CATEGORICAL, one_hot_max_size=10, class_weights=[1, 4],
                                 thread_count=4, verbose=False, allow_writing_files=False)
    reference = {c: float(train[c].median()) for c in NUMERIC}
    reference.update({c: train[c].mode().iloc[0] for c in CATEGORICAL})
    bundles = {}
    for name, model in [("Logistic regression", logistic), ("CatBoost", boosted)]:
        model.fit(train[INPUTS], train.label_30fc.astype(int))
        bundles[name] = {"name": name, "model": model, "reference": reference}
    return bundles


def pick_examples(rows, ledger, events, bundle):
    choices = []
    opened = rows[rows.core & rows.new_alert]
    for label, name in [(1, "Detected removal"), (0, "False alert")]:
        able = opened[opened.label_30fc.eq(label)]
        if not able.empty:
            choices.append((name, able.risk_score.idxmax()))
    leads = event_leads(rows, rows.new_alert, events)
    missed = events[~events.install_id.isin(leads.index)].set_index("install_id")
    distance = rows.install_id.map(missed.removal_cycle) - rows.aircraft_cycle
    eligible = rows[distance.between(1, HORIZON)]
    if not eligible.empty:
        choices.append(("Missed removal", eligible.risk_score.idxmax()))
    exits = ledger.set_index("install_id")
    samples = []
    for case, index in choices:
        row = rows.loc[index]
        earlier_ticket_lead = None
        next_exit = exits.loc[row.install_id] if row.install_id in exits.index else None
        lead = int(next_exit.removal_cycle - row.aircraft_cycle) if next_exit is not None else 0
        matches = next_exit is not None and next_exit.reason == "unscheduled" and 1 <= lead <= HORIZON
        if row.new_alert:
            decision = "Ticket opened"
        elif row.risk_score < bundle["threshold"]:
            decision = "Below threshold"
        else:
            previous = rows[rows.install_id.eq(row.install_id) & rows.new_alert &
                            rows.aircraft_cycle.lt(row.aircraft_cycle)]
            suppressed = not previous.empty and row.aircraft_cycle - previous.aircraft_cycle.max() < HORIZON
            decision = "Earlier ticket still open" if suppressed else "Daily capacity reached"
            if suppressed and matches:
                earlier_ticket_lead = int(next_exit.removal_cycle - previous.aircraft_cycle.max())
        samples.append({
            "case": case, "date": row.date, "aircraft_id": row.aircraft_id, "serial": row.serial,
            "risk_score": row.risk_score, "decision": decision,
            "prior_ticket_lead_cycles": earlier_ticket_lead,
            "outcome": f"Unscheduled removal in {lead} cycles" if matches else "No unscheduled removal in 30 cycles",
            "main_factors": explain(bundle, rows.loc[[index]])[0],
        })
    return pd.DataFrame(samples)


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "observations.csv.gz", dtype={"label_30fc": "Int8"})
    ledger = pd.read_csv(DATA / "removals.csv")
    verified = derive_labels(raw, ledger)
    if not verified.equals(raw.label_30fc):
        raise ValueError("Published labels differ from the dated removal ledger")
    features = build_features(raw)
    features["label_30fc"] = verified
    train, val_block, test_block = split_cohorts(features)
    val, val_events = evaluation_period(val_block, ledger)
    bundles = fit_candidates(train)
    sweep = []
    for name, bundle in bundles.items():
        scores = predict(bundle, val)
        for threshold in THRESHOLDS:
            metrics, _ = evaluate(val, val_events, scores, threshold)
            sweep.append({"model": name, **metrics})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(RESULTS / "validation_thresholds.csv", index=False)
    ranking = sweep.sort_values(["event_recall", "alerts", "snapshot_AP", "threshold"],
                                ascending=[False, True, False, False], kind="stable")
    chosen = ranking.drop_duplicates("model")
    for row in chosen.itertuples():
        bundles[row.model]["threshold"] = row.threshold
    winner = bundles[chosen.iloc[0].model]
    joblib.dump(winner, RESULTS / "model.joblib", compress=3)

    # Test outcomes are first used after both model thresholds and the winner are frozen.
    test, test_events = evaluation_period(test_block, ledger)
    comparisons = chosen.assign(split="validation").to_dict("records")
    for name, bundle in bundles.items():
        scores = predict(bundle, test)
        metrics, decisions = evaluate(test, test_events, scores, bundle["threshold"])
        comparisons.append({"split": "test", "model": name, **metrics})
        if name == winner["name"]:
            test["risk_score"] = scores
            test["new_alert"] = decisions
    quota = test.groupby("date").new_alert.sum()
    random_scores = np.random.default_rng(734).random(len(test))
    metrics, _ = evaluate(test, test_events, random_scores, 0., quota)
    comparisons.append({"split": "test", "model": "Random, matched workload", **metrics})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(RESULTS / "comparison.csv", index=False)
    test[IDS + ["core", "risk_score", "new_alert", "label_30fc"]].to_csv(
        RESULTS / "test_scores.csv.gz", index=False, float_format="%.8f",
        compression={"method": "gzip", "compresslevel": 5, "mtime": 0})
    leads = event_leads(test, test.new_alert, test_events)
    audit = test_events.copy()
    audit["first_alert_lead_cycles"] = audit.install_id.map(leads).astype("Int64")
    audit["detected"] = audit.first_alert_lead_cycles.notna()
    audit.to_csv(RESULTS / "test_events.csv", index=False)
    interval = recall_interval(test_events, leads, sorted(test.aircraft_id.unique()))
    samples = pick_examples(test, ledger, test_events, winner)
    samples.to_csv(RESULTS / "examples.csv", index=False, float_format="%.6f")
    latest_scores(winner, features).to_csv(RESULTS / "latest_fleet_scores.csv", index=False,
                                          float_format="%.6f")
    write_report(raw, ledger, train, val, test, comparison, samples, winner, interval, sweep)
    columns = ["split", "model", "threshold", "hits", "events", "event_recall", "alerts_per_100",
               "peak_daily_per_100", "alert_precision", "snapshot_AP", "median_lead_cycles"]
    print(comparison[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Selected on validation: {winner['name']}, threshold {winner['threshold']}")


if __name__ == "__main__":
    main()
