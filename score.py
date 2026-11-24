"""Batch inference and local score explanations using the saved assessment model."""
import argparse

import joblib
import pandas as pd

from features import IDS, INPUTS, RESULTS, build_features

GROUPS = {
    "vibration": ["vibration", "vibration_mean3", "vibration_missing"],
    "temperature": ["temperature_c", "temperature_mean3", "temperature_missing"],
    "faults": ["fault_code", "faults_last5"],
    "service history": ["cycles_since_install", "cycles_since_service", "history_days"],
    "utilization": ["cycles_today"],
    "equipment": ["airframe_family", "part_family"],
}


def predict(bundle, rows):
    return bundle["model"].predict_proba(rows[INPUTS])[:, 1]


def explain(bundle, rows):
    """Decrease in score when a signal group is replaced with its training reference."""
    scores = predict(bundle, rows)
    effects = {}
    for name, cols in GROUPS.items():
        changed = rows[INPUTS].copy()
        for col in cols:
            changed[col] = bundle["reference"][col]
        effects[name] = scores - predict(bundle, changed)
    effects = pd.DataFrame(effects)
    reasons = []
    for i, row in enumerate(rows.itertuples(index=False)):
        displayed = {
            "vibration": f"vibration={row.vibration:.2f} mm/s, 3-reading mean={row.vibration_mean3:.2f}",
            "temperature": f"temperature={row.temperature_c:.1f} C, 3-reading mean={row.temperature_mean3:.1f}",
            "faults": f"fault={row.fault_code}, {int(row.faults_last5)} flagged reports in last 5",
            "service history": f"install age={row.cycles_since_install} cycles, since service={row.cycles_since_service}",
            "utilization": f"{row.cycles_today} cycles today",
            "equipment": f"families {row.airframe_family}/{row.part_family}",
        }
        ranked = effects.iloc[i].nlargest(2)
        factors = [f"{displayed[name]} (score +{delta:.3f})"
                   for name, delta in ranked.items() if delta >= .0005]
        reason = " | ".join(factors) or "No group raised the score by 0.0005 against its reference"
        reasons.append(reason.replace("=nan", "=missing"))
    return reasons


def latest_scores(bundle, features):
    last = features[features.date == features.date.max()].copy()
    output = last[IDS + ["part_family", "history_days"]].copy()
    output["risk_score"] = predict(bundle, last)
    output["main_factors"] = explain(bundle, last)
    return output.sort_values(["risk_score", "serial"], ascending=[False, True])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations")
    parser.add_argument("output")
    args = parser.parse_args()
    bundle = joblib.load(RESULTS / "model.joblib")
    latest_scores(bundle, build_features(pd.read_csv(args.observations))).to_csv(
        args.output, index=False, float_format="%.6f")
