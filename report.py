import pandas as pd

from features import ROOT


def write_report(raw, ledger, train, val, test, comparison, samples, winner, interval, sweep):
    results = comparison[comparison.split == "test"]
    best = results[results.model == winner["name"]].iloc[0]
    target = best.event_recall >= .80 and best.peak_daily_per_100 <= 5
    decision = "met both targets" if target else "fell short of the joint target"
    core = test[test.core]
    vcore = val[val.core]
    latest = raw[raw.date == raw.date.max()]
    report = [
        "# Forecasting unscheduled component removals",
        f"**{winner['name']} {decision} on the synthetic holdout.** It flagged "
        f"{int(best.hits)} of {int(best.events)} actual unscheduled removals "
        f"({best.event_recall:.1%}). The maintenance queue received {best.alerts_per_100:.2f} new "
        f"tickets per 100 active components per day, with a daily peak of {best.peak_daily_per_100:.2f}. "
        f"The median first warning preceded a detected removal by {best.median_lead_cycles:.0f} flight cycles. "
        f"An aircraft-level bootstrap gives a 95% interval of {interval[0]:.1%} to {interval[1]:.1%} "
        "for event recall on this simulated population.",
        "The [independent fleet study](FLEET_STUDY.md) adds a competing-risk model and compares two frozen "
        "ticket policies on fresh fleets, including how often alerts arrive at least five cycles ahead.",
        "## Data and target",
        f"The assessment supplied no historical records. The fixed-seed simulation produces "
        f"{len(raw):,} end-of-day observations across {raw.aircraft_id.nunique()} aircraft and "
        f"{raw.serial.nunique():,} serials. The ledger contains {ledger.reason.eq('unscheduled').sum():,} "
        f"unscheduled exits and {ledger.reason.eq('planned').sum():,} planned exits. "
        "A hidden degradation process, temporary load excursions and independent sudden failures drive the mock fleet. "
        "Replacement retires the old serial before the new end-of-day snapshot is scored. "
        f"Among {raw.label_30fc.notna().sum():,} resolved records, {raw.label_30fc.mean():.2%} are positive. "
        f"Another {raw.label_30fc.isna().sum():,} have incomplete follow-up.",
        "A positive snapshot precedes an unscheduled exit of the same installation by 1–30 flown cycles. "
        "A planned exit within that horizon ends the installation with a negative outcome. "
        "Available predictors include component family, age, time since service, today's utilization, current and "
        "trailing sensor readings, and recent fault reports. Windows use the last three or five daily observations "
        "and reset at installation. Readings beyond the mock instruments' range become missing. Plausible spikes "
        "are retained. Missing flags and a capped observation count identify sparse histories.",
        "## Validation design and ticket policy",
        f"The training set uses {train.aircraft_id.nunique()} aircraft and {len(train):,} mature rows from "
        f"{train.date.min()} to {train.date.max()}. Each training aircraft has 30 further cycles of observed "
        "history before the validation boundary. The scored validation core spans "
        f"{vcore.date.min()} to {vcore.date.max()} on {vcore.aircraft_id.nunique()} different aircraft. "
        f"The final test core spans {core.date.min()} to {core.date.max()} on {core.aircraft_id.nunique()} further aircraft. "
        "The code checks chronological order and zero shared aircraft, serials and installations. Each evaluation "
        "period has a 30-cycle warm-up for ticket state and 30 cycles of follow-up for the last core scores. "
        "Prior observations from the same held-out serial are available only as trailing inference history.",
        "A daily fleet snapshot is the workload denominator. A ticket requires a score at or above threshold and an "
        "available slot among the highest eligible scores. Capacity is the active roster size divided by 20, rounded down. "
        "A ticket expires after 30 flown cycles or at installation exit. An event counts once if any issued ticket on the "
        "same installation leads it by 1–30 cycles. Recall includes all unscheduled core events, including sudden removals "
        "and newly installed components. Ticket precision follows each core ticket to its outcome.",
        "Both candidates weight positive training records fourfold. Validation and test retain their observed prevalence. "
        "Logistic regression uses training medians, standardization and one-hot categories with unknown categories ignored. "
        "CatBoost uses native numerical missing-value handling and categorical inputs. Both receive identical features. "
        "For each model, nine fixed thresholds were replayed on validation. Selection maximizes event recall under the "
        "hard daily cap, then favors fewer tickets, higher snapshot average precision and the higher threshold. "
        "Model and thresholds were locked before test evaluation. Scores range from 0 to 1 and serve as risk ranks. "
        "Probability use would require separate calibration. "
        "[Threshold validation guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).",
        "## Operating results",
    ]
    table = [
        "| Period | Model | Threshold | Events detected | Recall | Tickets / 100 | Peak / 100 | Ticket precision | Snapshot AP | Median lead (cycles) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples():
        cut = "daily quota" if r.model.startswith("Random") else f"{r.threshold:.2f}"
        table.append(f"| {r.split} | {r.model} | {cut} | {r.hits}/{r.events} | {r.event_recall:.1%} "
                     f"| {r.alerts_per_100:.2f} | {r.peak_daily_per_100:.2f} | {r.alert_precision:.1%} "
                     f"| {r.snapshot_AP:.3f} | {r.median_lead_cycles:g} |")
    report.append("\n".join(table))
    report.append("The seeded random comparator opens the same number of tickets as the selected model on each day "
                  "and observes the same expiry rule. Snapshot AP scores individual observations. Event recall measures "
                  "how many distinct removals planners could see coming. [Validation threshold sweep](results/validation_thresholds.csv) "
                  "and [event-level audit](results/test_events.csv).")
    confirmed = round(best.alert_precision * best.alerts)
    report.append(f"**Review workload.** Of {int(best.alerts):,} core test tickets, {confirmed} were followed by an "
                  f"unscheduled exit within 30 cycles. A ticket and an event can fall on opposite sides of the core boundary, "
                  f"so this ticket count and the {int(best.hits)} core events detected have different denominators.")
    lower_work = sweep[(sweep.model == winner["name"]) & sweep.event_recall.ge(.80)].sort_values("alerts")
    if not lower_work.empty and lower_work.iloc[0].threshold != winner["threshold"]:
        alternative = lower_work.iloc[0]
        report.append(f"The validation sweep also identifies a lower-workload option. A {alternative.threshold:.2f} "
                      f"threshold detected {int(alternative.hits)}/{int(alternative.events)} validation events with "
                      f"{int(alternative.alerts):,} tickets. The selected policy prioritizes the most detected events. "
                      "The alternative gives planners a concrete trade-off to evaluate on a new cohort.")
    report.append("## Example decisions")
    for r in samples.itertuples():
        readable = r.main_factors.replace(" | ", ". ")
        prior = (f" The earlier ticket was issued {int(r.prior_ticket_lead_cycles)} cycles before removal."
                 if pd.notna(r.prior_ticket_lead_cycles) else "")
        report.append(f"**{r.case}** · {r.date} · {r.aircraft_id} / {r.serial} · score {r.risk_score:.4f}. "
                      f"{r.decision}. {r.outcome}.{prior} Contributing signals included {readable}.")
    report.append("The examples show the highest-scoring correct and false tickets and, when present, the highest score "
                  "in a missed event's forecast window. Each local effect is the decrease in score when related readings "
                  "are jointly replaced by their training medians or modes. Correlated groups can share an effect. "
                  "The readings give an engineer a starting point for inspection.")
    report.extend([
        "## Deployment, monitoring and limitations",
        f"The batch scoring example outputs {len(latest):,} active serials on {latest.date.iloc[0]}, including fresh installs. "
        "Each new serial starts with family, known service age, current readings and the number of reports available. "
        "Missing sensors and unseen categories follow the training-time handling. Daily rows carry the serial, aircraft, "
        "cycle, score and local factors into the MRO planning queue. In a live integration, persist the model version, "
        "decision time and open-ticket state and upsert by installation and issue cycle. Planners review source readings "
        "and record their disposition before any maintenance action.",
        "Monitor roster coverage, input freshness, missingness, unseen families, score distributions and ticket volume daily. "
        "After 30 cycles of follow-up, track event recall, ticket precision and warning lead by family, new-install status "
        "and missing-sensor status. Investigate persistent shifts in routes, sensor calibration or maintenance practice. "
        "Retrain on mature recent outcomes, repeat a later grouped holdout, and run a candidate in shadow before rollout.",
        "The simulator sets the rates of progressive and abrupt failures, fleet composition and sensor behavior. It "
        "assumes a complete daily roster, immediate data availability and one installation per serial. Sudden removals can "
        "have little warning. On a real fleet, record event timing, loading delays, transfers, refits, maintenance interventions "
        "and ticket closure explicitly. Test escalation of an existing ticket when risk rises sharply and measure the "
        "extra review workload. The mock results support a workflow demonstration. Real fleet performance, the "
        "workload accounting period, and the minimum useful planning lead require prospective validation with MRO staff.",
    ])
    (ROOT / "REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
