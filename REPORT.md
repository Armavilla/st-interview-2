# Forecasting unscheduled component removals

**Logistic regression met both targets on the synthetic holdout.** It flagged 70 of 83 actual unscheduled removals (84.3%). The maintenance queue received 4.18 new tickets per 100 active components per day, with a daily peak of 5.00. The median first warning preceded a detected removal by 20 flight cycles. An aircraft-level bootstrap gives a 95% interval of 76.5% to 91.4% for event recall on this simulated population.

## Data and target

The assessment supplied no historical records. The fixed-seed simulation produces 614,400 end-of-day observations across 160 aircraft and 2,634 serials. The ledger contains 1,135 unscheduled exits and 221 planned exits. A hidden degradation process, temporary load excursions and independent sudden failures drive the mock fleet. Replacement retires the old serial before the new end-of-day snapshot is scored. Among 600,090 resolved records, 1.99% are positive. Another 14,310 have incomplete follow-up.

A positive snapshot precedes an unscheduled exit of the same installation by 1–30 flown cycles. A planned exit within that horizon ends the installation with a negative outcome. Available predictors include component family, age, time since service, today's utilization, current and trailing sensor readings, and recent fault reports. Windows use the last three or five daily observations and reset at installation. Readings beyond the mock instruments' range become missing. Plausible spikes are retained. Missing flags and a capped observation count identify sparse histories.

## Validation design and ticket policy

The training set uses 80 aircraft and 109,176 mature rows from 2024-01-01 to 2024-06-21. Each training aircraft has 30 further cycles of observed history before the validation boundary. The scored validation core spans 2024-07-15 to 2024-11-01 on 40 different aircraft. The final test core spans 2024-11-29 to 2025-04-10 on 40 further aircraft. The code checks chronological order and zero shared aircraft, serials and installations. Each evaluation period has a 30-cycle warm-up for ticket state and 30 cycles of follow-up for the last core scores. Prior observations from the same held-out serial are available only as trailing inference history.

A daily fleet snapshot is the workload denominator. A ticket requires a score at or above threshold and an available slot among the highest eligible scores. Capacity is the active roster size divided by 20, rounded down. A ticket expires after 30 flown cycles or at installation exit. An event counts once if any issued ticket on the same installation leads it by 1–30 cycles. Recall includes all unscheduled core events, including sudden removals and newly installed components. Ticket precision follows each core ticket to its outcome.

Both candidates weight positive training records fourfold. Validation and test retain their observed prevalence. Logistic regression uses training medians, standardization and one-hot categories with unknown categories ignored. CatBoost uses native numerical missing-value handling and categorical inputs. Both receive identical features. For each model, nine fixed thresholds were replayed on validation. Selection maximizes event recall under the hard daily cap, then favors fewer tickets, higher snapshot average precision and the higher threshold. Model and thresholds were locked before test evaluation. Scores range from 0 to 1 and serve as risk ranks. Probability use would require separate calibration. [Threshold validation guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Operating results

| Period | Model | Threshold | Events detected | Recall | Tickets / 100 | Peak / 100 | Ticket precision | Snapshot AP | Median lead (cycles) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Logistic regression | 0.06 | 53/54 | 98.1% | 3.88 | 5.00 | 4.0% | 0.622 | 19 |
| validation | CatBoost | 0.03 | 53/54 | 98.1% | 4.61 | 5.00 | 3.4% | 0.607 | 19 |
| test | Logistic regression | 0.06 | 70/83 | 84.3% | 4.18 | 5.00 | 4.0% | 0.595 | 20 |
| test | CatBoost | 0.03 | 73/83 | 88.0% | 4.69 | 5.00 | 3.7% | 0.588 | 20 |
| test | Random, matched workload | daily quota | 38/83 | 45.8% | 4.18 | 5.00 | 2.1% | 0.021 | 14.5 |

The seeded random comparator opens the same number of tickets as the selected model on each day and observes the same expiry rule. Snapshot AP scores individual observations. Event recall measures how many distinct removals planners could see coming. [Validation threshold sweep](results/validation_thresholds.csv) and [event-level audit](results/test_events.csv).

**Review workload.** Of 1,778 core test tickets, 71 were followed by an unscheduled exit within 30 cycles. A ticket and an event can fall on opposite sides of the core boundary, so this ticket count and the 70 core events detected have different denominators.

The validation sweep also identifies a lower-workload option. A 0.95 threshold detected 51/54 validation events with 54 tickets. The selected policy prioritizes the most detected events. The alternative gives planners a concrete trade-off to evaluate on a new cohort.

## Example decisions

**Detected removal** · 2024-12-12 · AC-144 / SN-001157 · score 0.9988. Ticket opened. Unscheduled removal in 7 cycles. Contributing signals included vibration=4.25 mm/s, 3-reading mean=4.05 (score +0.895). temperature=74.0 C, 3-reading mean=71.5 (score +0.011).

**False alert** · 2024-12-06 · AC-157 / SN-001263 · score 0.8597. Ticket opened. No unscheduled removal in 30 cycles. Contributing signals included vibration=3.02 mm/s, 3-reading mean=3.07 (score +0.807). temperature=68.0 C, 3-reading mean=69.8 (score +0.306).

**Missed removal** · 2024-12-31 · AC-139 / SN-001630 · score 0.9998. Earlier ticket still open. Unscheduled removal in 3 cycles. The earlier ticket was issued 31 cycles before removal. Contributing signals included vibration=4.00 mm/s, 3-reading mean=3.82 (score +0.331). fault=THERM, 4 flagged reports in last 5 (score +0.002).

The examples show the highest-scoring correct and false tickets and, when present, the highest score in a missed event's forecast window. Each local effect is the decrease in score when related readings are jointly replaced by their training medians or modes. Correlated groups can share an effect. The readings give an engineer a starting point for inspection.

## Deployment, monitoring and limitations

The batch scoring example outputs 1,280 active serials on 2025-04-24, including fresh installs. Each new serial starts with family, known service age, current readings and the number of reports available. Missing sensors and unseen categories follow the training-time handling. Daily rows carry the serial, aircraft, cycle, score and local factors into the MRO planning queue. In a live integration, persist the model version, decision time and open-ticket state and upsert by installation and issue cycle. Planners review source readings and record their disposition before any maintenance action.

Monitor roster coverage, input freshness, missingness, unseen families, score distributions and ticket volume daily. After 30 cycles of follow-up, track event recall, ticket precision and warning lead by family, new-install status and missing-sensor status. Investigate persistent shifts in routes, sensor calibration or maintenance practice. Retrain on mature recent outcomes, repeat a later grouped holdout, and run a candidate in shadow before rollout.

The simulator sets the rates of progressive and abrupt failures, fleet composition and sensor behavior. It assumes a complete daily roster, immediate data availability and one installation per serial. Sudden removals can have little warning. On a real fleet, record event timing, loading delays, transfers, refits, maintenance interventions and ticket closure explicitly. Test escalation of an existing ticket when risk rises sharply and measure the extra review workload. The mock results support a workflow demonstration. Real fleet performance, the workload accounting period, and the minimum useful planning lead require prospective validation with MRO staff.
