"""Render the policy comparison from the fleet-level measurements."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import ROOT, RESULTS

MODELS = ["Original logistic", "Logistic regression", "CatBoost", "Competing risks"]
POLICIES = ["Maximize recall", "Minimize tickets"]
COLORS = ["#27558f", "#b04d1b"]


def plot_fleets(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.35))
    fig.subplots_adjust(left=.20, right=.97, bottom=.21, top=.78, wspace=.16)
    fig.text(.035, .935, "How often does each policy catch removals, and at what workload?",
             weight="bold", size=14, color="#1d2e41")
    fig.text(.035, .865, "Each dot is one fresh mock fleet. Diamonds pool all six fleets.", size=10)
    for ax, field, title, scale, target in zip(
        axes, ["event_recall", "alerts_per_100"], ["Removal events detected (%)", "New tickets per 100 active / day"],
        [100, 1], [80, 5],
    ):
        ax.axvline(target, ls="--", lw=1, color="#727b83")
        for i, model in enumerate(MODELS):
            for j, policy in enumerate(POLICIES):
                group = runs[(runs.model == model) & (runs.policy == policy)]
                if group.empty:
                    continue
                y = 3-i + (.16 if j == 0 else -.16)
                pooled = (100 * group.hits.sum() / group.events.sum() if field == "event_recall"
                          else 100 * group.alerts.sum() / group.active_component_days.sum())
                ax.scatter(scale * group[field], y + np.linspace(-.055,.055,len(group)), s=25,
                           color=COLORS[j], alpha=.58, linewidths=0)
                ax.scatter(pooled, y, marker="D", s=48, color=COLORS[j], edgecolor="white", linewidth=.8)
        ax.set(xlabel=title, ylim=(-.48,3.48), yticks=[3,2,1,0])
        ax.set_yticklabels(MODELS if ax is axes[0] else [])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e6e8", lw=.7)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(min(74, runs.event_recall.min()*100-2),101)
    axes[1].set_xscale("log")
    axes[1].set_xticks([.1,.2,.5,1,2,5], labels=["0.1","0.2","0.5","1","2","5"])
    axes[1].set_xlim(min(.09, runs.alerts_per_100.min()*.9),5.5)
    for j, policy in enumerate(POLICIES):
        fig.text(.22+j*.32, .055, policy, color=COLORS[j], weight="bold", size=10)
    fig.savefig(RESULTS / "fleet_policy_comparison.png", dpi=165, facecolor="white")
    plt.close(fig)


def publish_study(runs, pilot, curves):
    plot_fleets(runs)
    total_events = int(runs.drop_duplicates("fleet_seed").events.sum())
    fleet_count = runs.fleet_seed.nunique()
    selected = runs[(runs.model == pilot.model) & (runs.policy == pilot.policy)]
    early_warning = runs[(runs.model == "Competing risks") & (runs.policy == "Maximize recall")]
    passed = (selected.event_recall.ge(.8) & selected.peak_daily_per_100.le(5)).sum()
    eligible_hits = int(selected.hits_at_least_5_cycles.sum())
    baseline = runs[runs.model == "Original logistic"].groupby("policy")[["alerts", "hits"]].sum()
    body = [
        "# Planning the maintenance review queue",
        f"Three models and two ticket policies were compared on {fleet_count} fresh mock fleets containing "
        f"{total_events} unscheduled removals in total. The two operational questions were how many removals "
        "a planning team could anticipate and how many tickets it would have to investigate.",
        f"**Shadow-pilot candidate.** Validation selected {pilot.model} at {pilot.threshold:.2f} as the "
        f"lowest-workload eligible candidate, before the new fleets were scored. It detected {selected.hits.sum()}/{total_events} "
        f"events ({selected.hits.sum()/total_events:.1%}) across those fleets. Per-fleet recall ranged from "
        f"{selected.event_recall.min():.1%} to {selected.event_recall.max():.1%}. It met both targets on "
        f"{passed}/{fleet_count} fleets, with {selected.alerts.sum():,} tickets and at least five cycles of "
        f"notice for {eligible_hits}/{total_events} events.",
        f"**If planners need five cycles to arrange parts or a maintenance slot**, the competing-risk model "
        f"at {early_warning.threshold.iloc[0]:.2f} delivered that notice for "
        f"{early_warning.hits_at_least_5_cycles.sum()}/{total_events} events, with {early_warning.alerts.sum():,} tickets. "
        "Compare this option in shadow against the workload-first candidate. Five cycles is an illustrative "
        "lead requirement pending confirmation by planners.",
        "![Six-fleet comparison of removal recall and daily review workload](results/fleet_policy_comparison.png)",
        "## Results on fresh fleets",
    ]
    table = [
        "| Model | Validation policy | Cutoff | Events detected | Recall | Fleets meeting both targets | New tickets / 100 per day | Ticket precision | Events warned ≥5 cycles ahead | Peak open / 100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, policy), group in runs.groupby(["model","policy"],sort=False):
        events, hits, alerts = group[["events","hits","alerts"]].sum()
        passed = (group.event_recall.ge(.8) & group.peak_daily_per_100.le(5)).sum()
        table.append(f"| {model} | {policy} | {group.threshold.iloc[0]:.2f} | {hits}/{events} | {hits/events:.1%} "
                     f"| {passed}/{len(group)} | {100*alerts/group.active_component_days.sum():.2f} "
                     f"| {group.confirmed_tickets.sum()/alerts:.1%} | {group.hits_at_least_5_cycles.sum()/events:.1%} "
                     f"| {group.peak_open_per_100.max():.2f} |")
    body.extend(["\n".join(table),
        "Original logistic replays the previously published model at the two cutoffs determined by its original "
        "validation cohort. This reference isolates the alert-policy change on exactly the same six fleets. "
        "The other three candidates use the extended common training window. When both selection objectives "
        "choose the same cutoff, the two reported rows coincide.",
        f"Changing only the original model's ticket rule moved workload from "
        f"{baseline.loc['Maximize recall', 'alerts']:,} to {baseline.loc['Minimize tickets', 'alerts']:,} "
        f"tickets while detected events moved from {baseline.loc['Maximize recall', 'hits']} to "
        f"{baseline.loc['Minimize tickets', 'hits']}. Early low-score tickets can remain open when the signal becomes "
        "stronger. An example in the [original case analysis](REPORT.md#example-decisions) documents this mechanism.",
        "Recall, precision and ticket rate pool their respective counts. The five-cycle column asks what "
        "fraction of all removal events had at least five cycles of planning notice. The final column counts "
        "all outstanding tickets. The assessment cap is enforced on newly opened tickets every day.",
        "## Model choices",
        "**Logistic regression** estimates how the measured signals combine additively on the log-odds scale. "
        "**CatBoost** uses the same features to learn nonlinear responses and interactions. Both directly predict "
        "the 30-cycle label and weight positive records fourfold. **The competing-risk model** uses CatBoost "
        "as a shared three-class estimator across six five-cycle intervals. At each landmark it estimates conditional "
        "chances of continued service, planned removal and unscheduled removal. Each interval's unscheduled "
        "risk is multiplied by the estimated probability of still being in service at its start. The six contributions "
        "sum to the 30-cycle cumulative incidence. A planned exit removes the installation from subsequent risk sets. "
        "All six forecasts use only signals available at the original scoring time. The hazard fit uses natural class "
        "frequencies and the same fully observed training landmarks as the direct models. Probability calibration on real "
        "fleet records would be needed for cost estimates in currency.",
        "## How the comparison was locked",
        "The first assessment's training period contained zero planned exits. For this study, the same 80 training "
        "aircraft were observed through November 14, 2024, by which time the ledger contained 38 planned exits. "
        "Training landmarks had to mature by that cutoff. All three candidates were refit on those same landmarks. "
        "Nine candidate cutoffs were replayed on 40 other validation aircraft between November 15, 2024 and "
        "April 14, 2025, with warm-up and follow-up guards. The original test aircraft were excluded. "
        "One policy maximizes event recall under the hard daily cap and breaks ties by fewer tickets. "
        "The other minimizes tickets among cutoffs reaching 80% event recall under the same cap, with higher recall "
        "breaking ties. A model with no qualifying cutoff has no minimum-ticket policy. Higher thresholds break remaining ties.",
        "The stored thresholds were frozen before the six new fleets were scored. Each seed generates 40 aircraft "
        "for 365 days starting May 1, 2025. Aircraft and serial IDs are namespaced by seed. Each fleet is evaluated "
        "from November onward, with the same warm-up and outcome-maturity guards measured in flown cycles. The two "
        "policies see identical observations and events within each fleet. They differ in ticket timing, which changes "
        "when a 30-cycle open ticket suppresses another alert.",
        "The experiment measures variation in the same simulator. Changes in fault mechanisms, aircraft utilization "
        "or work-order practice require real observations. Five cycles is a planning sensitivity check. The MRO team "
        "would set its minimum useful notice and record engineer minutes and avoided disruption in a shadow pilot.",
        "[Frozen validation policies](results/locked_policies.csv) · [All validation operating points](results/study_validation.csv) "
        "· [Results for every fleet](results/fleet_policy_results.csv)",
    ])
    case_table = ["| Case from the original holdout | Serial | By 5 cycles | By 10 | By 20 | By 30 | Outcome |",
                  "|---|---|---:|---:|---:|---:|---|"]
    for _, case in curves.iterrows():
        case_table.append(f"| {case['case']} | {case.serial} | {case.by_5_cycles:.1%} | {case.by_10_cycles:.1%} "
                          f"| {case.by_20_cycles:.1%} | {case.by_30_cycles:.1%} | {case.outcome} |")
    body.extend(["## When might the removal occur?",
                 "The competing-risk model also estimates cumulative unscheduled-removal incidence at intermediate "
                 "horizons. These are the three cases chosen in the original assessment, scored using the observations "
                 "available that day. Case labels refer to the original logistic policy. Values are model estimates "
                 "on synthetic data. [All six horizons](results/horizon_examples.csv).", "\n".join(case_table)])
    (ROOT / "FLEET_STUDY.md").write_text("\n\n".join(body)+"\n",encoding="utf-8")
