"""Landmark competing risks over six successive five-flight-cycle intervals."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import CATEGORICAL, HORIZON, INPUTS

ENDS = tuple(range(5, HORIZON + 1, 5))


def training_periods(rows, ledger):
    exits = ledger.set_index("install_id", verify_integrity=True)
    distance = rows.install_id.map(exits.removal_cycle) - rows.aircraft_cycle
    reason = rows.install_id.map(exits.reason)
    periods, targets = [], []
    for end in ENDS:
        at_risk = distance.isna() | distance.gt(end - 5)
        period = rows.loc[at_risk, INPUTS].assign(interval_end=end)
        inside = distance[at_risk].le(end)
        # State 0 continues, state 1 exits as planned, state 2 exits unscheduled.
        target = np.select([inside & reason[at_risk].eq("planned"),
                            inside & reason[at_risk].eq("unscheduled")], [1, 2], default=0)
        periods.append(period)
        targets.append(target)
    return pd.concat(periods, ignore_index=True), np.concatenate(targets)


def cumulative_incidence(hazards):
    survival = np.ones(len(hazards[0]))
    incidence = np.zeros_like(survival)
    curve = []
    for interval in hazards:
        incidence += survival * interval[:, 2]
        survival *= interval[:, 0]
        curve.append(incidence.copy())
    return np.column_stack(curve)


class CompetingRiskModel:
    """Use only measurements available at the landmark for every future interval."""

    def fit(self, rows, ledger):
        x, y = training_periods(rows, ledger)
        if set(np.unique(y)) != {0, 1, 2}:
            raise ValueError("Training landmarks must contain both planned and unscheduled exits")
        self.model = CatBoostClassifier(iterations=160, depth=5, learning_rate=.085,
                                        cat_features=CATEGORICAL, one_hot_max_size=10,
                                        loss_function="MultiClass", classes_count=3,
                                        thread_count=4, random_seed=190, verbose=False,
                                        allow_writing_files=False)
        self.model.fit(x, y)
        return self

    def risk_curve(self, features):
        hazards = [self.model.predict_proba(features[INPUTS].assign(interval_end=end), thread_count=4)
                   for end in ENDS]
        return cumulative_incidence(hazards)

    def predict_proba(self, features):
        risk = self.risk_curve(features)[:, -1]
        return np.column_stack([1 - risk, risk])
