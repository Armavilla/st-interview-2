import unittest

import numpy as np
import pandas as pd

from features import build_features, derive_labels
from policy import event_leads, issue_alerts, split_cohorts


class AssessmentChecks(unittest.TestCase):
    def test_future_window_planned_exit_and_censoring(self):
        obs = pd.DataFrame({"aircraft_id": ["AC-001"] * 8,
                            "install_id": [1, 1, 1, 2, 3, 3, 4, 4],
                            "aircraft_cycle": [99, 100, 129, 135, 145, 150, 120, 121]})
        events = pd.DataFrame({"install_id": [1, 2], "removal_cycle": [130, 147],
                               "reason": ["unscheduled", "planned"]})
        expected = pd.Series([0, 1, 1, 0, pd.NA, pd.NA, 0, pd.NA], dtype="Int8")
        pd.testing.assert_series_equal(derive_labels(obs, events), expected, check_names=False)

    def test_future_reading_and_old_installation_cannot_change_current_features(self):
        base = {"aircraft_id": "AC-000", "position": 1, "part_family": "A",
                "airframe_family": "F", "cycles_today": 1, "cycles_since_install": 1,
                "cycles_since_service": 1, "temperature_c": 65., "fault_code": "NONE"}
        obs = pd.DataFrame([{**base, "date": f"2024-01-0{i + 1}", "aircraft_cycle": i + 1,
                             "install_id": inst, "serial": f"S-{inst}", "vibration": v}
                           for i, (inst, v) in enumerate([(1, 2.), (1, 4.), (2, -99.), (2, 3.)])])
        full = build_features(obs)
        pd.testing.assert_frame_equal(full.iloc[:2], build_features(obs.iloc[:2]))
        self.assertEqual(full.vibration_mean3.iloc[1], 3.)
        self.assertTrue(np.isnan(full.vibration_mean3.iloc[2]))
        self.assertEqual(full.vibration_missing.iloc[2], 1)
        self.assertEqual(full.history_days.iloc[2], 1)
        self.assertEqual(full.vibration_mean3.iloc[3], 3.)

    def test_daily_cap_and_expiry_for_tied_scores(self):
        rows = pd.DataFrame([(day, i, cycle) for day, cycle in [("A", 10), ("B", 39), ("C", 40)]
                             for i in range(40)], columns=["date", "install_id", "aircraft_cycle"])
        fired = issue_alerts(rows, np.ones(len(rows)), .5)
        self.assertEqual(rows.loc[fired, "install_id"].tolist(), [0, 1, 2, 3, 0, 1])
        self.assertEqual(rows.loc[fired].groupby("date").size().tolist(), [2, 2, 2])
        self.assertFalse(issue_alerts(rows.iloc[:19], np.ones(19), .5).any())

    def test_event_matching_counts_each_exit_once_and_excludes_same_cycle(self):
        rows = pd.DataFrame({"install_id": [1, 1, 2, 3], "aircraft_cycle": [70, 99, 100, 69]})
        ledger = pd.DataFrame({"install_id": [1, 2, 3], "removal_cycle": [100, 100, 100]})
        leads = event_leads(rows, np.ones(4, dtype=bool), ledger)
        self.assertEqual(leads.to_dict(), {1: 30})

    def test_transferred_serial_across_cohorts_is_rejected(self):
        sample = pd.DataFrame([
            ["AC-001", "shared", 1, 100, "2024-01-01"],
            ["AC-001", "shared", 1, 140, "2024-06-30"],
            ["AC-081", "V", 2, 180, "2024-07-03"],
            ["AC-121", "shared", 3, 300, "2024-11-16"],
        ], columns=["aircraft_id", "serial", "install_id", "aircraft_cycle", "date"])
        with self.assertRaisesRegex(ValueError, "share serial"):
            split_cohorts(sample)


if __name__ == "__main__":
    unittest.main()
