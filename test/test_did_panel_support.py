import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


# Import this dependency-light module without executing peatfire.__init__, which
# eagerly imports optional geospatial packages not needed by these unit tests.
_SPEC = spec_from_file_location(
    "peatfire_did", Path(__file__).parents[1] / "src/peatfire/modeling/did.py"
)
_DID = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DID)
build_panel = _DID.build_panel
restrict_to_supported_cohorts = _DID.restrict_to_supported_cohorts
panel_support_table = _DID.panel_support_table


class BuildPanelSupportTests(unittest.TestCase):
    def _frame(self, treated_years, cohort=2010):
        rows = [
            {"unit_id": 1, "year": year, "burned": 0, "g": cohort}
            for year in treated_years
        ]
        rows += [
            {"unit_id": 2, "year": year, "burned": 0, "g": 0}
            for year in treated_years
        ]
        return pd.DataFrame(rows)

    def test_rejects_panel_with_only_always_treated_entities(self):
        with self.assertRaisesRegex(ValueError, "No treated entity.*always treated"):
            build_panel(self._frame([2010, 2011]), cluster_col=None)

    def test_rejects_panel_whose_treatment_is_after_coverage(self):
        with self.assertRaisesRegex(ValueError, "treatment after their last"):
            build_panel(self._frame([2008, 2009]), cluster_col=None)

    def test_accepts_treated_entity_spanning_its_cohort(self):
        panel = build_panel(self._frame([2009, 2010]), cluster_col=None)

        self.assertEqual(panel.index.names, ["unit_id", "year"])

    def test_rejects_time_varying_cohort(self):
        frame = self._frame([2009, 2010])
        frame.loc[(frame["unit_id"] == 1) & (frame["year"] == 2009), "g"] = 0

        with self.assertRaisesRegex(ValueError, "fixed first-treatment year"):
            build_panel(frame, cluster_col=None)


class RestrictToSupportedCohortsTests(unittest.TestCase):
    def test_fireccis311_keeps_only_bracketed_cohorts_and_controls(self):
        frame = pd.DataFrame(
            {"End_Yr": [2019, 2021, 2023, 2026, None], "value": range(5)}
        )

        with self.assertWarnsRegex(UserWarning, "excluding cohorts"):
            result = restrict_to_supported_cohorts(frame, range(2019, 2025))

        self.assertEqual(result["value"].tolist(), [1, 2, 4])

    def test_longer_mcd64a1_window_recovers_2019_cohort(self):
        frame = pd.DataFrame({"End_Yr": [2019, 2021, 2023, 2026, None]})

        with self.assertWarns(UserWarning):
            result = restrict_to_supported_cohorts(frame, range(2001, 2025))

        self.assertEqual(result["End_Yr"].dropna().tolist(), [2019, 2021, 2023])

    def test_rejects_invalid_cohort_instead_of_treating_it_as_control(self):
        frame = pd.DataFrame({"End_Yr": [2021, "unknown", None]})

        with self.assertRaisesRegex(ValueError, "non-numeric cohort"):
            restrict_to_supported_cohorts(frame, range(2019, 2025))


class PanelSupportTableTests(unittest.TestCase):
    def test_reports_effective_response_window_and_spanning_entities(self):
        frame = pd.DataFrame(
            [
                {"unit_id": 1, "year": 2018, "burned": 0, "g": 2019},
                {"unit_id": 1, "year": 2019, "burned": 1, "g": 2019},
                {"unit_id": 2, "year": 2019, "burned": 0, "g": 2019},
                {"unit_id": 3, "year": 2018, "burned": 0, "g": 0},
                {"unit_id": 3, "year": 2019, "burned": None, "g": 0},
            ]
        )

        result = panel_support_table(frame).set_index("g")

        self.assertEqual(result.loc[2019, "spanning_entities"], 1)
        self.assertEqual(result.loc[2019, "pre_entities"], 1)
        self.assertEqual(result.loc[2019, "post_entities"], 2)
        self.assertEqual(result.loc[0, "last_outcome_year"], 2018)


if __name__ == "__main__":
    unittest.main()
