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
attach_cluster_column = _DID._attach_cluster_column
aggregate_att = _DID.aggregate_att
validate_differences_results = _DID._validate_differences_results


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

    def test_accepts_an_already_built_panel(self):
        panel = build_panel(self._frame([2009, 2010]), cluster_col=None)

        rebuilt = build_panel(panel, cluster_col=None)

        pd.testing.assert_frame_equal(rebuilt, panel)

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


class AttachClusterColumnTests(unittest.TestCase):
    def test_encodes_string_site_labels_for_differences_bootstrap(self):
        index = pd.MultiIndex.from_product(
            [[1, 2], [2019, 2020]], names=["unit_id", "year"]
        )
        data = pd.DataFrame(
            {"site_id": ["site-b", "site-b", "site-a", "site-a"]}, index=index
        )

        class FakeATT:
            pass

        att = FakeATT()
        att.data = data
        att._data_matrix = pd.DataFrame({"burned": 0}, index=index)

        attach_cluster_column(att, "site_id")

        # differences reserves zero in its cluster-membership construction.  In
        # particular, a group-time comparison containing only the first cluster
        # must still have one cluster column rather than an empty matrix.
        self.assertEqual(att._data_matrix["site_id"].tolist(), [1, 1, 2, 2])
        self.assertGreater(att._data_matrix["site_id"].min(), 0)
        self.assertTrue(pd.api.types.is_integer_dtype(att._data_matrix["site_id"]))


class AggregateAttDiagnosticsTests(unittest.TestCase):
    def test_distinguishes_cluster_bootstrap_failure_from_empty_att(self):
        class FakeATT:
            peatfire_cluster_var = "site_id"
            peatfire_boot_iterations = 100
            peatfire_random_state = 42

            def aggregate(self, kind, cluster_var, boot_iterations, random_state):
                if cluster_var:
                    raise IndexError("index 0 is out of bounds for axis 1 with size 0")
                return pd.DataFrame({"ATT": [0.1]})

        with self.assertRaisesRegex(
            ValueError, "point aggregation.*multiplier bootstrap failed"
        ):
            aggregate_att(FakeATT(), kind="simple")

    def test_keeps_support_diagnosis_when_analytic_retry_also_fails(self):
        class FakeATT:
            peatfire_cluster_var = "site_id"
            peatfire_boot_iterations = 100
            peatfire_random_state = 42

            def aggregate(self, kind, cluster_var, boot_iterations, random_state):
                raise IndexError("index 0 is out of bounds for axis 1 with size 0")

        with self.assertRaisesRegex(ValueError, "support/bookkeeping problem"):
            aggregate_att(FakeATT(), kind="simple")


class FittedAttValidationTests(unittest.TestCase):
    def test_reports_all_nan_post_treatment_effects_before_aggregation(self):
        from collections import namedtuple

        Element = namedtuple("Element", "cohort time ATT")

        class Aggregate:
            ntl = [
                Element(2019, 2018, float("nan")),
                Element(2019, 2019, float("nan")),
                Element(2019, 2020, float("nan")),
            ]

        class FakeATT:
            _result_dict = {"full_sample": {"aggregate_inst": Aggregate()}}

        with self.assertRaisesRegex(
            ValueError, "no finite post-treatment ATT estimates.*0/2"
        ):
            validate_differences_results(FakeATT())

if __name__ == "__main__":
    unittest.main()
