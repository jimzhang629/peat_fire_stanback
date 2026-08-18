import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


# Import this dependency-light module without executing peatfire.__init__, which
# eagerly imports optional geospatial packages not needed by these unit tests.
# Register it in sys.modules before executing: @dataclass resolves annotations
# through sys.modules[cls.__module__] and fails on an unregistered module.
_SPEC = spec_from_file_location(
    "peatfire_power", Path(__file__).parents[1] / "src/peatfire/modeling/power.py"
)
_POWER = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _POWER
_SPEC.loader.exec_module(_POWER)

DesignSpec = _POWER.DesignSpec
design_from_panel = _POWER.design_from_panel
design_summary = _POWER.design_summary
did_site_year = _POWER.did_site_year
minimum_detectable_effect = _POWER.minimum_detectable_effect
power_curve = _POWER.power_curve
randomization_inference = _POWER.randomization_inference
sample_size_curve = _POWER.sample_size_curve
simulate_power = _POWER.simulate_power
site_year_panel = _POWER.site_year_panel


def pixel_panel(rows):
    """Build a minimal pixel-year panel: (site, year, cohort, n_pixels, n_burned)."""
    out = []
    entity = 0
    for site, year, g, n_pixels, n_burned in rows:
        for i in range(n_pixels):
            out.append(
                {
                    "site_id": site,
                    "year": year,
                    "g": g,
                    "entity": f"{site}_{i}",
                    "burned": 1 if i < n_burned else 0,
                }
            )
        entity += 1
    return pd.DataFrame(out)


class SiteYearPanelTests(unittest.TestCase):
    def test_collapses_to_one_row_per_site_year_with_fractions(self):
        panel = pixel_panel(
            [
                ("A", 2020, 2022, 10, 2),
                ("A", 2023, 2022, 10, 0),
                ("B", 2020, 0, 10, 5),
                ("B", 2023, 0, 10, 1),
            ]
        )
        sy = site_year_panel(panel)
        self.assertEqual(len(sy), 4)
        a2020 = sy[(sy.site_id == "A") & (sy.year == 2020)].iloc[0]
        self.assertAlmostEqual(a2020["burn_fraction"], 0.2)
        self.assertEqual(a2020["post"], 0)  # 2020 < restoration year 2022
        a2023 = sy[(sy.site_id == "A") & (sy.year == 2023)].iloc[0]
        self.assertEqual(a2023["post"], 1)
        # never-treated site is never "post" even in late years
        self.assertEqual(sy[sy.site_id == "B"]["post"].sum(), 0)

    def test_missing_column_raises_actionable_error(self):
        with self.assertRaisesRegex(KeyError, "prepare_panel"):
            site_year_panel(pd.DataFrame({"site_id": ["A"], "year": [2020]}))


class DesignSummaryTests(unittest.TestCase):
    def test_flags_site_with_no_pre_period_as_unusable(self):
        panel = pixel_panel(
            [
                ("A", 2019, 2019, 4, 0),  # restored in the first outcome year
                ("A", 2020, 2019, 4, 1),
                ("B", 2019, 2021, 4, 1),
                ("B", 2020, 2021, 4, 0),
                ("B", 2021, 2021, 4, 0),
            ]
        )
        table = design_summary(site_year_panel(panel))
        a = table[table.site_id == "A"].iloc[0]
        b = table[table.site_id == "B"].iloc[0]
        self.assertFalse(bool(a["usable_for_did"]))  # no pre-restoration year
        self.assertTrue(bool(b["usable_for_did"]))
        total = table[table.site_id == "TOTAL"].iloc[0]
        self.assertEqual(total["usable_for_did"], "1/2 sites")


class DidSiteYearTests(unittest.TestCase):
    def _panel(self, treated_post_fraction):
        """Two treated + two control sites; controls burn 0.2 in every year."""
        rows = []
        for site, g in [("T1", 2022), ("T2", 2022), ("C1", 0), ("C2", 0)]:
            for year in range(2019, 2025):
                post = g > 0 and year >= g
                frac = treated_post_fraction if post else 0.2
                rows.append(
                    {
                        "site_id": site,
                        "year": year,
                        "g": g,
                        "post": int(post),
                        "treated": int(post),
                        "burn_fraction": frac,
                    }
                )
        return pd.DataFrame(rows)

    def test_recovers_a_known_effect_exactly(self):
        res = did_site_year(self._panel(0.05))
        self.assertAlmostEqual(res["estimate"], -0.15, places=6)
        self.assertEqual(res["n_clusters"], 4)
        self.assertEqual(res["df"], 3)  # G - 1, not the pixel count

    def test_reports_zero_effect_when_nothing_changes(self):
        res = did_site_year(self._panel(0.2))
        self.assertAlmostEqual(res["estimate"], 0.0, places=6)

    def test_degrees_of_freedom_track_clusters_not_observations(self):
        res = did_site_year(self._panel(0.05))
        self.assertLess(res["df"], res["n_obs"])


class RandomizationInferenceTests(unittest.TestCase):
    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(0)
        rows = []
        for i, (site, g) in enumerate(
            [("T1", 2022), ("T2", 2022), ("C1", 0), ("C2", 0), ("C3", 0)]
        ):
            for year in range(2019, 2025):
                post = int(g > 0 and year >= g)
                rows.append(
                    {
                        "site_id": site,
                        "year": year,
                        "g": g,
                        "post": post,
                        "treated": post,
                        "burn_fraction": float(rng.random() * 0.1),
                    }
                )
        res = randomization_inference(pd.DataFrame(rows), n_permutations=200)
        self.assertGreater(res["p_value"], 0.05)
        self.assertGreater(res["n_permutations"], 0)

    def test_returns_a_p_value_when_a_site_year_has_no_fire(self):
        # The case that makes the cluster bootstrap emit NaN intervals.
        rows = []
        for site, g in [("T1", 2022), ("C1", 0), ("C2", 0)]:
            for year in range(2019, 2025):
                post = int(g > 0 and year >= g)
                rows.append(
                    {
                        "site_id": site,
                        "year": year,
                        "g": g,
                        "post": post,
                        "treated": post,
                        "burn_fraction": 0.0 if year != 2021 else 0.3,
                    }
                )
        res = randomization_inference(pd.DataFrame(rows), n_permutations=100)
        self.assertTrue(np.isfinite(res["p_value"]))


class SimulatePowerTests(unittest.TestCase):
    def _spec(self, n_sites, n_years, fire_prob):
        sites = [f"s{i}" for i in range(n_sites)]
        years = list(range(2000, 2000 + n_years))
        cohorts = {s: years[n_years // 2] for s in sites[: n_sites // 2]}
        return DesignSpec(
            sites=sites,
            years=years,
            cohorts=cohorts,
            site_fire_prob=fire_prob,
            burn_fraction_draws=np.array([0.1, 0.3, 0.5]),
        )

    def test_no_fire_means_no_power(self):
        res = simulate_power(self._spec(6, 6, 0.0), reduction=1.0, n_sims=25)
        self.assertEqual(res["power"], 0.0)
        self.assertEqual(res["n_informative"], 0)

    def test_tiny_design_cannot_detect_even_total_prevention(self):
        # 6 sites x 6 years at a realistic NC peat fire rate: the real study.
        res = simulate_power(self._spec(6, 6, 0.05), reduction=1.0, n_sims=200)
        self.assertLess(res["power"], 0.8)

    def test_power_increases_with_sample_size(self):
        small = simulate_power(self._spec(6, 6, 0.2), reduction=0.8, n_sims=200)
        large = simulate_power(self._spec(60, 24, 0.2), reduction=0.8, n_sims=200)
        self.assertGreater(large["power"], small["power"])

    def test_power_increases_with_effect_size(self):
        spec = self._spec(40, 20, 0.2)
        weak = simulate_power(spec, reduction=0.1, n_sims=200)
        strong = simulate_power(spec, reduction=0.9, n_sims=200)
        self.assertGreater(strong["power"], weak["power"])

    def test_false_positive_rate_is_near_alpha_under_the_null(self):
        spec = self._spec(20, 12, 0.25)
        res = simulate_power(spec, reduction=0.0, n_sims=400, alpha=0.05)
        self.assertLess(res["power"], 0.15)  # nominal 0.05, allowing MC slack

    def test_power_curve_returns_one_row_per_effect(self):
        curve = power_curve(
            self._spec(10, 8, 0.2), reductions=(0.25, 0.75), n_sims=25
        )
        self.assertEqual(list(curve["reduction"]), [0.25, 0.75])


class MinimumDetectableEffectTests(unittest.TestCase):
    def test_returns_none_when_even_full_prevention_is_undetectable(self):
        spec = DesignSpec(
            sites=["a", "b", "c"],
            years=list(range(2019, 2025)),
            cohorts={"a": 2022},
            site_fire_prob=0.02,
            burn_fraction_draws=np.array([0.2]),
        )
        res = minimum_detectable_effect(spec, reductions=(0.5, 1.0), n_sims=100)
        self.assertIsNone(res["mde"])
        self.assertLess(res["power_at_full_prevention"], 0.8)

    def test_finds_an_mde_when_the_design_is_large_enough(self):
        spec = DesignSpec(
            sites=[f"s{i}" for i in range(80)],
            years=list(range(2000, 2030)),
            cohorts={f"s{i}": 2015 for i in range(40)},
            site_fire_prob=0.3,
            burn_fraction_draws=np.array([0.2, 0.4]),
        )
        res = minimum_detectable_effect(spec, reductions=(0.5, 1.0), n_sims=100)
        self.assertIsNotNone(res["mde"])
        self.assertLessEqual(res["mde"], 1.0)


class DesignFromPanelTests(unittest.TestCase):
    def test_reads_per_year_fire_probability_off_the_panel(self):
        rows = []
        for site in ["A", "B", "C", "D"]:
            for year in (2019, 2020):
                rows.append(
                    {
                        "site_id": site,
                        "year": year,
                        "g": 0,
                        "post": 0,
                        "treated": 0,
                        # every site burns in 2020, none in 2019
                        "burn_fraction": 0.25 if year == 2020 else 0.0,
                    }
                )
        spec = design_from_panel(pd.DataFrame(rows))
        self.assertEqual(spec.year_prob(2019), 0.0)
        self.assertEqual(spec.year_prob(2020), 1.0)
        self.assertEqual(sorted(spec.years), [2019, 2020])


class SampleSizeCurveTests(unittest.TestCase):
    def test_power_rises_with_site_years(self):
        spec = DesignSpec(
            sites=["a"],
            years=list(range(2010, 2016)),
            cohorts={},
            site_fire_prob=0.3,
            burn_fraction_draws=np.array([0.2, 0.4]),
        )
        curve = sample_size_curve(
            spec,
            reduction=0.8,
            site_counts=(6, 60),
            year_counts=(12,),
            n_sims=100,
        )
        small = curve[curve.n_sites == 6]["power"].iloc[0]
        large = curve[curve.n_sites == 60]["power"].iloc[0]
        self.assertGreater(large, small)

    def test_every_treated_site_keeps_a_pre_and_post_period(self):
        spec = DesignSpec(
            sites=["a"],
            years=list(range(2010, 2022)),
            cohorts={},
            site_fire_prob=0.3,
            burn_fraction_draws=np.array([0.3]),
        )
        curve = sample_size_curve(
            spec, reduction=0.5, site_counts=(8,), year_counts=(12,), n_sims=5
        )
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve["n_treated_sites"].iloc[0], 4)


if __name__ == "__main__":
    unittest.main()
