import math
import unittest

from research.domain_gap import analyse_predictions, analyse_summary, regression_metrics


class DomainGapTest(unittest.TestCase):
    def test_regression_metrics_perfect_prediction(self):
        metrics = regression_metrics([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
        self.assertEqual(metrics, {"mae": 0.0, "rmse": 0.0, "r2": 1.0})

    def test_prediction_analysis_is_grouped_and_reproducible(self):
        rows = [
            {"model": "a", "domain": "sim", "y_true": str(i), "y_pred": str(i + 1)}
            for i in range(1, 11)
        ]
        first = analyse_predictions(rows, iterations=100, seed=7)
        second = analyse_predictions(rows, iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["n"], 10)
        self.assertAlmostEqual(first[0]["mae"], 1.0)
        self.assertTrue(math.isfinite(first[0]["r2_ci_low"]))

    def test_summary_selects_real_domain_not_simulation_champion(self):
        rows = [
            {"model": "sim_champion", "domain": "sim", "mae": "1", "rmse": "1", "r2": "0.99"},
            {"model": "sim_champion", "domain": "real", "mae": "30", "rmse": "35", "r2": "-2"},
            {"model": "robust", "domain": "sim", "mae": "5", "rmse": "6", "r2": "0.85"},
            {"model": "robust", "domain": "real", "mae": "15", "rmse": "20", "r2": "0.1"},
        ]
        report = analyse_summary(rows)
        self.assertEqual(report["best_existing_baseline"]["model"], "robust")


if __name__ == "__main__":
    unittest.main()

