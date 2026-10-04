"""분산 효과와 학습 효과를 같은 회차·시드에서 구분하는지 확인한다."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from benchmark_audit import load_module, paired_comparison, run_benchmark
from test_lotto_audit import dated_history


class BenchmarkTests(unittest.TestCase):
    def test_uniform_fallback_matches_balanced_control_exactly(self):
        old = load_module("baseline_for_test", Path(__file__).resolve().parent / "analysis_cache/portfolio_v4.py")
        def constant_scores(prefix):
            return {"constant": {n: 0.5 for n in range(1, 46)}}
        with redirect_stdout(io.StringIO()):
            report = run_benchmark(dated_history(202), constant_scores, old, draws=2, repetitions=2)
        self.assertEqual(set(report["summaries"]), {"v4", "model", "balanced_uniform"})
        self.assertEqual(report["balanced_baseline"]["identical_ticket_draws"], 2)
        self.assertEqual(report["balanced_baseline"]["active_signal_draws"], 0)
        self.assertEqual(report["uniform_runs"]["repetitions"], 2)
        for metric in report["model_vs_balanced_uniform"].values():
            self.assertEqual(metric["mean_model_minus_baseline"], 0)
            self.assertEqual(metric["ties"], 2)

    def test_paired_summary_keeps_sign_and_rejects_mismatched_rounds(self):
        model = [{"target_draw": 201, "hits": [2, 2]}, {"target_draw": 202, "hits": [0, 0]}]
        baseline = [{"target_draw": 201, "hits": [1, 1]}, {"target_draw": 202, "hits": [2, 2]}]
        summary = paired_comparison(model, baseline)
        self.assertEqual(summary["mean_ticket_hits"]["mean_model_minus_baseline"], -0.5)
        self.assertEqual(summary["best_hits"]["wins"], 1)
        self.assertEqual(summary["best_hits"]["losses"], 1)
        baseline[0]["target_draw"] = 200
        with self.assertRaises(ValueError):
            paired_comparison(model, baseline)


if __name__ == "__main__":
    unittest.main()
