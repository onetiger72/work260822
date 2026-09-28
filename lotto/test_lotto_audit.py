import tempfile
import unittest
import runpy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from audit_predictions import audit, compare_saved_predictions
from test_lotto_portfolio import make_history


def dated_history(count):
    history = make_history(count)
    history["날짜"] = [(datetime(2002, 12, 7) + timedelta(weeks=i)).strftime("%Y%m%d")
                       for i in range(count)]
    return history


class AuditTests(unittest.TestCase):
    def test_main_audit_flag_routes_to_audit_without_running_generation(self):
        script = Path(__file__).resolve().parent / "로또 당첨번호 예측.py"
        with patch("sys.argv", [str(script), "--audit-only"]), patch("audit_predictions.audit") as review:
            runpy.run_path(str(script), run_name="__main__")
        review.assert_called_once()
        self.assertFalse(review.call_args.kwargs["offline"])

    def test_recounts_main_and_bonus_instead_of_trusting_saved_totals(self):
        history = dated_history(3)
        row = history.iloc[-1]
        numbers = [int(row[f"번호{i}"]) for i in range(1, 6)] + [int(row["보너스"])]
        predictions = pd.DataFrame([{"예측회차": 3, "세트": 1, "본번호일치수": 6,
                                     **{f"번호{i}": n for i, n in enumerate(numbers, 1)}}])
        report = compare_saved_predictions(history, predictions)[0]
        self.assertEqual(report["main_hits"], [5])
        self.assertEqual(report["bonus_hits"], [1])
        self.assertEqual(report["stored_evaluation_mismatches"][0]["correct"], 5)

    def test_audit_fetches_missing_draw_without_changing_csv_or_generating_tickets(self):
        history = dated_history(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path, prediction_path = root / "history.csv", root / "predictions.csv"
            history.iloc[:3].to_csv(history_path, index=False)
            predictions = history.iloc[-1:].rename(columns={"회차": "예측회차"}).assign(세트=1)
            predictions.to_csv(prediction_path, index=False)
            before = [p.read_bytes() for p in (history_path, prediction_path)]
            app = SimpleNamespace(CSV_FILE=history_path, PREDICTION_FILE=prediction_path,
                                  get_lotto_draw=Mock(side_effect=lambda n: history.iloc[n - 1].to_dict()),
                                  calculate_analysis_scores=lambda _: {},
                                  generate_prediction_sets=Mock(side_effect=AssertionError("no tickets")))
            with patch("audit_predictions.latest_completed_draw", return_value=4):
                report = audit(app)
            self.assertTrue(report["ready_for_generation"])
            self.assertEqual(report["comparisons"][0]["main_hits"], [6])
            self.assertEqual(report["logic_review"]["target_draw"], 5)
            self.assertEqual([c.args[0] for c in app.get_lotto_draw.call_args_list], [3, 4])
            self.assertEqual(before, [p.read_bytes() for p in (history_path, prediction_path)])
            self.assertEqual({p.name for p in root.iterdir()}, {"history.csv", "predictions.csv"})
            app.generate_prediction_sets.assert_not_called()
            app.get_lotto_draw.return_value = None
            app.get_lotto_draw.side_effect = None
            with patch("audit_predictions.latest_completed_draw", return_value=4):
                report = audit(app)
            self.assertFalse(report["ready_for_generation"])
            self.assertEqual(report["official_status"], "unavailable")
            self.assertEqual({p.name for p in root.iterdir()}, {"history.csv", "predictions.csv"})

    def test_duplicate_and_fractional_saved_numbers_are_rejected(self):
        history = dated_history(3)
        predictions = history.iloc[-1:].rename(columns={"회차": "예측회차"}).assign(세트=1)
        with self.assertRaises(ValueError):
            compare_saved_predictions(history, pd.concat([predictions, predictions]))
        predictions["번호1"] = 1.5
        with self.assertRaises(ValueError):
            compare_saved_predictions(history, predictions)


if __name__ == "__main__":
    unittest.main()
