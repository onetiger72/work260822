import tempfile
import unittest
import runpy
from datetime import datetime, timedelta, timezone
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

    def test_audit_checks_filled_actual_numbers_bonus_prize_and_status(self):
        history = dated_history(1)
        actual = sorted(int(history.at[0, f"번호{i}"]) for i in range(1, 7))
        bonus = int(history.at[0, "보너스"])
        wrong = next(n for n in range(1, 46) if n not in actual and n != bonus)
        predictions = pd.DataFrame([{
            "예측회차": 1, "세트": 1, "실제보너스": wrong, "당첨등수": "5등", "평가상태": "대기",
            **{f"번호{i}": n for i, n in enumerate(actual, 1)},
            **{f"실제번호{i}": n for i, n in enumerate([wrong] + actual[1:], 1)}
        }])
        stale = compare_saved_predictions(history, predictions)[0]["stored_evaluation_mismatches"]
        self.assertEqual({r["column"]: r["correct"] for r in stale},
                         {"실제번호1": actual[0], "실제보너스": bonus, "당첨등수": "1등", "평가상태": "평가완료"})

    def test_saved_and_feedback_reports_distinguish_scattered_six_from_exact_six(self):
        from benchmark_audit import load_module
        history = dated_history(3)
        actual = history.iloc[-1][[f"번호{i}" for i in range(1, 7)]].tolist()
        outsiders = [n for n in range(1, 46) if n not in actual][:6]
        tickets = [actual[:3] + outsiders[:3], actual[3:] + outsiders[3:]]
        predictions = pd.DataFrame([
            {"예측회차": 3, "세트": i + 1,
             **{f"번호{j}": n for j, n in enumerate(ticket, 1)},
             **{f"실제번호{j}": n for j, n in enumerate(actual, 1)}}
            for i, ticket in enumerate(tickets)
        ])
        record = compare_saved_predictions(history, predictions)[0]
        self.assertTrue(record["all_six_scattered"])
        self.assertEqual(record["best_main_hits"], 3)
        app = load_module("feedback_test_app", Path(__file__).resolve().parent / "로또 당첨번호 예측.py")
        summary, details = app.calculate_prediction_feedback(predictions)
        self.assertEqual(summary["all_six_scattered_draws"], 1)
        self.assertEqual(summary["exact_six_draws"], 0)
        self.assertEqual(summary["mean_best_main_hits"], 3)
        self.assertEqual(details.iloc[0]["전체포함본번호수"], 6)
        self.assertEqual(details.iloc[0]["분산손실"], 3)

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
            before_cutoff = datetime(2003, 1, 4, 10, 59, tzinfo=timezone.utc)
            report = audit(app, now=before_cutoff)
            self.assertTrue(report["ready_for_generation"])
            self.assertEqual(report["comparisons"][0]["main_hits"], [6])
            self.assertEqual(report["logic_review"]["target_draw"], 5)
            self.assertEqual([c.args[0] for c in app.get_lotto_draw.call_args_list], [3, 4])
            self.assertEqual(before, [p.read_bytes() for p in (history_path, prediction_path)])
            self.assertEqual({p.name for p in root.iterdir()}, {"history.csv", "predictions.csv"})
            app.generate_prediction_sets.assert_not_called()
            # 같은 4회 결과를 공식 대조해도 한국시간 토요일 20시에는 5회 접수가 닫힌다.
            closed = audit(app, now=before_cutoff + timedelta(minutes=1))
            self.assertEqual(closed["official_status"], "matched")
            self.assertFalse(closed["ready_for_generation"])
            self.assertIn("판매 마감", closed["prediction_window_error"])
            app.get_lotto_draw.return_value = None
            app.get_lotto_draw.side_effect = None
            report = audit(app, now=before_cutoff)
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
