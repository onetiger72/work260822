"""Persistent output is limited to history and prediction CSVs."""
import ast
import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lotto_portfolio import select_portfolio
from lotto_validation import final_review
from lotto_storage import CsvDataError, load_history, load_predictions
from test_lotto_audit import dated_history

BASE = Path(__file__).resolve().parent


def load_app():
    spec = importlib.util.spec_from_file_location("storage_test_app", BASE / "로또 당첨번호 예측.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


class StorageTests(unittest.TestCase):
    def test_missing_file_is_distinct_from_empty_or_malformed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            self.assertTrue(load_history(path, missing_ok=True).empty)
            self.assertTrue(load_predictions(path, missing_ok=True).empty)
            for raw in [b"", b'"unclosed', b'a,a\n1,2\n', b'a,b\n1\n']:
                path.write_bytes(raw)
                for loader in [load_history, load_predictions]:
                    with self.subTest(raw=raw, loader=loader), self.assertRaises(CsvDataError):
                        loader(path, missing_ok=True)
                    self.assertEqual(path.read_bytes(), raw)

    def test_corrupt_history_stops_before_fetch_and_preserves_original(self):
        app = load_app()
        history = dated_history(3)
        corrupt = pd.concat([history, history.iloc[-1:]], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            corrupt.to_csv(path, index=False)
            before = path.read_bytes()
            with patch.object(app, "CSV_FILE", str(path)), patch.object(app, "get_lotto_draw") as fetch:
                with self.assertRaises(CsvDataError):
                    app.collect_all_lotto(query_round=4, sleep_time=0)
                fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_partial_fetch_and_invalid_official_data_never_save(self):
        app = load_app()
        history = dated_history(3)
        bad_date = history.iloc[1].to_dict()
        bad_date["날짜"] = "20020101"
        cases = [[history.iloc[1].to_dict(), None], [history.iloc[0].to_dict()], [bad_date]]
        for responses in cases:
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "history.csv"
                history.iloc[:1].to_csv(path, index=False)
                before = path.read_bytes()
                with patch.object(app, "CSV_FILE", str(path)), \
                        patch.object(app, "get_lotto_draw", side_effect=responses), \
                        patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("must not save")):
                    with self.assertRaises(ValueError):
                        app.collect_all_lotto(query_round=3, sleep_time=0)
                self.assertEqual(path.read_bytes(), before)

    def test_successful_collection_appends_only_fully_valid_history(self):
        app = load_app()
        history = dated_history(3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            history.iloc[:1].to_csv(path, index=False)
            with patch.object(app, "CSV_FILE", str(path)), \
                    patch.object(app, "get_lotto_draw", side_effect=lambda n: history.iloc[n - 1].to_dict()) as fetch:
                result = app.collect_all_lotto(query_round=3, sleep_time=0)
            self.assertEqual([call.args[0] for call in fetch.call_args_list], [2, 3])
            self.assertEqual(result["회차"].tolist(), [1, 2, 3])
            pd.testing.assert_frame_equal(load_history(path)[app.NUMBER_COLUMNS], history[app.NUMBER_COLUMNS])
            self.assertEqual({p.name for p in Path(folder).iterdir()}, {"history.csv"})

    def test_bad_prediction_identity_is_never_guessed_truncated_or_deleted(self):
        app = load_app()
        history = dated_history(3)
        good = pd.DataFrame([{"예측회차": 3, "세트": 1,
                              **dict(zip(app.NUMBER_COLUMNS, range(1, 7)))}])
        cases = [good.drop(columns="예측회차"), good.drop(columns="세트"), pd.concat([good, good])]
        for column, value in [("번호1", 1.9), ("번호1", True), ("번호1", float("nan")),
                              ("예측회차", 3.5), ("세트", -1), ("번호1", 46)]:
            cases.append(good.assign(**{column: value}))
        for frame in cases:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "predictions.csv"
                frame.to_csv(path, index=False)
                before = path.read_bytes()
                with patch.object(app, "get_lotto_draw") as fetch, \
                        patch.object(app, "generate_prediction_sets") as generate:
                    with self.assertRaises(CsvDataError):
                        app.save_prediction_sets(history, filename=str(path))
                    fetch.assert_not_called()
                    generate.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_main_checks_predictions_before_history_collection(self):
        app = load_app()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_bytes(b'"broken')
            with patch.object(app, "PREDICTION_FILE", str(path)), \
                    patch("sys.argv", ["lotto-app"]), patch.object(app, "collect_all_lotto") as collect:
                with self.assertRaises(CsvDataError):
                    app.main()
                collect.assert_not_called()
            self.assertEqual(path.read_bytes(), b'"broken')

    def test_new_batch_and_existing_batch_write_only_prediction_csv(self):
        app = load_app()
        history = dated_history(3)
        generated = select_portfolio(np.full(45, 6 / 45), seed=44)
        generated.attrs["weekly_review"] = {"selected_strength": 0.0, "effective_methods": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history_path = root / "과거로또 당첨번호.csv"
            prediction_path = root / "당첨예상번호.csv"
            history.to_csv(history_path, index=False, encoding="utf-8-sig")
            history_bytes = history_path.read_bytes()
            prior = generated.copy()
            prior["예측회차"] = 3
            prior["종합점수"] = prior.pop("조합점수")
            for column in app.PREDICTION_FILE_COLUMNS:
                if column not in prior:
                    prior[column] = pd.NA
            prior[app.PREDICTION_FILE_COLUMNS].to_csv(prediction_path, index=False, encoding="utf-8-sig")
            with patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "manual"}), \
                    patch.object(app, "latest_completed_draw", return_value=3), \
                    patch.object(app, "get_lotto_draw", return_value=history.iloc[-1].to_dict()), \
                    patch.object(app, "generate_prediction_sets", return_value=generated) as generate, \
                    redirect_stdout(io.StringIO()):
                result = app.save_prediction_sets(history, filename=str(prediction_path))
                generate.assert_called_once()
                self.assertEqual(result.attrs["final_review"]["status"], "manual_review_pending")
                request = result.attrs["final_review"]["request"]
                proposal = {"input_sha256": request["input_sha256"], "target_draw": 4,
                            "verdict": "pass", "reason": "Checked supplied batch", "checked_set_ids": list(range(1, 11))}
                generate.reset_mock()
                second = app.save_prediction_sets(history, filename=str(prediction_path), manual_review=proposal)
                generate.assert_not_called()
                self.assertTrue(second.attrs["final_review"]["finalized"])
            saved = pd.read_csv(prediction_path)
            self.assertEqual(len(saved), 20)
            self.assertEqual(list(saved.columns), app.PREDICTION_FILE_COLUMNS)
            pd.testing.assert_frame_equal(saved[saved["예측회차"] == 3][app.NUMBER_COLUMNS].reset_index(drop=True),
                                          prior[app.NUMBER_COLUMNS].reset_index(drop=True))
            self.assertTrue(saved[saved["예측회차"] == 3]["본번호일치수"].notna().all())
            self.assertEqual(history_bytes, history_path.read_bytes())
            self.assertEqual({p.name for p in root.iterdir()}, {history_path.name, prediction_path.name})

    def test_failed_official_check_leaves_no_outputs(self):
        app = load_app()
        history = dated_history(3)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, "latest_completed_draw", return_value=3), \
                patch.object(app, "get_lotto_draw", return_value=None), \
                patch.object(app, "generate_prediction_sets") as generate:
            with self.assertRaises(ValueError):
                app.save_prediction_sets(history, filename=str(Path(folder) / "당첨예상번호.csv"))
            generate.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_final_review_has_no_filesystem_io(self):
        history = dated_history(1)
        candidate = {"예측회차": 2, "세트": 1,
                     **{f"번호{i}": i for i in range(1, 7)}}
        with patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "manual"}), \
                patch("builtins.open", side_effect=AssertionError("no filesystem I/O")), \
                patch.object(Path, "open", side_effect=AssertionError("no pathlib I/O")):
            result = final_review(history, pd.DataFrame([candidate]), pd.DataFrame(),
                                  lambda _: history.iloc[0].to_dict(), expected_count=1)
        self.assertEqual(result["status"], "manual_review_pending")
        self.assertIn("request", result)

    def test_all_python_sources_compile_and_only_core_modules_write_csv(self):
        allowed_writers = {"로또 당첨번호 예측.py", "lotto.py", "lotto_engine.py"}
        paths = list(BASE.glob("*.py")) + list((BASE / "analysis_cache").glob("*.py"))
        for path in paths:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
            if path.name.startswith("test_"):
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Attribute):
                    self.assertNotIn(function.attr, {"write_text", "write_bytes", "dump"}, str(path))
                    if function.attr == "to_csv" and (node.args or any(k.arg == "path_or_buf" for k in node.keywords)):
                        self.assertIn(path.name, allowed_writers)


if __name__ == "__main__":
    unittest.main()
