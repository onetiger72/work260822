import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from lotto_validation import NUMBERS, RESULT_FIELDS, strict_int, validate_history, final_review
from lotto_weekly import PredictionWindowError


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2002, 12, 13, tzinfo=timezone.utc)
        self.draw = {"회차": 1, "날짜": "20021207", **dict(zip(NUMBERS, range(1, 7))), "보너스": 7}
        self.history = pd.DataFrame([self.draw])
        self.candidates = pd.DataFrame([{"예측회차": 2, "세트": 1, **dict(zip(NUMBERS, range(2, 8)))}])

    def run_review(self, reply=None, source=True):
        def create(**kwargs):
            request = json.loads(kwargs["input"])
            proposal = {"input_sha256": request["input_sha256"], "target_draw": 2,
                        "verdict": "pass", "reason": "All supplied sets checked", "checked_set_ids": [1]}
            if reply:
                proposal.update(reply)
            return SimpleNamespace(status="completed", output_text=json.dumps(proposal))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "api"}):
            result = final_review(self.history, self.candidates, pd.DataFrame(),
                                  lambda _: self.draw if source else None,
                                  client=SimpleNamespace(responses=SimpleNamespace(create=create)), expected_count=1,
                                  now=self.now)
            self.assertEqual(list(Path(folder).iterdir()), [])
            return result

    def test_fraction_bool_and_nonfinite_rejected(self):
        for value in [1.5, True, float("nan"), float("inf"), "2.4"]:
            with self.assertRaises(ValueError):
                strict_int(value)

    def test_history_conflicts_and_dates(self):
        with self.assertRaises(ValueError):
            validate_history(pd.concat([self.history, self.history]))
        bad = self.history.copy()
        bad.loc[0, "날짜"] = "20021208"
        with self.assertRaises(ValueError):
            validate_history(bad)

    def test_valid_api_review(self):
        self.assertTrue(self.run_review()["finalized"])

    def test_unavailable_source_cannot_pass(self):
        self.assertFalse(self.run_review(source=False)["finalized"])

    def test_wrong_hash_and_missing_sets(self):
        for reply in [{"input_sha256": "stale"}, {"checked_set_ids": []}, {"target_draw": 3}]:
            self.assertEqual(self.run_review(reply)["status"], "validation_error")

    def test_source_mismatch(self):
        self.draw = copy.deepcopy(self.draw)
        self.draw["보너스"] = 8
        self.assertEqual(self.run_review()["status"], "source_mismatch")

    def test_bad_candidate(self):
        self.candidates.loc[0, "번호1"] = 3
        self.assertEqual(self.run_review()["status"], "validation_error")

    def test_api_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "api"}):
            result = final_review(self.history, self.candidates, pd.DataFrame(), lambda _: self.draw,
                                  client=SimpleNamespace(), expected_count=1, now=self.now)
            self.assertFalse(result["finalized"])
            self.assertEqual(result["status"], "validation_error")

    def test_manual_review_round_trip_without_files_and_stale_response_rejected(self):
        with patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "manual"}):
            pending = final_review(self.history, self.candidates, pd.DataFrame(),
                                   lambda _: self.draw, expected_count=1, now=self.now)
            self.assertEqual(pending["status"], "manual_review_pending")
            request = pending["request"]
            proposal = {"input_sha256": request["input_sha256"], "target_draw": 2,
                        "verdict": "pass", "reason": "Checked all sets", "checked_set_ids": [1]}
            passed = final_review(self.history, self.candidates, pd.DataFrame(),
                                  lambda _: self.draw, expected_count=1, manual_review=proposal, now=self.now)
            self.assertTrue(passed["finalized"])
            proposal["input_sha256"] = "stale"
            stale = final_review(self.history, self.candidates, pd.DataFrame(),
                                 lambda _: self.draw, expected_count=1, manual_review=proposal, now=self.now)
            self.assertFalse(stale["finalized"])
            self.assertEqual(stale["status"], "manual_review_pending_stale_proposal")

    def test_review_evidence_is_bound_and_session_mode_is_local(self):
        args = (self.history, self.candidates, pd.DataFrame(), lambda _: self.draw)
        evidence = {"seed": 17}
        with patch.dict(os.environ, {"LOTTO_FINAL_REVIEW_MODE": "api"}):
            first = final_review(*args, expected_count=1, selection_evidence=evidence,
                                 now=self.now, mode="manual")
            self.assertEqual(first["status"], "manual_review_pending")
            proposal = {"input_sha256": first["input_sha256"], "target_draw": 2, "verdict": "pass",
                        "reason": "Reviewed", "checked_set_ids": [1]}
            evidence["seed"] = 18
            self.assertEqual(first["request"]["payload"]["selection_evidence"]["seed"], 17)
            changed = final_review(*args, expected_count=1, selection_evidence=evidence,
                                   now=self.now, mode="manual", manual_review=proposal)
        self.assertFalse(changed["finalized"])
        self.assertEqual(changed["status"], "manual_review_pending_stale_proposal")

    def test_expired_target_is_rejected_before_official_fetch(self):
        def unexpected_fetch(_):
            self.fail("Expired review attempted official fetch")

        result = final_review(self.history, self.candidates, pd.DataFrame(), unexpected_fetch,
                              expected_count=1, now=datetime(2002, 12, 14, 11, tzinfo=timezone.utc))
        self.assertFalse(result["finalized"])
        self.assertEqual(result["error_type"], "PredictionWindowError")

    def test_current_candidate_has_no_results_before_draw(self):
        for field in RESULT_FIELDS + ["평가상태"]:
            with self.subTest(field=field):
                candidates = self.candidates.copy()
                candidates[field] = "평가완료" if field == "평가상태" else 0
                result = final_review(self.history, candidates, pd.DataFrame(), lambda _: self.draw,
                                      expected_count=1, now=self.now, mode="manual")
                self.assertEqual(result["status"], "validation_error")
                self.assertNotIn("request", result)
        self.candidates["평가상태"] = "대기"
        self.candidates["본번호일치수"] = pd.NA
        result = final_review(self.history, self.candidates, pd.DataFrame(), lambda _: self.draw,
                              expected_count=1, now=self.now, mode="manual")
        self.assertEqual(result["status"], "manual_review_pending")

    def test_review_fingerprint_includes_both_source_snapshots(self):
        predictions = pd.DataFrame()
        self.history.attrs["source_sha256"] = "a" * 64
        predictions.attrs["source_sha256"] = "b" * 64

        def preview():
            return final_review(self.history, self.candidates, predictions, lambda _: self.draw,
                                expected_count=1, now=self.now, mode="manual")

        original = preview()
        self.assertEqual(original["payload"]["input_history_sha256"], "a" * 64)
        self.assertEqual(original["payload"]["input_predictions_sha256"], "b" * 64)
        self.history.attrs["source_sha256"] = "c" * 64
        updated = preview()
        self.assertNotEqual(original["input_sha256"], updated["input_sha256"])
        predictions.attrs["source_sha256"] = "d" * 64
        self.assertNotEqual(updated["input_sha256"], preview()["input_sha256"])

    def test_passing_response_after_deadline_cannot_finalize(self):
        with patch("lotto_validation.verify_prediction_window",
                   side_effect=[2, PredictionWindowError("판매 마감")]):
            result = self.run_review()
        self.assertFalse(result["finalized"])
        self.assertEqual(result["error_type"], "PredictionWindowError")


if __name__ == "__main__":
    unittest.main()
