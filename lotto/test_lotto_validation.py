import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from lotto_validation import NUMBERS, strict_int, validate_history, final_review


class ValidationTests(unittest.TestCase):
    def setUp(self):
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
            return final_review(self.history, self.candidates, pd.DataFrame(),
                                lambda _: self.draw if source else None, folder,
                                SimpleNamespace(responses=SimpleNamespace(create=create)), expected_count=1)

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
                                  folder, SimpleNamespace(), expected_count=1)
            self.assertFalse(result["finalized"])
            self.assertEqual(result["status"], "validation_error")


if __name__ == "__main__":
    unittest.main()
