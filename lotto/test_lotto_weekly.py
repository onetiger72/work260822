import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

import lotto_portfolio as portfolio
from lotto_weekly import latest_completed_draw, verify_latest_result
from test_lotto_portfolio import make_history


class WeeklyTests(unittest.TestCase):
    def test_publication_boundary(self):
        korea = timezone(timedelta(hours=9))
        self.assertEqual(latest_completed_draw(datetime(2026, 9, 19, 20, 59, tzinfo=korea)), 1241)
        self.assertEqual(latest_completed_draw(datetime(2026, 9, 19, 21, 0, tzinfo=korea)), 1242)

    def test_source_unavailable_mismatch_and_stale_history_block_generation(self):
        history = make_history(3)
        history["날짜"] = ["20021207", "20021214", "20021221"]
        latest = history.iloc[-1].to_dict()
        self.assertEqual(verify_latest_result(history, lambda _: latest, 3)["회차"], 3)
        for fetch, expected in [(lambda _: None, 3),
                                (lambda _: history.iloc[0].to_dict(), 3),
                                (lambda _: latest, 4)]:
            with self.assertRaises(ValueError):
                verify_latest_result(history, fetch, expected)

    def test_review_uses_only_prefixes_and_rejects_harmful_signal(self):
        history = make_history(180)
        prefixes = []

        def harmful(prefix, scores, cache, methods=None):
            prefixes.append(len(prefix))
            # An intentionally wrong estimate for this synthetic sequence.
            next_numbers = make_history(len(prefix) + 1).iloc[-1][portfolio.NUMBER_COLUMNS].to_numpy(dtype=int)
            q = np.full(45, (6 - 6 * 0.04) / 39)
            q[next_numbers - 1] = 0.04
            return q

        with patch.object(portfolio, "calibrated_probabilities", side_effect=harmful):
            q, report = portfolio.review_probabilities(history, lambda _: {})
        self.assertEqual(prefixes, list(range(156, 181)))
        self.assertEqual(report["review_rounds"], list(range(157, 181)))
        self.assertEqual(report["selected_strength"], 0)
        np.testing.assert_allclose(q, 6 / 45)

    def test_reliable_signal_is_retained(self):
        history = make_history(180)

        def useful(prefix, scores, cache, methods=None):
            numbers = make_history(len(prefix) + 1).iloc[-1][portfolio.NUMBER_COLUMNS].to_numpy(dtype=int)
            q = np.full(45, (6 - 6 * 0.3) / 39)
            q[numbers - 1] = 0.3
            return q

        with patch.object(portfolio, "calibrated_probabilities", side_effect=useful):
            _, report = portfolio.review_probabilities(history, lambda _: {})
        self.assertEqual(report["selected_strength"], 1)

    def test_method_screening_excludes_noise_and_keeps_predictive_feature(self):
        history = make_history(210)

        def scores(prefix):
            # Known deterministic synthetic sequence, not production knowledge.
            future = set(make_history(len(prefix) + 1).iloc[-1][portfolio.NUMBER_COLUMNS])
            return {"signal": {n: float(n in future) for n in range(1, 46)},
                    "constant": {n: 0.5 for n in range(1, 46)}}

        _, report = portfolio.review_probabilities(history, scores)
        screen = report["method_screening"]
        self.assertEqual(screen["selected_methods"], ["signal"])
        self.assertLess(max(screen["rounds"]), min(report["review_rounds"]))
        self.assertEqual(report["effective_methods"], ["signal"])

    def test_no_eligible_method_means_uniform_probabilities(self):
        history = make_history(210)
        q, report = portfolio.review_probabilities(history, lambda _: {
            "constant": {n: 0.5 for n in range(1, 46)}})
        self.assertEqual(report["effective_methods"], [])
        np.testing.assert_allclose(q, 6 / 45)


if __name__ == "__main__":
    unittest.main()
