"""Behavioral checks for chronological calibration and six-main-number tickets."""
import unittest
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pandas as pd

import lotto_portfolio as portfolio


def make_history(count):
    rows = []
    for draw in range(1, count + 1):
        numbers = sorted(((draw * 7 + offset) % 45) + 1 for offset in range(6))
        bonus = next(number for number in range(1, 46) if number not in numbers)
        rows.append({"회차": draw, "보너스": bonus,
                     **dict(zip(portfolio.NUMBER_COLUMNS, numbers))})
    return pd.DataFrame(rows)


class ProbabilityTests(unittest.TestCase):
    def assert_valid_probabilities(self, probabilities):
        self.assertEqual(probabilities.shape, (45,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0.04).all())
        self.assertTrue((probabilities <= 0.30).all())
        self.assertAlmostEqual(float(probabilities.sum()), 6.0, places=10)

    def test_fit_returns_valid_probabilities_even_for_extreme_current_features(self):
        rng = np.random.default_rng(712)
        features = rng.normal(size=(8, 45, 3))
        outcomes = np.zeros((8, 45))
        outcomes[:, :6] = 1
        current = rng.normal(size=(45, 3)) * 1000
        probabilities = portfolio.fit_inclusion_probabilities(features, outcomes, current)
        self.assert_valid_probabilities(probabilities)

    def test_no_resolved_labels_or_constant_features_produce_uniform_prior(self):
        cases = [(np.empty((0, 45, 2)), np.empty((0, 45))),
                 (np.ones((3, 45, 2)), np.tile(np.r_[np.ones(6), np.zeros(39)], (3, 1)))]
        for features, outcomes in cases:
            with self.subTest(shape=features.shape):
                result = portfolio.fit_inclusion_probabilities(features, outcomes, np.ones((45, 2)))
                np.testing.assert_allclose(result, np.full(45, 6 / 45), atol=1e-12)

    def test_nonfinite_features_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                portfolio.fit_inclusion_probabilities(
                    np.full((2, 45, 1), value), np.zeros((2, 45)), np.zeros((45, 1)))

    def test_calibration_features_precede_labels_and_future_draw_is_excluded(self):
        history_with_future = make_history(154)
        training = history_with_future.iloc[:153].copy()
        observed_prefixes = []

        def scores(prefix):
            observed_prefixes.append(tuple(prefix["회차"]))
            latest = set(prefix.iloc[-1][portfolio.NUMBER_COLUMNS].astype(int))
            return {
                "main": {number: float(number in latest) for number in range(1, 46)},
                # If this unrelated signal enters the fit, finite validation fails.
                "보너스보조신호": {number: np.nan for number in range(1, 46)},
            }

        original_fit = portfolio.fit_inclusion_probabilities
        with patch.object(portfolio, "fit_inclusion_probabilities", wraps=original_fit) as fit:
            result = portfolio.calibrated_probabilities(training, scores)

        self.assert_valid_probabilities(result)
        self.assertEqual({len(prefix) for prefix in observed_prefixes}, {150, 151, 152, 153})
        for prefix in observed_prefixes:
            self.assertEqual(prefix, tuple(range(1, len(prefix) + 1)))
            self.assertLessEqual(max(prefix), 153)

        features, outcomes, current = fit.call_args.args
        self.assertEqual(features.shape, (3, 45, 1))
        for offset, label_index in enumerate(range(150, 153)):
            previous_numbers = training.iloc[label_index - 1][portfolio.NUMBER_COLUMNS].astype(int)
            label_numbers = training.iloc[label_index][portfolio.NUMBER_COLUMNS].astype(int)
            expected_feature = np.isin(np.arange(1, 46), previous_numbers).astype(float)
            expected_label = np.isin(np.arange(1, 46), label_numbers).astype(float)
            np.testing.assert_array_equal(features[offset, :, 0], expected_feature)
            np.testing.assert_array_equal(outcomes[offset], expected_label)
        np.testing.assert_array_equal(np.asarray(current)[:, 0], outcomes[-1])
        future_numbers = history_with_future.iloc[153][portfolio.NUMBER_COLUMNS].astype(int)
        future_label = np.isin(np.arange(1, 46), future_numbers).astype(float)
        self.assertFalse(any(np.array_equal(label, future_label) for label in outcomes))


class TicketTests(unittest.TestCase):
    def test_ranked_sets_match_exhaustive_model_optimum_and_repeat_core(self):
        probabilities = np.full(45, 0.00001)
        probabilities[:9] = np.linspace(0.29, 0.12, 9)
        result = portfolio.select_ranked_portfolio(probabilities, seed=1)
        other = portfolio.select_ranked_portfolio(probabilities, seed=999)
        pd.testing.assert_frame_equal(result, other)
        odds = probabilities / (1 - probabilities)
        exhaustive = sorted(combinations(range(9), 6),
                            key=lambda row: -np.log(odds[list(row)]).sum())[:10]
        actual = result[portfolio.NUMBER_COLUMNS].to_numpy() - 1
        self.assertEqual([tuple(row) for row in actual], exhaustive)
        self.assertGreater(int((actual == 0).sum()), 1)

    def test_sampling_is_reproducible_and_six_numbers_are_unique(self):
        weights = np.linspace(0.1, 2.0, 45)
        first = portfolio.sample_tickets(np.random.default_rng(77), weights, 30)
        second = portfolio.sample_tickets(np.random.default_rng(77), weights, 30)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (30, 6))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue(((first >= 0) & (first < 45)).all())
        self.assertTrue((np.diff(first, axis=1) > 0).all())

    def test_selected_portfolio_is_reproducible_valid_and_has_no_duplicate_sets(self):
        probabilities = np.full(45, 6 / 45)
        options = dict(set_count=10, seed=940, candidate_count=50, scenario_count=80)
        first = portfolio.select_portfolio(probabilities, **options)
        second = portfolio.select_portfolio(probabilities, **options)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["세트"].tolist(), list(range(1, 11)))
        numbers = first[portfolio.NUMBER_COLUMNS].to_numpy()
        self.assertTrue(np.issubdtype(numbers.dtype, np.integer))
        self.assertTrue(((numbers >= 1) & (numbers <= 45)).all())
        self.assertTrue((np.diff(numbers, axis=1) > 0).all())
        self.assertEqual(len({tuple(row) for row in numbers}), 10)

    def test_balancing_prevents_uniform_and_extreme_score_concentration(self):
        for probabilities in [np.full(45, 6 / 45), np.linspace(0.04, 0.30, 45)]:
            for seed in range(20):
                result = portfolio.select_portfolio(probabilities, seed=seed)
                numbers = result[portfolio.NUMBER_COLUMNS].to_numpy()
                diagnostics = portfolio.portfolio_diagnostics(numbers)
                self.assertEqual(diagnostics["unique_numbers"], 45)
                self.assertEqual(diagnostics["max_number_exposure"], 2)
                self.assertLessEqual(diagnostics["max_pairwise_overlap"], 2)

    def test_invalid_probabilities_and_ticket_count_are_rejected(self):
        for probabilities in [np.ones(44) / 7, np.full(45, np.nan), np.zeros(45), np.ones(45)]:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                portfolio.select_portfolio(probabilities, candidate_count=20, scenario_count=30)
        with self.assertRaises(ValueError):
            portfolio.select_portfolio(np.full(45, 6 / 45), set_count=21, candidate_count=20)

    def test_hits_count_only_actual_main_numbers(self):
        actual_main = [1, 2, 3, 4, 5, 6]
        bonus = 7
        tickets = [[1, 2, 3, 4, 5, bonus], [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        self.assertEqual(portfolio.ticket_hits(tickets, actual_main), [5, 6, 0])

    def test_target_with_observed_or_future_gap_draw_is_rejected_before_scoring(self):
        history = make_history(154)
        with patch.object(portfolio, "calibrated_probabilities") as calibrate:
            for target in [153, 154, 156]:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    portfolio.generate_prediction_sets(history, lambda _: {}, target_draw=target)
            calibrate.assert_not_called()

    def test_duplicate_training_rounds_are_rejected(self):
        history = make_history(3)
        with self.assertRaises(ValueError):
            portfolio.generate_prediction_sets(pd.concat([history, history.iloc[-1:]]), lambda _: {})


if __name__ == "__main__":
    unittest.main()
