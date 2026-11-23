"""Exact portfolio coverage, including dependence between overlapping tickets."""
from math import comb
import unittest

from lotto_coverage import coverage


class CoverageTests(unittest.TestCase):
    def test_one_ticket_matches_hypergeometric_probability(self):
        result = coverage([[1, 21, 22, 23, 24, 45]])
        self.assertEqual(result["total_outcomes"], 8_145_060)
        self.assertEqual(result["set_count"], 1)
        for k in range(3, 7):
            favorable = sum(comb(6, h) * comb(39, 6 - h) for h in range(k, 7))
            self.assertEqual(result["hit_counts"][str(k)], favorable)
            self.assertEqual(result["hit_probabilities"][str(k)], favorable / 8_145_060)
        self.assertEqual(result["random_probabilities"], result["hit_probabilities"])

    def test_two_disjoint_tickets_count_shared_prize_draws_once(self):
        result = coverage([[1, 2, 3, 4, 5, 6], [40, 41, 42, 43, 44, 45]])
        for k in range(3, 7):
            single = sum(comb(6, h) * comb(39, 6 - h) for h in range(k, 7))
            intersection = comb(6, 3) ** 2 if k == 3 else 0
            self.assertEqual(result["hit_counts"][str(k)], 2 * single - intersection)
            # Two uniform purchases without replacement. Given that the first misses,
            # one fewer losing ticket is available for the second choice.
            total = result["total_outcomes"]
            random_chance = single / total + ((total - single) / total) * (single / (total - 1))
            self.assertAlmostEqual(result["random_probabilities"][str(k)], random_chance)

    def test_distinct_tickets_with_five_shared_numbers(self):
        result = coverage([[45, 44, 23, 22, 21, 1], [45, 44, 23, 22, 21, 2]])
        # Pick c of the five shared numbers, u of the two unique numbers,
        # and the rest from the 38 numbers outside both tickets.
        for k in range(3, 7):
            favorable = 0
            for c in range(6):
                for u in range(3):
                    if c + u <= 6 and c + min(u, 1) >= k:
                        favorable += comb(5, c) * comb(2, u) * comb(38, 6 - c - u)
            self.assertEqual(result["hit_counts"][str(k)], favorable)

    def test_invalid_tickets(self):
        ticket = [1, 2, 3, 4, 5, 6]
        cases = [[], [ticket, ticket[::-1]], [ticket[:-1]], [[1, 2, 3, 4, 5, 5]],
                 [[0, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 46]], [[1, 2, 3, 4, 5, 5.5]],
                 [[True, 2, 3, 4, 5, 6]], ["123456"],
                 [[1, 2, 3, 4, 5, n] for n in range(6, 17)]]
        for batch in cases:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                coverage(batch)


if __name__ == "__main__":
    unittest.main()
