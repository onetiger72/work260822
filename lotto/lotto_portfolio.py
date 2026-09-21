"""Prequential main-number calibration and best-of-ten ticket selection.

All historical feature rows are computed before their labeled draw. Coefficients
are regularized toward the uniform 6/45 baseline. No number-pattern exclusions.
Default selection ranks exact six-number sets without random sampling.
The older simulation selector is retained for comparative evaluation.
"""
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMBER_COLUMNS = [f"번호{i}" for i in range(1, 7)]
VERSION = "weekly_review_v4"
CALIBRATION_DRAWS = 120
RIDGE = 120.0
# Convex gain gives progressively greater value to approaching six main hits.
HIT_UTILITY = np.array([0, 1, 3, 9, 27, 81, 243], dtype=float)


def history_fingerprint(df):
    return hashlib.sha256(df[["회차"] + NUMBER_COLUMNS + ["보너스"]].to_csv(index=False).encode()).hexdigest()


def fit_inclusion_probabilities(features, outcomes, current):
    """Fit past labels; halve linear deviations from 6/45 before projection."""
    base = 6 / 45
    x = np.asarray(features, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    current = np.asarray(current, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(current).all() or not np.isfinite(y).all():
        raise ValueError("non-finite analysis feature")
    if len(x) == 0:
        return np.full(45, base)
    x = x - x.mean(axis=1, keepdims=True)
    current = current - current.mean(axis=0, keepdims=True)
    flat = x.reshape(-1, x.shape[-1])
    coefficients = np.linalg.solve(flat.T @ flat + RIDGE * np.eye(flat.shape[1]),
                                   flat.T @ (y.reshape(-1) - base))
    q = base + 0.5 * (current @ coefficients)
    # Projection to a bounded simplex keeps marginal total exactly six.
    low, high = 0.04 - q.max(), 0.30 - q.min()
    for _ in range(60):
        shift = (low + high) / 2
        if np.clip(q + shift, 0.04, 0.30).sum() < 6:
            low = shift
        else:
            high = shift
    return np.clip(q + (low + high) / 2, 0.04, 0.30)


def calibrated_probabilities(df, calculate_scores, cache=None):
    """For target n+1: train feature at length t against row t, t < n."""
    cache = {} if cache is None else cache
    n = len(df)
    if n < 151:
        return np.full(45, 6 / 45)

    def scores_at(length):
        if length not in cache:
            cache[length] = calculate_scores(df.iloc[:length])
        return cache[length]

    current_scores = scores_at(n)
    methods = sorted(m for m in current_scores if m != "보너스보조신호")

    def matrix(length):
        scores = scores_at(length)
        return [[scores[m][number] for m in methods] for number in range(1, 46)]

    start = max(150, n - CALIBRATION_DRAWS)
    features = np.array([matrix(t) for t in range(start, n)])
    outcomes = np.zeros((n - start, 45))
    actual = df.iloc[start:n][NUMBER_COLUMNS].to_numpy(dtype=int) - 1
    for index, numbers in enumerate(actual):
        outcomes[index, numbers] = 1
    return fit_inclusion_probabilities(features, outcomes, matrix(n))


def sample_tickets(rng, weights, count):
    # Plackett-Luce sampling via exponential races; always six unique integers.
    races = -np.log(np.maximum(rng.random((count, 45)), 1e-15)) / weights
    return np.sort(np.argpartition(races, 5, axis=1)[:, :6], axis=1)


def review_probabilities(df, calculate_scores, cache=None, review_draws=24):
    """Choose signal strength using strictly past, rolling Brier losses.

    A one-standard-error rule favors weaker signals when differences are small.
    These are tuning results, not an independent estimate of future accuracy.
    """
    cache = {} if cache is None else cache
    strengths = np.array([0.0, 0.25, 0.5, 1.0])
    losses, rounds = [], []
    base = np.full(45, 6 / 45)
    for t in range(max(151, len(df) - review_draws), len(df)):
        q = calibrated_probabilities(df.iloc[:t], calculate_scores, cache)
        actual = np.zeros(45)
        actual[df.iloc[t][NUMBER_COLUMNS].to_numpy(dtype=int) - 1] = 1
        candidates = base + strengths[:, None] * (q - base)
        losses.append(np.mean((candidates - actual) ** 2, axis=1))
        rounds.append(int(df.iloc[t]["회차"]))
    chosen = 0
    mean_losses = None
    if len(losses) >= 12:
        losses = np.asarray(losses)
        mean_losses = losses.mean(axis=0)
        best = int(np.argmin(mean_losses))
        # Paired errors account for the same actual draw in each candidate.
        differences = losses - losses[:, best, None]
        uncertainty = differences.std(axis=0, ddof=1) / np.sqrt(len(losses))
        chosen = int(np.flatnonzero(mean_losses - mean_losses[best] <= uncertainty + 1e-12)[0])
    q = calibrated_probabilities(df, calculate_scores, cache)
    report = {
        "version": VERSION, "history_sha256": history_fingerprint(df),
        "target_draw": int(df.iloc[-1]["회차"]) + 1,
        "review_rounds": rounds, "selected_strength": float(strengths[chosen]),
        "candidate_strengths": strengths.tolist(),
        "mean_brier_losses": None if mean_losses is None else mean_losses.tolist(),
        "last_draw_brier_losses": None if not len(losses) else np.asarray(losses)[-1].tolist(),
        "reason": "paired_one_standard_error" if len(losses) >= 12 else "insufficient_history",
        "limits": "Retrospective tuning, not held-out proof of improvement or winning probabilities.",
    }
    return base + strengths[chosen] * (q - base), report


def select_portfolio(probabilities, set_count=10, seed=0, candidate_count=1600, scenario_count=2400):
    """Exact top distinct six-number sets under a conditional Bernoulli model.

    Conditioning independent inclusions on exactly six numbers makes set mass
    proportional to the product of inclusion odds. This is a model assumption,
    not evidence of unequal real lottery probabilities. Seed and pool arguments
    remain accepted for caller compatibility; no random search is performed.
    The displayed score remains the sum of marginals for CSV compatibility.
    """
    q = np.asarray(probabilities, dtype=float)
    if (q.shape != (45,) or not np.isfinite(q).all()
            or np.any(q <= 0) or np.any(q >= 1)):
        raise ValueError("invalid probabilities")
    if not isinstance(set_count, (int, np.integer)) or not 1 <= set_count <= candidate_count:
        raise ValueError("invalid set_count")
    # Uniform fallback must not prefer small numbers merely by array position.
    order = np.lexsort((np.random.default_rng(seed).random(45), -q))
    log_odds = np.log(q[order]) - np.log1p(-q[order])
    start = tuple(range(6))
    heap = [(-float(log_odds[list(start)].sum()), start)]
    seen = {start}
    rows = []
    while heap and len(rows) < set_count:
        _, ranks = heapq.heappop(heap)
        numbers = np.sort(order[list(ranks)])
        row = {"세트": len(rows) + 1,
               "조합점수": round(float(q[numbers].sum()), 6)}
        row.update(dict(zip(NUMBER_COLUMNS, (numbers + 1).tolist())))
        rows.append(row)
        for position in range(6):
            limit = ranks[position + 1] if position < 5 else 45
            if ranks[position] + 1 >= limit:
                continue
            neighbor = list(ranks)
            neighbor[position] += 1
            neighbor = tuple(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                heapq.heappush(heap, (-float(log_odds[list(neighbor)].sum()), neighbor))
    return pd.DataFrame(rows)[["세트"] + NUMBER_COLUMNS + ["조합점수"]]


def select_simulated_portfolio(probabilities, set_count=10, seed=0, candidate_count=1600, scenario_count=2400):
    if not 1 <= set_count <= candidate_count:
        raise ValueError("invalid set_count")
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (45,) or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0) or np.any(probabilities >= 1)):
        raise ValueError("invalid probabilities")
    rng = np.random.default_rng(seed)
    weights = probabilities / (1 - probabilities)
    # Candidate pool includes diverse uniform draws and high-score draws.
    candidates = np.unique(np.vstack([
        sample_tickets(rng, weights, candidate_count),
        sample_tickets(rng, np.ones(45), candidate_count // 2),
        np.sort(np.argsort(-probabilities)[:6])[None, :]
    ]), axis=0)
    if len(candidates) < set_count:
        raise ValueError("insufficient distinct candidates")
    # Weighted sampling is a scenario heuristic: its inclusion marginals are
    # not exactly the input estimates, and utility is not a jackpot probability.
    scenarios = sample_tickets(rng, weights, scenario_count)
    candidate_matrix = np.zeros((len(candidates), 45), dtype=np.int16)
    scenario_matrix = np.zeros((len(scenarios), 45), dtype=np.int16)
    candidate_matrix[np.arange(len(candidates))[:, None], candidates] = 1
    scenario_matrix[np.arange(len(scenarios))[:, None], scenarios] = 1
    utilities = HIT_UTILITY[candidate_matrix @ scenario_matrix.T]
    covered = np.zeros(scenario_count)
    chosen = []
    # Optimize marginal expected best-ticket gain, not total number coverage.
    for _ in range(set_count):
        gains = np.maximum(utilities - covered, 0).mean(axis=1)
        gains[chosen] = -np.inf
        index = int(np.argmax(gains))
        chosen.append(index)
        covered = np.maximum(covered, utilities[index])
    rows = []
    for set_id, index in enumerate(chosen, 1):
        numbers = candidates[index]
        row = {"세트": set_id, "조합점수": round(float(probabilities[numbers].sum()), 6)}
        row.update(dict(zip(NUMBER_COLUMNS, (numbers + 1).tolist())))
        rows.append(row)
    return pd.DataFrame(rows)[["세트"] + NUMBER_COLUMNS + ["조합점수"]]


def generate_prediction_sets(df, calculate_scores, set_count=10, target_draw=None,
                             random_seed=None, cache=None):
    df = df.sort_values("회차").reset_index(drop=True)
    if df["회차"].duplicated().any():
        raise ValueError("duplicate training rounds")
    target = int(df["회차"].max()) + 1
    if target_draw is not None and target_draw != target:
        raise ValueError("target must follow the last training draw")
    probabilities, report = review_probabilities(df, calculate_scores, cache)
    seed = int(random_seed) if random_seed is not None else target * 10007 + len(df)
    result = select_portfolio(probabilities, set_count=set_count, seed=seed)
    result.attrs["weekly_review"] = report
    return result


def ticket_hits(tickets, actual):
    actual = set(int(n) for n in actual)
    return [len(actual & set(int(n) for n in row)) for row in tickets]


def summarize(records):
    best = [max(row["hits"]) for row in records]
    return {"draws": len(records), "mean_ticket_hits": float(np.mean([row["hits"] for row in records])),
            "mean_best_hits": float(np.mean(best)),
            "draws_at_least": {str(n): sum(h >= n for h in best) for n in range(3, 7)}}


def load_feature_cache(path):
    # The caller must bind this cache to the same immutable historical prefix.
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(length): {m: {int(n): v for n, v in scores.items()} for m, scores in methods.items()}
            for length, methods in source.items()}
