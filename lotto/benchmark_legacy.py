"""Read-only chronological benchmark of the captured legacy ticket pipeline."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import pandas as pd


BASE = Path(__file__).resolve().parent
CACHE = BASE / "analysis_cache"
SOURCE = CACHE / "legacy_app.py"
HISTORY = BASE / "과거로또 당첨번호.csv"


def main():
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("legacy_app_benchmark", SOURCE)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.CSV_FILE = str(HISTORY)
    history = app.load_history_csv().sort_values("회차").reset_index(drop=True)
    # Preserve the original comparison range as newer draws arrive.
    history = history[history["회차"] <= 1240].reset_index(drop=True)
    if int(history["회차"].max()) != 1240:
        raise ValueError("legacy comparison requires history through draw 1240")
    assert app.PREDICTION_CANDIDATE_ATTEMPTS == 15000
    original_scores = app.calculate_analysis_scores
    score_cache = {}
    start_length = max(150, len(history) - 360)
    for train_length in range(start_length, len(history) + 1):
        score_cache[train_length] = original_scores(history.iloc[:train_length])
        if (train_length - start_length) % 20 == 0:
            print(f"score cache {train_length}/{len(history)}, seconds={time.perf_counter()-started:.1f}", flush=True)

    def cached_scores(frame):
        if len(frame) in score_cache:
            # Every caller uses an unmodified chronological prefix of history.
            assert int(frame["회차"].max()) == len(frame)
            return score_cache[len(frame)]
        return original_scores(frame)

    app.calculate_analysis_scores = cached_scores
    records = []
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    history_hash = hashlib.sha256(HISTORY.read_bytes()).hexdigest()

    def summarize():
        rows = len(records)
        payload = {
            "label": "legacy full 10-ticket chronological evaluation",
            "source_sha256": source_hash,
            "history_sha256": history_hash,
            "history_path": str(HISTORY),
            "train_rule": "only draws strictly before target",
            "live_feedback": "disabled; unavailable historical timestamps prevent trustworthy reconstruction",
            "llm_audit": "disabled",
            "candidate_attempts": app.PREDICTION_CANDIDATE_ATTEMPTS,
            "inner_backtest_draws": app.BACKTEST_DRAWS,
            "seed_rule": "target_draw * 10007 + training_length",
            "target_first": 1181,
            "target_last": 1240,
            "completed_draws": rows,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "mean_hits_per_ticket": sum(sum(x["hits"]) for x in records) / (rows * 10) if rows else None,
            "mean_best_hits": sum(x["best_hits"] for x in records) / rows if rows else None,
            "draws_best_at_least": {str(k): sum(x["best_hits"] >= k for x in records) for k in (3, 4, 5, 6)},
            "draws_best_exactly": {str(k): sum(x["best_hits"] == k for x in records) for k in range(7)},
            "tickets_hits_exactly": {str(k): sum(x["hits"].count(k) for x in records) for k in range(7)},
            "rounds": records,
        }
        return payload

    for target in range(1181, 1241):
        round_started = time.perf_counter()
        train = history.loc[history["회차"] < target].copy()
        performance = app.backtest_analysis_methods(train)
        predicted = app.generate_prediction_sets(
            train, set_count=10, target_draw=target,
            method_performance_df=performance,
            prediction_feedback=None, llm_audit=None,
        )
        tickets = [sorted(map(int, row)) for row in predicted[app.NUMBER_COLUMNS].to_numpy()]
        assert len(tickets) == 10 and len({tuple(t) for t in tickets}) == 10
        actual = sorted(map(int, history.loc[history["회차"] == target, app.NUMBER_COLUMNS].iloc[0]))
        hits = [len(set(ticket) & set(actual)) for ticket in tickets]
        records.append({
            "target_draw": target, "training_last_draw": target - 1,
            "tickets": tickets, "actual": actual, "hits": hits,
            "mean_hits": sum(hits) / len(hits), "best_hits": max(hits),
            "runtime_seconds": round(time.perf_counter() - round_started, 3),
        })
        print(f"legacy target={target} best={max(hits)} mean={sum(hits)/10:.2f} total_seconds={time.perf_counter()-started:.1f}", flush=True)
    report = summarize()
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
