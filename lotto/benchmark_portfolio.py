"""Fixed-parameter retrospective, chronological end-to-end comparison."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lotto_portfolio import (NUMBER_COLUMNS, VERSION, generate_prediction_sets,
                             history_fingerprint, summarize, ticket_hits)
from lotto_validation import validate_history


def main():
    base = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("lotto_app", base / "로또 당첨번호 예측.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    history = pd.read_csv(app.CSV_FILE, encoding="utf-8-sig")
    validate_history(history)
    history = history.sort_values("회차").reset_index(drop=True)
    cache = {}
    records = []
    # Many random repetitions make the same-budget baseline less seed-sensitive.
    uniform_repetitions = [[] for _ in range(100)]
    for n in range(len(history) - 60, len(history)):
        target = int(history.iloc[n]["회차"])
        generated = generate_prediction_sets(history.iloc[:n], app.calculate_analysis_scores, cache=cache)
        actual = history.iloc[n][NUMBER_COLUMNS].tolist()
        records.append({"target_draw": target, "tickets": generated[NUMBER_COLUMNS].values.tolist(),
                        "hits": ticket_hits(generated[NUMBER_COLUMNS].values.tolist(), actual)})
        for repetition in range(100):
            rng = np.random.default_rng(7321 + 1000 * target + repetition)
            tickets = set()
            while len(tickets) < 10:
                tickets.add(tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False))))
            uniform_repetitions[repetition].append({"hits": ticket_hits(tickets, actual)})
        if len(records) % 10 == 0:
            print(f"new portfolio: {len(records)}/60", flush=True)
    summaries = [summarize(rows) for rows in uniform_repetitions]
    report = {"version": VERSION, "history_sha256": history_fingerprint(history),
              "protocol": "Fixed settings; train only prior draws. Retrospective 60-draw validation; not unseen prospective evidence. No post-result parameter search.",
              "score_semantics": "sum of model main-number marginal estimates per ticket; not a jackpot probability",
              "new_summary": summarize(records), "new_records": records,
              "uniform_100_runs": {"mean_ticket_hits": float(np.mean([s["mean_ticket_hits"] for s in summaries])),
                                   "mean_best_hits": float(np.mean([s["mean_best_hits"] for s in summaries])),
                                   "mean_draws_at_least": {str(n): float(np.mean([s["draws_at_least"][str(n)] for s in summaries])) for n in range(3, 7)}}}
    print(json.dumps({k: v for k, v in report.items() if k != "new_records"}, ensure_ascii=True), flush=True)
    return report


if __name__ == "__main__":
    main()
