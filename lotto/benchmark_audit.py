"""Compare v4, v5, and uniform ten-ticket batches on past draws only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lotto_portfolio as current
from lotto_validation import validate_draw, validate_history, write_json


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base = Path(__file__).resolve().parent
    app = load_module("audit_app", base / "로또 당첨번호 예측.py")
    old = load_module("portfolio_v4", base / "analysis_cache/portfolio_v4.py")
    history = pd.read_csv(app.CSV_FILE, encoding="utf-8-sig")
    snapshot = base / "analysis_cache/latest_official_draw.json"
    if snapshot.exists():
        row = validate_draw(json.loads(snapshot.read_text(encoding="utf-8")))
        if row["회차"] == int(history["회차"].max()) + 1:
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    validate_history(history)
    history = history.sort_values("회차").reset_index(drop=True)
    # Fresh cache, shared only across immutable prefixes in this process.
    cache, records = {}, {"v4": [], "v5": []}
    uniform = [[] for _ in range(100)]
    start = max(200, len(history) - 60)
    for n in range(max(150, start - 168), len(history)):
        cache[n] = app.calculate_analysis_scores(history.iloc[:n])
        if n % 25 == 0:
            print(f"Fresh feature prefix {n}/{len(history) - 1}", flush=True)
    for n in range(start, len(history)):
        target = int(history.iloc[n]["회차"])
        actual = history.iloc[n][current.NUMBER_COLUMNS].tolist()
        for name, model in [("v4", old), ("v5", current)]:
            tickets = model.generate_prediction_sets(history.iloc[:n], app.calculate_analysis_scores, cache=cache)
            numbers = tickets[current.NUMBER_COLUMNS].to_numpy()
            records[name].append({"target_draw": target, "hits": current.ticket_hits(numbers, actual),
                                  **current.portfolio_diagnostics(numbers),
                                  "selected_strength": tickets.attrs["weekly_review"]["selected_strength"]})
        for repetition in range(100):
            rng = np.random.default_rng(7321 + 1000 * target + repetition)
            tickets = set()
            while len(tickets) < 10:
                tickets.add(tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False))))
            uniform[repetition].append({"hits": current.ticket_hits(tickets, actual)})
        if (n - start + 1) % 5 == 0:
            print(f"Past-draw comparison {n - start + 1}/{len(history) - start}", flush=True)
    summaries = [current.summarize(rows) for rows in uniform]
    report = {"version": current.VERSION, "old_version": old.VERSION,
              "history_sha256": current.history_fingerprint(history),
              "first_draw": int(history.iloc[start]["회차"]), "last_draw": int(history.iloc[-1]["회차"]),
              "protocol": "Fixed rules before replay; only prior draws per target, fresh feature cache. Retrospective, not unseen prospective proof. No current-week tickets generated.",
              "summaries": {name: current.summarize(rows) for name, rows in records.items()},
              "uniform_100_runs": {key: float(np.mean([s[key] for s in summaries]))
                                   for key in ["mean_ticket_hits", "mean_best_hits", "zero_hit_draws"]},
              "records": records}
    write_json(base / "실행전비교.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
