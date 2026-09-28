"""Pre-generation audit: fetch results and review logic without writing tickets."""
import argparse
import hashlib
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lotto_portfolio import NUMBER_COLUMNS, portfolio_diagnostics, review_probabilities, ticket_hits
from lotto_validation import strict_int, validate_draw, validate_history, write_json
from lotto_weekly import latest_completed_draw


def compare_saved_predictions(history, predictions):
    actuals = {row["회차"]: row for row in validate_history(history)}
    records = []
    if predictions.empty:
        return records
    predictions = predictions.copy()
    for column in ["예측회차", "세트"] + NUMBER_COLUMNS:
        predictions[column] = predictions[column].map(strict_int)
    if predictions.duplicated(["예측회차", "세트"]).any():
        raise ValueError("duplicate prediction round/set")
    for target, group in predictions.groupby("예측회차", sort=True):
        tickets = group[NUMBER_COLUMNS].to_numpy(dtype=int)
        if (int(target) < 1 or (group["세트"] < 1).any()
                or any(len(set(row)) != 6 or min(row) < 1 or max(row) > 45 for row in tickets)):
            raise ValueError("invalid saved prediction")
        record = {"draw": int(target), "sets": len(group),
                  "set_ids": group["세트"].tolist(), **portfolio_diagnostics(tickets)}
        if target in actuals:
            actual = actuals[target]
            hits = ticket_hits(tickets, [actual[c] for c in NUMBER_COLUMNS])
            bonus = [int(actual["보너스"] in row) for row in tickets]
            record.update(actual=actual, main_hits=hits, bonus_hits=bonus,
                          best_main_hits=max(hits), mean_main_hits=sum(hits) / len(hits),
                          zero_hit_sets=hits.count(0))
            stale = []
            for (_, row), main, extra in zip(group.iterrows(), hits, bonus):
                for column, expected in [("본번호일치수", main), ("보너스일치", extra),
                                         ("총일치수", main + extra)]:
                    if column in row and pd.notna(row[column]) and strict_int(row[column]) != expected:
                        stale.append({"set_id": int(row["세트"]), "column": column,
                                      "stored": strict_int(row[column]), "correct": expected})
            record["stored_evaluation_mismatches"] = stale
        else:
            record["status"] = "result_unavailable"
        records.append(record)
    return records


def audit(app, offline=False):
    history_path, prediction_path = Path(app.CSV_FILE), Path(app.PREDICTION_FILE)
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    validate_history(history)
    history = history.sort_values("회차").reset_index(drop=True)
    expected = latest_completed_draw()
    official_status, unavailable = "offline", []
    if not offline:
        # Verify the stored last row as well as collect every missing round.
        official_status = "matched"
        for target in range(int(history["회차"].max()), expected + 1):
            row = app.get_lotto_draw(target)
            if row is None:
                unavailable.append(target)
                official_status = "unavailable"
                break
            normalized = validate_draw(row)
            if normalized["회차"] != target:
                raise ValueError("official response round mismatch")
            stored = history[history["회차"] == target]
            if not stored.empty:
                if validate_draw(stored.iloc[0].to_dict()) != normalized:
                    raise ValueError("official/stored result mismatch")
            else:
                history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    validate_history(history)
    predictions = (pd.read_csv(prediction_path, encoding="utf-8-sig") if prediction_path.exists()
                   else pd.DataFrame())
    comparisons = compare_saved_predictions(history, predictions)
    print("Reviewing methods on chronological historical prefixes...", flush=True)
    _, review = review_probabilities(history, app.calculate_analysis_scores)
    report = {"mode": "audit_only_no_new_tickets", "checked_at": datetime.now(timezone.utc).isoformat(),
              "expected_completed_draw": expected, "available_completed_draw": int(history["회차"].max()),
              "official_status": official_status, "unavailable_rounds": unavailable,
              "ready_for_generation": official_status == "matched" and int(history["회차"].max()) == expected,
              "input_history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
              "input_predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()
              if prediction_path.exists() else None,
              "comparisons": comparisons, "logic_review": review,
              "limits": "CSV files are unchanged. Historical replay is not proof of pre-draw predictions or future superiority."}
    output = history_path.parent / "실행전점검.json"
    write_json(output, report)
    print(f"Audit saved: {output.name}; official_status={official_status}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "로또 당첨번호 예측.py"
    spec = importlib.util.spec_from_file_location("lotto_app_audit", path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    audit(app, offline=args.offline)


if __name__ == "__main__":
    main()
