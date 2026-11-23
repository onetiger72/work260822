"""Prepare an auditable replacement batch without deleting existing predictions.

Run with --target set to the next unobserved draw, never a completed draw.
Prepared candidates and review results are returned in memory and printed.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lotto_selection import select_weekly_portfolio
from lotto_validation import final_review, print_review
from lotto_weekly import latest_completed_draw, verify_latest_result, verify_prediction_window
from lotto_storage import load_history, load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--manual-response", action="store_true",
                        help="Read the manual review object from stdin")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    app_path = base / "로또 당첨번호 예측.py"
    spec = importlib.util.spec_from_file_location("lotto_app", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    history = load_history(app.CSV_FILE)
    predictions = load_predictions(app.PREDICTION_FILE)
    verify_latest_result(history, app.get_lotto_draw, latest_completed_draw())
    if args.target != int(history["회차"].max()) + 1:
        raise ValueError("only the next unobserved round may be regenerated")
    verify_prediction_window(args.target)
    original_bytes = Path(app.PREDICTION_FILE).read_bytes()
    generated, evidence = select_weekly_portfolio(history, app.generate_prediction_sets,
                                                 set_count=app.PREDICTION_SET_COUNT)
    rows = []
    for row in generated.to_dict("records"):
        candidate = {column: "" for column in app.PREDICTION_FILE_COLUMNS}
        candidate.update({column: int(row[column]) for column in ["세트"] + app.NUMBER_COLUMNS})
        candidate.update({"예측회차": args.target, "종합점수": float(row["종합점수"])})
        rows.append(candidate)
    model_file, version, score_semantics = {
        "screened": ("lotto_portfolio.py", evidence["screened_version"], "model-estimated main hits per ticket"),
        "cluster": ("lotto.py", "cluster", "cluster heuristic score"),
        "shared_pairs": ("lotto_selection.py", "shared_pairs_uniform", "expected main hits per ticket under uniform 6/45"),
    }[evidence["selected"]]
    prepared = {"target_draw": args.target, "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_csv_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "history_sha256": hashlib.sha256(Path(app.CSV_FILE).read_bytes()).hexdigest(),
                "model_sha256": evidence["source_sha256"][model_file],
                "columns": app.PREDICTION_FILE_COLUMNS, "rows": rows,
                "weekly_review": generated.attrs.get("weekly_review"),
                "selection_evidence": evidence,
                "score_semantics": score_semantics}
    proposal = json.load(sys.stdin) if args.manual_response else None
    result = final_review(history, pd.DataFrame(rows), predictions, app.get_lotto_draw,
                          expected_count=app.PREDICTION_SET_COUNT, manual_review=proposal,
                          selection_evidence=evidence, mode="manual" if args.manual_response else None)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    print_review(result)
    print(generated.to_string(index=False))
    if Path(app.PREDICTION_FILE).read_bytes() != original_bytes:
        raise RuntimeError("prediction file changed during preparation")
    return {"prepared": prepared, "review": result}


if __name__ == "__main__":
    main()
