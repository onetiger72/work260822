"""Prepare an auditable replacement batch without deleting existing predictions.

Run: python -B lotto/regenerate_predictions.py --target 1241
The replacement builder consumes regenerated_1241.json after GPT review.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lotto_portfolio import VERSION
from lotto_validation import final_review, validate_history
from lotto_weekly import latest_completed_draw, verify_latest_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    app_path = base / "로또 당첨번호 예측.py"
    spec = importlib.util.spec_from_file_location("lotto_app", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    history = pd.read_csv(app.CSV_FILE, encoding="utf-8-sig")
    validate_history(history)
    verify_latest_result(history, app.get_lotto_draw, latest_completed_draw())
    if args.target != int(history["회차"].max()) + 1:
        raise ValueError("only the next unobserved round may be regenerated")
    original_bytes = Path(app.PREDICTION_FILE).read_bytes()
    predictions = pd.read_csv(app.PREDICTION_FILE, encoding="utf-8-sig")
    generated = app.generate_prediction_sets(history, target_draw=args.target)
    rows = []
    for row in generated.to_dict("records"):
        candidate = {column: "" for column in app.PREDICTION_FILE_COLUMNS}
        candidate.update({column: int(row[column]) for column in ["세트"] + app.NUMBER_COLUMNS})
        candidate.update({"예측회차": args.target, "종합점수": float(row["조합점수"])})
        rows.append(candidate)
    prepared = {"target_draw": args.target, "version": VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_csv_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "history_sha256": hashlib.sha256(Path(app.CSV_FILE).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256((base / "lotto_portfolio.py").read_bytes()).hexdigest(),
                "columns": app.PREDICTION_FILE_COLUMNS, "rows": rows,
                "weekly_review": generated.attrs["weekly_review"],
                "score_semantics": "model-estimated main hits per ticket, not a jackpot probability"}
    (base / f"regenerated_{args.target}.json").write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    result = final_review(history, pd.DataFrame(rows), predictions, app.get_lotto_draw, base)
    print("review_status:", result["status"], "finalized:", result["finalized"])
    print(generated.to_string(index=False))
    if Path(app.PREDICTION_FILE).read_bytes() != original_bytes:
        raise RuntimeError("prediction file changed during preparation")


if __name__ == "__main__":
    main()
