"""Review saved predictions without regenerating numbers or modifying CSVs."""
import importlib.util
import argparse
from pathlib import Path
import pandas as pd
from lotto_validation import final_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Inspect local data and manual review; official source remains unverified")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("lotto_app", base / "로또 당첨번호 예측.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    history = pd.read_csv(app.CSV_FILE, encoding="utf-8-sig")
    predictions = pd.read_csv(app.PREDICTION_FILE, encoding="utf-8-sig")
    target = int(history["회차"].max()) + 1
    fetch_draw = (lambda _: None) if args.offline else app.get_lotto_draw
    result = final_review(history, predictions[predictions["예측회차"] == target],
                          predictions, fetch_draw, base)
    print("status:", result["status"], "finalized:", result["finalized"])
    if "payload" in result:
        print("official_status:", result["payload"]["official_status"])
    return result


if __name__ == "__main__":
    main()
