"""Review saved predictions without regenerating numbers or modifying CSVs."""
import importlib.util
import argparse
import json
import sys
from pathlib import Path
from lotto_validation import final_review, print_review
from lotto_storage import load_history, load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Inspect local data and manual review; official source remains unverified")
    parser.add_argument("--manual-response", action="store_true",
                        help="Read the manual review object from stdin; no intermediate files")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("lotto_app", base / "로또 당첨번호 예측.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    # 후보를 추측해 복원하지 않는다. 손상된 입력이면 검토 API 호출 전에 중단한다.
    history = load_history(app.CSV_FILE)
    predictions = load_predictions(app.PREDICTION_FILE)
    target = int(history["회차"].max()) + 1
    fetch_draw = (lambda _: None) if args.offline else app.get_lotto_draw
    proposal = json.load(sys.stdin) if args.manual_response else None
    result = final_review(history, predictions[predictions["예측회차"] == target],
                          predictions, fetch_draw, manual_review=proposal)
    print_review(result)
    if "payload" in result:
        print("official_status:", result["payload"]["official_status"])
    return result


if __name__ == "__main__":
    main()
