"""생성 전 읽기 전용 점검: 공식 조회·저장 예측 재채점·기법 검증을 화면에 출력한다.

공식 조회로 받은 새 회차도 이번 실행의 메모리에만 추가한다. --offline은
로컬 점검이며 공식 확인 성공을 뜻하지 않는다. 새 후보나 중간 파일은 저장하지 않는다.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lotto_portfolio import NUMBER_COLUMNS, portfolio_diagnostics, review_probabilities, ticket_hits
from lotto_validation import strict_int, validate_draw, validate_history
from lotto_storage import load_history, load_predictions, validate_prediction_history
from lotto_weekly import latest_completed_draw


def compare_saved_predictions(history, predictions):
    """원래 예측번호로 재채점해 저장 평가값과의 차이를 보고한다. 파일은 수정하지 않는다."""
    actuals = {row["회차"]: row for row in validate_history(history)}
    records = []
    if predictions.empty:
        return records
    # 일반 생성과 동일한 입력 검증을 사용한다. 점검에서 거부한 자료를
    # 일반 실행에서는 정수 절삭·회차 추정으로 받아들이는 차이를 없앤다.
    predictions = validate_prediction_history(predictions)
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
    """두 CSV 검증을 먼저 끝낸 후 조회한다. 일부 조회 실패는 보고 상태에 명시한다."""
    history_path, prediction_path = Path(app.CSV_FILE), Path(app.PREDICTION_FILE)
    history = load_history(history_path)
    predictions = load_predictions(prediction_path, missing_ok=True)
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
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
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
