"""시간 순서 비교: 과거 v4, 현재 모델, 동일 분산의 균등확률, 일반 무작위.

현재 모델과 balanced_uniform은 같은 선택기·세트 수·시드를 사용한다. 따라서
둘의 차이는 학습한 q의 효과이고, balanced_uniform과 일반 무작위의 차이는
분산 규칙의 효과다. 모든 상세 결과는 메모리로 반환하며 파일을 만들지 않는다.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import lotto_portfolio as current
from lotto_validation import validate_history
from lotto_storage import load_history


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_comparison(model_records, baseline_records):
    """같은 회차의 적중 차이를 요약한다. 표준오차는 예측 우위의 확정 판정이 아니다."""
    if not model_records or len(model_records) != len(baseline_records):
        raise ValueError("paired comparison needs matching, nonempty records")
    if [r["target_draw"] for r in model_records] != [r["target_draw"] for r in baseline_records]:
        raise ValueError("paired comparison round mismatch")
    result = {}
    for name, aggregate in [("mean_ticket_hits", np.mean), ("best_hits", max)]:
        deltas = np.array([float(aggregate(a["hits"])) - float(aggregate(b["hits"]))
                           for a, b in zip(model_records, baseline_records)])
        result[name] = {
            "mean_model_minus_baseline": float(deltas.mean()),
            "paired_standard_error": float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else None,
            "wins": int(np.count_nonzero(deltas > 1e-12)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            "losses": int(np.count_nonzero(deltas < -1e-12)),
        }
    return result


def run_benchmark(history, calculate_scores, old_model, draws=60, repetitions=100):
    """고정 규칙으로 과거 회차만 재생한다. 이 결과를 보고 규칙을 재조정하지 않는다.

    반복 횟수는 일반 무작위 기준의 시드 편차를 줄이기 위한 것이다. 현재 모델과
    균등 분산 기준은 실제 기본 생성과 같은 회차별 시드를 사용해 짝지어 비교한다.
    """
    validate_history(history)
    if not isinstance(draws, int) or draws < 1 or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("draws and repetitions must be positive integers")
    if len(history) <= 200:
        raise ValueError("benchmark requires at least 201 historical draws")
    history = history.sort_values("회차").reset_index(drop=True)
    # 캐시를 파일에서 읽거나 다른 이력과 공유하지 않는다. 공통 과거 접두 구간의
    # 원시 특징만 재사용하며, 모델별 선택/보정 결과는 각각 다시 계산한다.
    cache, records = {}, {"v4": [], "model": [], "balanced_uniform": []}
    uniform = [[] for _ in range(repetitions)]
    start = max(200, len(history) - draws)
    for n in range(max(150, start - 168), len(history)):
        cache[n] = calculate_scores(history.iloc[:n])
        if n % 25 == 0:
            print(f"Fresh feature prefix {n}/{len(history) - 1}", flush=True)
    identical_ticket_draws, active_signal_draws = 0, 0
    for n in range(start, len(history)):
        target = int(history.iloc[n]["회차"])
        actual = history.iloc[n][current.NUMBER_COLUMNS].tolist()
        generated = current.generate_prediction_sets(history.iloc[:n], calculate_scores, cache=cache)
        review = generated.attrs["weekly_review"]
        # 확률만 6/45로 바꾸고 선택기/시드는 같게 둔다. 모든 기법이 제외되어
        # 모델도 균등확률이면 두 결과가 완전히 같아야 한다.
        balanced = current.select_portfolio(np.full(45, 6 / 45), seed=review["selection_seed"])
        batches = {"v4": old_model.generate_prediction_sets(history.iloc[:n], calculate_scores, cache=cache),
                   "model": generated, "balanced_uniform": balanced}
        identical_ticket_draws += int(np.array_equal(generated[current.NUMBER_COLUMNS].to_numpy(),
                                                     balanced[current.NUMBER_COLUMNS].to_numpy()))
        active_signal_draws += int(bool(review["effective_methods"]))
        for name, tickets in batches.items():
            numbers = tickets[current.NUMBER_COLUMNS].to_numpy()
            records[name].append({"target_draw": target, "hits": current.ticket_hits(numbers, actual),
                                  **current.portfolio_diagnostics(numbers),
                                  "selected_strength": tickets.attrs.get("weekly_review", {}).get("selected_strength", 0.0)})
        for repetition in range(repetitions):
            rng = np.random.default_rng(7321 + 1000 * target + repetition)
            tickets = set()
            while len(tickets) < 10:
                tickets.add(tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False))))
            uniform[repetition].append({"hits": current.ticket_hits(tickets, actual)})
        if (n - start + 1) % 5 == 0:
            print(f"Past-draw comparison {n - start + 1}/{len(history) - start}", flush=True)
    summaries = [current.summarize(rows) for rows in uniform]
    report = {"version": current.VERSION, "old_version": old_model.VERSION,
              "history_sha256": current.history_fingerprint(history),
              "first_draw": int(history.iloc[start]["회차"]), "last_draw": int(history.iloc[-1]["회차"]),
              "protocol": "Fixed rules before replay; only prior draws per target, fresh feature cache. Retrospective, not unseen prospective proof. No current-week tickets generated.",
              "summaries": {name: current.summarize(rows) for name, rows in records.items()},
              "balanced_baseline": {"same_selector_and_seed": True,
                                    "identical_ticket_draws": identical_ticket_draws,
                                    "active_signal_draws": active_signal_draws},
              "model_vs_balanced_uniform": paired_comparison(records["model"], records["balanced_uniform"]),
              "uniform_runs": {"repetitions": repetitions,
                               **{key: float(np.mean([s[key] for s in summaries]))
                                  for key in ["mean_ticket_hits", "mean_best_hits", "zero_hit_draws"]},
                               "mean_draws_at_least": {str(k): float(np.mean([s["draws_at_least"][str(k)] for s in summaries]))
                                                       for k in range(1, 7)}},
              "records": records}
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=True), flush=True)
    return report


def main():
    base = Path(__file__).resolve().parent
    app = load_module("audit_app", base / "로또 당첨번호 예측.py")
    old = load_module("portfolio_v4", base / "analysis_cache/portfolio_v4.py")
    return run_benchmark(load_history(app.CSV_FILE), app.calculate_analysis_scores, old)


if __name__ == "__main__":
    main()
