"""추첨 이전 자료만 사용하는 본번호 보정·기법 선별·가중 분산 선택.

원시 특징 → 앞 24회 기법 선별 → 뒤 24회 반영 강도 결정 → 다음 회차 보정 →
노출/번호쌍 제약 안의 가중 추출 순서다. 학습 계수는 균등 6/45 쪽으로 줄인다.
기법을 모두 제외하면 같은 선택기에 균등확률을 전달한다. 구형 순위/모의 선택기는
비교용이며 기본 생성에서 호출하지 않는다. 계산과 보고서는 메모리에서만 유지한다.
"""
import hashlib
import heapq

import numpy as np
import pandas as pd

NUMBER_COLUMNS = [f"번호{i}" for i in range(1, 7)]
VERSION = "screened_weighted_balanced_v6"
CALIBRATION_DRAWS = 120
RIDGE = 120.0
# Convex gain gives progressively greater value to approaching six main hits.
HIT_UTILITY = np.array([0, 1, 3, 9, 27, 81, 243], dtype=float)


def history_fingerprint(df):
    # 경로 없는 to_csv는 메모리 문자열 직렬화다. 중간 CSV 파일은 만들지 않는다.
    return hashlib.sha256(df[["회차"] + NUMBER_COLUMNS + ["보너스"]].to_csv(index=False).encode()).hexdigest()


def fit_inclusion_probabilities(features, outcomes, current):
    """과거 특징(회차×45×기법)과 본번호 정답(회차×45)으로 계수를 학습한다.

    회차 안 특징 평균을 빼 상대 차이만 사용하고 ridge로 과적합을 억제한다.
    균등확률에서의 편차를 절반으로 줄인 뒤 [0.04, 0.30], 합계 6으로 투영한다.
    반환값은 보정된 모델 추정치이며 실제 추첨의 번호별 확률을 입증하지 않는다.
    """
    base = 6 / 45
    x = np.asarray(features, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    current = np.asarray(current, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(current).all() or not np.isfinite(y).all():
        raise ValueError("non-finite analysis feature")
    if len(x) == 0:
        return np.full(45, base)
    x = x - x.mean(axis=1, keepdims=True)
    current = current - current.mean(axis=0, keepdims=True)
    flat = x.reshape(-1, x.shape[-1])
    coefficients = np.linalg.solve(flat.T @ flat + RIDGE * np.eye(flat.shape[1]),
                                   flat.T @ (y.reshape(-1) - base))
    q = base + 0.5 * (current @ coefficients)
    # Projection to a bounded simplex keeps marginal total exactly six.
    low, high = 0.04 - q.max(), 0.30 - q.min()
    for _ in range(60):
        shift = (low + high) / 2
        if np.clip(q + shift, 0.04, 0.30).sum() < 6:
            low = shift
        else:
            high = shift
    return np.clip(q + (low + high) / 2, 0.04, 0.30)


def calibrated_probabilities(df, calculate_scores, cache=None, methods=None):
    """n+1회 예측에서 길이 t의 이력 특징을 0기반 t행 정답에 대응시킨다.

    t < n인 과거 사례만 학습한다. methods=None은 본번호 기법 전체, 빈 목록은
    기법 전부 제외를 뜻하므로 둘을 구분해야 한다. 자료 부족 시 균등확률이다.
    """
    cache = {} if cache is None else cache
    n = len(df)
    if n < 151:
        return np.full(45, 6 / 45)

    def scores_at(length):
        # cache는 동일 이력의 접두 구간끼리만 공유한다. 길이만 같은 다른 이력이나
        # 다른 특징 계산 함수에 재사용하면 결과가 섞이므로 호출자가 분리해야 한다.
        if length not in cache:
            cache[length] = calculate_scores(df.iloc[:length])
        return cache[length]

    current_scores = scores_at(n)
    methods = sorted(m for m in current_scores if m != "보너스보조신호") if methods is None else list(methods)
    if not methods:
        return np.full(45, 6 / 45)

    def matrix(length):
        scores = scores_at(length)
        return [[scores[m][number] for m in methods] for number in range(1, 46)]

    start = max(150, n - CALIBRATION_DRAWS)
    features = np.array([matrix(t) for t in range(start, n)])
    outcomes = np.zeros((n - start, 45))
    actual = df.iloc[start:n][NUMBER_COLUMNS].to_numpy(dtype=int) - 1
    for index, numbers in enumerate(actual):
        outcomes[index, numbers] = 1
    return fit_inclusion_probabilities(features, outcomes, matrix(n))


def sample_tickets(rng, weights, count):
    # Plackett-Luce sampling via exponential races; always six unique integers.
    races = -np.log(np.maximum(rng.random((count, 45)), 1e-15)) / weights
    return np.sort(np.argpartition(races, 5, axis=1)[:, :6], axis=1)


def screen_methods(df, calculate_scores, cache=None, screen_draws=24):
    """Retain individually calibrated signals only with >2 paired SE gain.

    This window ends before the subsequent strength-review window. A removed
    method contributes no feature to the final fit and is reconsidered next run.
    Screening is a conservative tuning heuristic, not a significance claim.
    """
    cache = {} if cache is None else cache
    # 이 함수에 전달되는 df는 후속 강도 검증 시작 직전에서 잘린 과거 이력이다.
    # 아래 평가 정답 행 t는 특징/학습 행 [:t] 안에 포함되지 않는다.
    start = max(151, len(df) - screen_draws)
    if len(df) not in cache:
        cache[len(df)] = calculate_scores(df)
    methods = sorted(m for m in cache[len(df)] if m != "보너스보조신호")
    selected, reports = [], []
    gains_by_method = {method: [] for method in methods}
    matrices = {}

    def matrix(t):
        if t not in matrices:
            if t not in cache:
                cache[t] = calculate_scores(df.iloc[:t])
            matrices[t] = np.array([[cache[t][m][n] for m in methods] for n in range(1, 46)])
        return matrices[t]

    for t in range(start, len(df)) if methods else []:
        train_start = max(150, t - CALIBRATION_DRAWS)
        features = np.array([matrix(k) for k in range(train_start, t)])
        outcomes = np.zeros((t - train_start, 45))
        actual_numbers = df.iloc[train_start:t][NUMBER_COLUMNS].to_numpy(dtype=int) - 1
        outcomes[np.arange(len(outcomes))[:, None], actual_numbers] = 1
        actual = np.zeros(45)
        actual[df.iloc[t][NUMBER_COLUMNS].to_numpy(dtype=int) - 1] = 1
        for index, method in enumerate(methods):
            q = fit_inclusion_probabilities(features[:, :, index:index + 1], outcomes,
                                            matrix(t)[:, index:index + 1])
            gains_by_method[method].append(float(np.mean((6 / 45 - actual) ** 2) - np.mean((q - actual) ** 2)))
    for method in methods:
        gains = gains_by_method[method]
        mean = float(np.mean(gains)) if gains else 0.0
        se = float(np.std(gains, ddof=1) / np.sqrt(len(gains))) if len(gains) >= 2 else None
        active = len(gains) >= 12 and mean > 2 * se + 1e-12
        if active:
            selected.append(method)
        reports.append({"method": method, "enabled": active, "mean_brier_gain": mean,
                        "paired_standard_error": se, "reason": "gain_above_two_se" if active
                        else "insufficient_history" if len(gains) < 12 else "no_reliable_gain"})
    return selected, {"rounds": df.iloc[start:]["회차"].astype(int).tolist(),
                      "selected_methods": selected, "methods": reports,
                      "excluded_by_policy": ["보너스보조신호"]}


def review_probabilities(df, calculate_scores, cache=None, review_draws=24):
    """Choose signal strength using strictly past, rolling Brier losses.

    A one-standard-error rule favors weaker signals when differences are small.
    These are tuning results, not an independent estimate of future accuracy.
    """
    cache = {} if cache is None else cache
    strengths = np.array([0.0, 0.25, 0.5, 1.0])
    losses, rounds = [], []
    base = np.full(45, 6 / 45)
    # 앞 구간에서 기법을 선택하고 뒤 구간에서는 그 목록을 고정한다.
    # 선택/강도 조정에 사용한 성적을 별도 실전 예측 성적으로 보고하지 않는다.
    review_start = max(151, len(df) - review_draws)
    methods, method_report = screen_methods(df.iloc[:min(review_start, len(df))], calculate_scores, cache)
    for t in range(review_start, len(df)):
        q = calibrated_probabilities(df.iloc[:t], calculate_scores, cache, methods=methods)
        actual = np.zeros(45)
        actual[df.iloc[t][NUMBER_COLUMNS].to_numpy(dtype=int) - 1] = 1
        candidates = base + strengths[:, None] * (q - base)
        losses.append(np.mean((candidates - actual) ** 2, axis=1))
        rounds.append(int(df.iloc[t]["회차"]))
    chosen = 0
    mean_losses = None
    if len(losses) >= 12:
        losses = np.asarray(losses)
        mean_losses = losses.mean(axis=0)
        best = int(np.argmin(mean_losses))
        # Paired errors account for the same actual draw in each candidate.
        differences = losses - losses[:, best, None]
        uncertainty = differences.std(axis=0, ddof=1) / np.sqrt(len(losses))
        chosen = int(np.flatnonzero(mean_losses - mean_losses[best] <= uncertainty + 1e-12)[0])
    q = calibrated_probabilities(df, calculate_scores, cache, methods=methods)
    report = {
        "version": VERSION, "history_sha256": history_fingerprint(df),
        "target_draw": int(df.iloc[-1]["회차"]) + 1,
        "review_rounds": rounds, "selected_strength": float(strengths[chosen]),
        "candidate_strengths": strengths.tolist(),
        "mean_brier_losses": None if mean_losses is None else mean_losses.tolist(),
        "last_draw_brier_losses": None if not len(losses) else np.asarray(losses)[-1].tolist(),
        "reason": "paired_one_standard_error" if len(losses) >= 12 else "insufficient_history",
        "method_screening": method_report,
        "effective_methods": methods if strengths[chosen] else [],
        "limits": "Retrospective tuning, not held-out proof of improvement or winning probabilities.",
    }
    return base + strengths[chosen] * (q - base), report


def select_ranked_portfolio(probabilities, set_count=10, seed=0, candidate_count=1600, scenario_count=2400):
    """Exact top distinct six-number sets under a conditional Bernoulli model.

    Conditioning independent inclusions on exactly six numbers makes set mass
    proportional to the product of inclusion odds. This is a model assumption,
    not evidence of unequal real lottery probabilities. Seed and pool arguments
    remain accepted for caller compatibility; no random search is performed.
    The displayed score remains the sum of marginals for CSV compatibility.
    """
    q = np.asarray(probabilities, dtype=float)
    if (q.shape != (45,) or not np.isfinite(q).all()
            or np.any(q <= 0) or np.any(q >= 1)):
        raise ValueError("invalid probabilities")
    if not isinstance(set_count, (int, np.integer)) or not 1 <= set_count <= candidate_count:
        raise ValueError("invalid set_count")
    # Uniform fallback must not prefer small numbers merely by array position.
    order = np.lexsort((np.random.default_rng(seed).random(45), -q))
    log_odds = np.log(q[order]) - np.log1p(-q[order])
    start = tuple(range(6))
    heap = [(-float(log_odds[list(start)].sum()), start)]
    seen = {start}
    rows = []
    while heap and len(rows) < set_count:
        _, ranks = heapq.heappop(heap)
        numbers = np.sort(order[list(ranks)])
        row = {"세트": len(rows) + 1,
               "조합점수": round(float(q[numbers].sum()), 6)}
        row.update(dict(zip(NUMBER_COLUMNS, (numbers + 1).tolist())))
        rows.append(row)
        for position in range(6):
            limit = ranks[position + 1] if position < 5 else 45
            if ranks[position] + 1 >= limit:
                continue
            neighbor = list(ranks)
            neighbor[position] += 1
            neighbor = tuple(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                heapq.heappush(heap, (-float(log_odds[list(neighbor)].sum()), neighbor))
    return pd.DataFrame(rows)[["세트"] + NUMBER_COLUMNS + ["조합점수"]]


def select_portfolio(probabilities, set_count=10, seed=0, candidate_count=1600, scenario_count=2400):
    """노출 균등화·번호쌍 재사용 최소화 뒤 확률 크기에 따른 가중 추출을 한다.

    10세트는 45개 전체를 포함하며 번호별 최대 2회 사용한다. 같은 제약 우선순위
    안에서 -log(U) / (q / (1-q))가 작은 번호를 선택한다. 기존 순위 비교와 달리
    확률 차이의 크기가 선택에 영향을 주므로 25/50/100% 수축도 의미가 있다.
    동일 seed는 같은 난수를 사용하지만 강도가 달라도 우연히 같은 결과일 수 있다.
    q는 샘플링 가중치의 근거이며 제약을 거친 실제 노출 빈도와 동일하지 않다.
    분산과 가중 추출은 실제 추첨의 예측 우위나 당첨을 보장하지 않는다.
    """
    q = np.asarray(probabilities, dtype=float)
    if (q.shape != (45,) or not np.isfinite(q).all()
            or np.any(q <= 0) or np.any(q >= 1)):
        raise ValueError("invalid probabilities")
    if (isinstance(set_count, (bool, np.bool_)) or not isinstance(set_count, (int, np.integer))
            or not 1 <= set_count <= min(10, candidate_count)):
        raise ValueError("set_count must be between 1 and 10")
    rng = np.random.default_rng(seed)
    weights = q / (1 - q)
    exposure = np.zeros(45, dtype=int)
    pairs = np.zeros((45, 45), dtype=int)
    rows = []
    for set_id in range(1, set_count + 1):
        chosen = []
        for _ in range(6):
            pair_cost = pairs[:, chosen].sum(axis=1)
            # 모든 슬롯에서 45개의 난수를 소비하므로 동일 seed의 균등 기준과
            # 비교할 때 난수 차이가 아니라 q의 효과를 관찰할 수 있다.
            races = -np.log(np.maximum(rng.random(45), 1e-15)) / weights
            # lexsort의 마지막 키가 최우선이다. 추정치가 강해도 노출 상한을
            # 무시하지 않으며, 선택한 번호는 현재 세트에서 다시 고르지 않는다.
            order = np.lexsort((races, pair_cost, exposure))
            number = next(int(n) for n in order if n not in chosen)
            chosen.append(number)
            exposure[number] += 1
        numbers = np.sort(chosen)
        pairs[np.ix_(numbers, numbers)] += 1
        rows.append({"세트": set_id, "조합점수": round(float(q[numbers].sum()), 6),
                     **dict(zip(NUMBER_COLUMNS, (numbers + 1).tolist()))})
    return pd.DataFrame(rows)[["세트"] + NUMBER_COLUMNS + ["조합점수"]]


def portfolio_diagnostics(tickets):
    """검증된 후보의 번호 커버리지·최대 사용 횟수·세트 간 최대 겹침을 요약한다."""
    numbers = np.asarray(tickets, dtype=int)
    counts = np.bincount(numbers.ravel(), minlength=46)[1:46]
    overlaps = [len(set(a) & set(b)) for i, a in enumerate(numbers) for b in numbers[i + 1:]]
    return {"unique_numbers": int(np.count_nonzero(counts)),
            "max_number_exposure": int(counts.max()),
            "max_pairwise_overlap": max(overlaps, default=0)}


def select_simulated_portfolio(probabilities, set_count=10, seed=0, candidate_count=1600, scenario_count=2400):
    if not 1 <= set_count <= candidate_count:
        raise ValueError("invalid set_count")
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (45,) or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0) or np.any(probabilities >= 1)):
        raise ValueError("invalid probabilities")
    rng = np.random.default_rng(seed)
    weights = probabilities / (1 - probabilities)
    # Candidate pool includes diverse uniform draws and high-score draws.
    candidates = np.unique(np.vstack([
        sample_tickets(rng, weights, candidate_count),
        sample_tickets(rng, np.ones(45), candidate_count // 2),
        np.sort(np.argsort(-probabilities)[:6])[None, :]
    ]), axis=0)
    if len(candidates) < set_count:
        raise ValueError("insufficient distinct candidates")
    # Weighted sampling is a scenario heuristic: its inclusion marginals are
    # not exactly the input estimates, and utility is not a jackpot probability.
    scenarios = sample_tickets(rng, weights, scenario_count)
    candidate_matrix = np.zeros((len(candidates), 45), dtype=np.int16)
    scenario_matrix = np.zeros((len(scenarios), 45), dtype=np.int16)
    candidate_matrix[np.arange(len(candidates))[:, None], candidates] = 1
    scenario_matrix[np.arange(len(scenarios))[:, None], scenarios] = 1
    utilities = HIT_UTILITY[candidate_matrix @ scenario_matrix.T]
    covered = np.zeros(scenario_count)
    chosen = []
    # Optimize marginal expected best-ticket gain, not total number coverage.
    for _ in range(set_count):
        gains = np.maximum(utilities - covered, 0).mean(axis=1)
        gains[chosen] = -np.inf
        index = int(np.argmax(gains))
        chosen.append(index)
        covered = np.maximum(covered, utilities[index])
    rows = []
    for set_id, index in enumerate(chosen, 1):
        numbers = candidates[index]
        row = {"세트": set_id, "조합점수": round(float(probabilities[numbers].sum()), 6)}
        row.update(dict(zip(NUMBER_COLUMNS, (numbers + 1).tolist())))
        rows.append(row)
    return pd.DataFrame(rows)[["세트"] + NUMBER_COLUMNS + ["조합점수"]]


def generate_prediction_sets(df, calculate_scores, set_count=10, target_draw=None,
                             random_seed=None, cache=None):
    """검증된 연속 이력 다음 회차의 후보와 생성 근거를 함께 반환한다.

    회차별 기본 시드는 재현과 균등 분산 기준의 짝비교에 공통으로 사용한다.
    호출자는 CSV/공식 결과 검증을 먼저 해야 한다. 이 함수는 저장을 하지 않는다.
    """
    df = df.sort_values("회차").reset_index(drop=True)
    if df["회차"].duplicated().any():
        raise ValueError("duplicate training rounds")
    target = int(df["회차"].max()) + 1
    if target_draw is not None and target_draw != target:
        raise ValueError("target must follow the last training draw")
    probabilities, report = review_probabilities(df, calculate_scores, cache)
    seed = int(random_seed) if random_seed is not None else target * 10007 + len(df)
    result = select_portfolio(probabilities, set_count=set_count, seed=seed)
    report["selection_policy"] = "balanced_exposure_then_pair_reuse_then_weighted_odds"
    report["selection_seed"] = seed
    report["portfolio_diagnostics"] = portfolio_diagnostics(result[NUMBER_COLUMNS].to_numpy())
    result.attrs["weekly_review"] = report
    return result


def ticket_hits(tickets, actual):
    """검증된 본번호 여섯 개와의 교집합만 센다. actual에 보너스를 넣지 않는다."""
    actual = set(int(n) for n in actual)
    return [len(actual & set(int(n) for n in row)) for row in tickets]


def summarize(records):
    """회차별 같은 세트 예산의 결과를 요약한다. 최고 적중은 회차마다 따로 구한다."""
    best = [max(row["hits"]) for row in records]
    return {"draws": len(records), "mean_ticket_hits": float(np.mean([row["hits"] for row in records])),
            "mean_best_hits": float(np.mean(best)),
            "zero_hit_draws": sum(h == 0 for h in best),
            "draws_at_least": {str(n): sum(h >= n for h in best) for n in range(1, 7)}}
