# 비교용 과거 알고리즘. 수치 계산은 보존하고 중간 파일 저장·앱 실행 경로는 제거했습니다.

import os


from lotto_validation import validate_history

import math


import requests

import pandas as pd


import random

from collections import Counter


from itertools import combinations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "과거로또 당첨번호.csv")

PREDICTION_SET_COUNT = 10

PREDICTION_CANDIDATE_ATTEMPTS = 15000

BACKTEST_DRAWS = 120

MIN_BACKTEST_TRAIN_DRAWS = 150

RECENT_REVIEW_DRAWS = 2

RECENT_REVIEW_PRIOR_DRAWS = 18

NUMBER_COLUMNS = [
    "번호1",
    "번호2",
    "번호3",
    "번호4",
    "번호5",
    "번호6"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Referer": "https://www.dhlottery.co.kr/lt645/result",
    "Origin": "https://www.dhlottery.co.kr",
    "X-Requested-With": "XMLHttpRequest"
}

HTTP_SESSION = requests.Session()

HTTP_SESSION.headers.update(HEADERS)

def load_history_csv():
    """
    기존 당첨 이력 CSV를 읽고 필수 열, 번호 범위,
    본 번호 중복, 보너스 번호 및 회차 중복을 검증합니다.
    """

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            CSV_FILE,
            encoding="utf-8-sig"
        )

        if df.empty or "회차" not in df.columns:
            return pd.DataFrame()

        required_columns = ["회차"] + NUMBER_COLUMNS + ["보너스"]
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            print(
                "[CSV 형식 오류] 필수 열 누락 : "
                + ", ".join(missing_columns)
            )
            return pd.DataFrame()

        # Validate raw values before numeric coercion can truncate fractions or
        # silently discard conflicting/missing historical draws.
        validate_history(df)

        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(
            subset=required_columns
        )

        for column in required_columns:
            df[column] = df[column].astype(int)

        valid_number_range = df[NUMBER_COLUMNS].apply(
            lambda column: column.between(1, 45)
        ).all(axis=1)
        unique_main_numbers = (
            df[NUMBER_COLUMNS].nunique(axis=1) == 6
        )
        valid_bonus = df["보너스"].between(1, 45)
        bonus_not_in_main = ~df.apply(
            lambda row: int(row["보너스"]) in {
                int(row[column])
                for column in NUMBER_COLUMNS
            },
            axis=1
        )
        valid_rows = (
            valid_number_range
            & unique_main_numbers
            & valid_bonus
            & bonus_not_in_main
            & (df["회차"] >= 1)
        )
        invalid_count = int((~valid_rows).sum())

        if invalid_count:
            print(f"[CSV 검증] 잘못된 당첨 데이터 {invalid_count}행을 제외합니다.")

        df = df[valid_rows].copy()
        duplicate_count = int(df["회차"].duplicated(keep="last").sum())

        if duplicate_count:
            print(f"[CSV 검증] 중복 회차 {duplicate_count}행을 제외합니다.")

        df = (
            df
            .drop_duplicates(subset=["회차"], keep="last")
            .sort_values("회차")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:
        print(f"[CSV 로드 오류] {e}")
        return pd.DataFrame()

def _get_draw_sets(df):
    """데이터프레임의 본 당첨번호를 회차별 set으로 변환합니다."""

    return [
        set(int(number) for number in row)
        for row in df[NUMBER_COLUMNS].itertuples(
            index=False,
            name=None
        )
    ]

def _normalize_score_map(raw_scores):
    """번호별 임의 점수를 0.05~1.00 범위로 정규화합니다."""

    cleaned = {}

    for number in range(1, 46):
        value = float(raw_scores.get(number, 0.0))
        cleaned[number] = value if math.isfinite(value) else 0.0

    minimum = min(cleaned.values())
    maximum = max(cleaned.values())

    if math.isclose(minimum, maximum):
        return {number: 0.5 for number in range(1, 46)}

    score_range = maximum - minimum

    return {
        number: 0.05 + 0.95 * (
            cleaned[number] - minimum
        ) / score_range
        for number in range(1, 46)
    }

def _score_frequency(df, recent_count=None):
    """전체 또는 최근 N회 단순 출현빈도 점수입니다."""

    sample = df if recent_count is None else df.tail(recent_count)
    counts = Counter(
        int(number)
        for number in sample[NUMBER_COLUMNS].values.flatten()
    )

    return _normalize_score_map({
        number: counts.get(number, 0)
        for number in range(1, 46)
    })

def _score_exponential_frequency(df, half_life=35):
    """최근 회차일수록 큰 값을 주는 지수 감쇠 출현빈도입니다."""

    draws = _get_draw_sets(df)
    decay = math.log(2) / max(half_life, 1)
    raw_scores = {number: 0.0 for number in range(1, 46)}

    for age, draw in enumerate(reversed(draws)):
        weight = math.exp(-decay * age)

        for number in draw:
            raw_scores[number] += weight

    return _normalize_score_map(raw_scores)

def _score_bonus_support(df, half_life=120):
    """
    lotto.py의 보너스 빈도와 시간감쇠 신호를 낮은 사전가중치로 사용합니다.
    보너스는 본번호와 역할이 다르므로 독립적인 주력 신호로 취급하지 않습니다.
    """

    bonus_counts = Counter(
        int(number)
        for number in df["보너스"].values.flatten()
    )
    bonus_decay = {number: 0.0 for number in range(1, 46)}
    decay = math.log(2) / max(half_life, 1)

    for age, number in enumerate(reversed(df["보너스"].astype(int).tolist())):
        bonus_decay[number] += math.exp(-decay * age)

    frequency_scores = _normalize_score_map(bonus_counts)
    decay_scores = _normalize_score_map(bonus_decay)

    return _normalize_score_map({
        number: (
            frequency_scores[number] * 0.50
            + decay_scores[number] * 0.50
        )
        for number in range(1, 46)
    })

def _score_recent_trend(df, recent_count=20, comparison_count=80):
    """최근 구간 출현률과 그 이전 구간 출현률의 변화량입니다."""

    total_count = len(df)

    if total_count <= recent_count:
        return _score_frequency(df)

    recent = df.tail(recent_count)
    comparison_end = total_count - recent_count
    comparison_start = max(0, comparison_end - comparison_count)
    comparison = df.iloc[comparison_start:comparison_end]

    if comparison.empty:
        return _score_frequency(df, recent_count=recent_count)

    recent_counts = Counter(
        int(number)
        for number in recent[NUMBER_COLUMNS].values.flatten()
    )
    comparison_counts = Counter(
        int(number)
        for number in comparison[NUMBER_COLUMNS].values.flatten()
    )

    raw_scores = {}

    for number in range(1, 46):
        recent_rate = recent_counts.get(number, 0) / len(recent)
        comparison_rate = (
            comparison_counts.get(number, 0) / len(comparison)
        )
        raw_scores[number] = recent_rate - comparison_rate

    return _normalize_score_map(raw_scores)

def _score_gap_cycle(df):
    """현재 미출현 길이와 번호별 과거 출현 간격을 함께 봅니다."""

    draws = _get_draw_sets(df)
    total_count = len(draws)
    raw_scores = {}

    for number in range(1, 46):
        appeared_at = [
            index
            for index, draw in enumerate(draws)
            if number in draw
        ]

        if len(appeared_at) < 2:
            raw_scores[number] = 0.0
            continue

        intervals = [
            appeared_at[index] - appeared_at[index - 1]
            for index in range(1, len(appeared_at))
        ]
        average_interval = sum(intervals) / len(intervals)
        variance = sum(
            (interval - average_interval) ** 2
            for interval in intervals
        ) / len(intervals)
        standard_deviation = math.sqrt(variance)

        current_missing = total_count - 1 - appeared_at[-1]
        next_interval = current_missing + 1

        proximity = math.exp(
            -abs(next_interval - average_interval)
            / max(average_interval, 1.0)
        )
        overdue = min(
            next_interval / max(average_interval, 1.0),
            2.5
        ) / 2.5
        consistency = 1.0 / (
            1.0 + standard_deviation / max(average_interval, 1.0)
        )

        raw_scores[number] = (
            proximity * 0.45
            + overdue * 0.40
            + consistency * 0.15
        )

    return _normalize_score_map(raw_scores)

def _score_draw_transition(df, lookback=500):
    """현재 직전 회차와 비슷했던 과거 회차의 다음 결과를 분석합니다."""

    draws = _get_draw_sets(df)

    if len(draws) < 2:
        return {number: 0.5 for number in range(1, 46)}

    latest_draw = draws[-1]
    raw_scores = {number: 0.0 for number in range(1, 46)}
    start_index = max(1, len(draws) - lookback)

    for index in range(start_index, len(draws)):
        similarity = len(draws[index - 1] & latest_draw)

        if similarity == 0:
            continue

        age = len(draws) - 1 - index
        recency_weight = math.exp(-age / 250)

        for number in draws[index]:
            raw_scores[number] += similarity * recency_weight

    return _normalize_score_map(raw_scores)

def _score_pair_network(df, lookback=500, anchor_count=10):
    """최근 활발한 번호들과 과거에 함께 나온 번호의 연결도를 봅니다."""

    draws = _get_draw_sets(df.tail(lookback))
    recent_draws = draws[-30:]
    recent_counts = Counter(
        number
        for draw in recent_draws
        for number in draw
    )
    anchors = [
        number
        for number, _ in sorted(
            recent_counts.items(),
            key=lambda item: (-item[1], item[0])
        )[:anchor_count]
    ]

    pair_counts = Counter()

    for age, draw in enumerate(reversed(draws)):
        weight = math.exp(-age / 250)

        for pair in combinations(sorted(draw), 2):
            pair_counts[pair] += weight

    raw_scores = {}

    for number in range(1, 46):
        score = 0.0

        for anchor in anchors:
            if anchor == number:
                continue

            pair = tuple(sorted((number, anchor)))
            anchor_weight = 1.0 + recent_counts.get(anchor, 0) / 10
            score += pair_counts.get(pair, 0.0) * anchor_weight

        raw_scores[number] = score

    return _normalize_score_map(raw_scores)

def _score_hot_cold_balance(df, lookback=10):
    """최근 단기 출현을 핫/웜/콜드 구간으로 나누어 균형 신호를 계산합니다."""
    draws = _get_draw_sets(df.tail(lookback))
    counts = Counter(number for draw in draws for number in draw)
    raw_scores = {}
    for number in range(1, 46):
        count = counts.get(number, 0)
        raw_scores[number] = (
            0.85 + 0.15 * min(count / 4, 1.0)
            if count >= 2 else 0.65 if count == 1 else 0.40
        )
    return _normalize_score_map(raw_scores)

def _score_markov_transition(
        df,
        lookback=400,
        prior_strength=18.0
):
    """직전 당첨번호에서 다음 회차로 이어진 번호의 전이 확률을 반영합니다."""
    draws = _get_draw_sets(df.tail(lookback))
    if len(draws) < 2:
        return {number: 0.5 for number in range(1, 46)}

    transitions = Counter()
    row_totals = Counter()
    for previous, current in zip(draws, draws[1:]):
        for source in previous:
            row_totals[source] += 1
            for target in current:
                transitions[(source, target)] += 1

    raw_scores = {number: 0.0 for number in range(1, 46)}
    latest = draws[-1]
    base_probability = 6 / 45
    for source in latest:
        denominator = row_totals[source] + prior_strength
        for target in range(1, 46):
            raw_scores[target] += (
                transitions[(source, target)]
                + prior_strength * base_probability
            ) / denominator
    return _normalize_score_map(raw_scores)

def _score_last_digit(df, lookback=100):
    """끝수별 전체·최근 분포를 결합해 특정 끝수 편중을 완화합니다."""
    draws = _get_draw_sets(df.tail(lookback))
    if not draws:
        return {number: 0.5 for number in range(1, 46)}
    digit_counts = Counter(number % 10 for draw in draws for number in draw)
    recent = draws[-5:]
    recent_counts = Counter(number % 10 for draw in recent for number in draw)
    digit_capacity = Counter(number % 10 for number in range(1, 46))
    raw_scores = {
        number: (
            digit_counts.get(number % 10, 0)
            / (len(draws) * digit_capacity[number % 10])
            * 0.4
            + recent_counts.get(number % 10, 0)
            / (max(len(recent), 1) * digit_capacity[number % 10])
            * 0.6
        )
        for number in range(1, 46)
    }
    return _normalize_score_map(raw_scores)

def _score_recent_two_draw_context(
        df,
        lookback=500,
        number_prior_strength=24.0,
        context_prior_strength=18.0
):
    """
    최근 두 회차의 출현 상태와 유사한 과거 2회 문맥의 다음 결과를 결합합니다.

    번호마다 '두 회차 모두 미출현/앞 회차만/최근 회차만/두 회차 모두'
    네 상태의 다음 회차 출현률을 구하고, 표본이 적은 상태는 전체 상태
    출현률 쪽으로 수축합니다. 이어서 현재 두 회차와 번호 겹침이 비슷했던
    과거 두 회차 뒤의 결과를 약한 보조 신호로 사용합니다.
    """

    draws = _get_draw_sets(df.tail(lookback + 2))

    if len(draws) < 3:
        return {number: 0.5 for number in range(1, 46)}

    current_older, current_latest = draws[-2:]
    state_trials = Counter()
    state_hits = Counter()
    number_state_trials = Counter()
    number_state_hits = Counter()
    context_hits = Counter()
    context_total_weight = 0.0
    context_candidates = []
    base_probability = 6 / 45

    def presence_state(older_draw, latest_draw, number):
        return (
            int(number in older_draw)
            + 2 * int(number in latest_draw)
        )

    for target_index in range(2, len(draws)):
        older_draw = draws[target_index - 2]
        latest_draw = draws[target_index - 1]
        target_draw = draws[target_index]
        age = len(draws) - 1 - target_index
        recency_weight = math.exp(-age / 300)

        for number in range(1, 46):
            state = presence_state(
                older_draw,
                latest_draw,
                number
            )
            key = (number, state)
            state_trials[state] += recency_weight
            number_state_trials[key] += recency_weight

            if number in target_draw:
                state_hits[state] += recency_weight
                number_state_hits[key] += recency_weight

        # 최근 회차 쪽을 조금 더 중요하게 보는 순서 보존 Jaccard 유사도입니다.
        older_union = older_draw | current_older
        latest_union = latest_draw | current_latest
        ordered_similarity = (
            0.40
            * len(older_draw & current_older)
            / max(len(older_union), 1)
            + 0.60
            * len(latest_draw & current_latest)
            / max(len(latest_union), 1)
        )
        context_candidates.append((
            ordered_similarity,
            recency_weight,
            target_draw
        ))

    # 유사하지 않은 모든 과거 회차를 섞으면 단순 빈도 신호와 중복됩니다.
    # 가장 가까운 60개 문맥만 사용하고, 유사도가 0이면 제외합니다.
    nearest_contexts = sorted(
        context_candidates,
        key=lambda item: item[0],
        reverse=True
    )[:60]

    for similarity, recency_weight, target_draw in nearest_contexts:
        similarity_weight = recency_weight * similarity ** 2

        if similarity_weight <= 0:
            continue

        context_total_weight += similarity_weight

        for number in target_draw:
            context_hits[number] += similarity_weight

    raw_scores = {}

    for number in range(1, 46):
        current_state = presence_state(
            current_older,
            current_latest,
            number
        )
        key = (number, current_state)

        # 전체 번호에서 학습한 상태 전이율을 먼저 안정화합니다.
        global_state_probability = (
            state_hits[current_state]
            + 45.0 * base_probability
        ) / (
            state_trials[current_state] + 45.0
        )

        # 번호별 상태 표본이 적으면 위의 전체 상태 전이율로 수축합니다.
        state_probability = (
            number_state_hits[key]
            + number_prior_strength * global_state_probability
        ) / (
            number_state_trials[key] + number_prior_strength
        )

        # 유사 문맥 결과도 상태 확률을 사전값으로 사용해 과신을 막습니다.
        context_probability = (
            context_hits[number]
            + context_prior_strength * state_probability
        ) / (
            context_total_weight + context_prior_strength
        )
        raw_scores[number] = (
            state_probability * 0.80
            + context_probability * 0.20
        )

    # 확률 차이가 작다는 사실 자체가 중요한 신호입니다. min-max를 적용하지
    # 않고 6/45 대비 상대 편차만 중립 점수 0.5에 약하게 반영합니다.
    reliability = 0.25

    return {
        number: min(
            0.65,
            max(
                0.35,
                0.5
                + reliability
                * (raw_scores[number] - base_probability)
                / base_probability
            )
        )
        for number in range(1, 46)
    }

ANALYSIS_METHODS = {
    "전체빈도": lambda df: _score_frequency(df),
    "최근30회빈도": lambda df: _score_frequency(df, recent_count=30),
    "최근100회빈도": lambda df: _score_frequency(df, recent_count=100),
    "지수가중빈도": _score_exponential_frequency,
    "보너스보조신호": _score_bonus_support,
    "최근추세": _score_recent_trend,
    "출현간격주기": _score_gap_cycle,
    "직전회차전이": _score_draw_transition,
    "동반출현연결": _score_pair_network,
    "핫콜드밸런스": _score_hot_cold_balance,
    "마코프전이": _score_markov_transition,
    "끝수다이내믹스": _score_last_digit,
    "최근2회문맥": _score_recent_two_draw_context
}

ANALYSIS_METHOD_PRIORS = {
    method_name: (
        0.55
        if method_name == "보너스보조신호"
        else 0.30
        if method_name == "최근2회문맥"
        else 1.0
    )
    for method_name in ANALYSIS_METHODS
}

def calculate_analysis_scores(df):
    """활성화된 모든 분석기법의 번호별 점수를 계산합니다."""

    return {
        method_name: method_function(df)
        for method_name, method_function in ANALYSIS_METHODS.items()
    }

def backtest_analysis_methods(
        df,
        backtest_draws=BACKTEST_DRAWS,
        min_train_draws=MIN_BACKTEST_TRAIN_DRAWS
):
    """
    각 목표 회차 이전 데이터만 사용해 최근 회차를 순차 백테스트합니다.
    전체 검증 성능에 완료된 최근 2회 성능을 소폭 더 반영합니다.
    최근 표본은 18회 사전표본으로 수축하고, 기준 성능 1.0 이하인 기법에는
    탐색 가중치만 남겨 약한 신호가 유효한 신호를 희석하지 않게 합니다.
    """

    method_names = list(ANALYSIS_METHODS)
    metrics = {
        method_name: {
            "top6_hits": 0.0,
            "top12_hits": 0.0,
            "rank_quality": 0.0,
            "recent_top6_hits": 0.0,
            "recent_top12_hits": 0.0,
            "recent_rank_quality": 0.0
        }
        for method_name in method_names
    }

    start_index = max(
        min_train_draws,
        len(df) - backtest_draws
    )
    tested_draws = max(0, len(df) - start_index)
    recent_review_draws = min(
        RECENT_REVIEW_DRAWS,
        tested_draws
    )

    if tested_draws == 0:
        prior_total = sum(
            ANALYSIS_METHOD_PRIORS[method_name]
            for method_name in method_names
        )

        return pd.DataFrame([
            {
                "분석기법": method_name,
                "백테스트회차수": 0,
                "TOP6평균적중": 0.0,
                "TOP12평균적중": 0.0,
                "평균순위점수": 0.0,
                "성능지수": 1.0,
                "최근보강회차수": 0,
                "최근2회성능지수": 1.0,
                "장기성능지수": 1.0,
                "보강성능지수": 1.0,
                "초과성능": 0.0,
                "사전가중치": ANALYSIS_METHOD_PRIORS[method_name],
                "가중치": (
                    ANALYSIS_METHOD_PRIORS[method_name] / prior_total
                )
            }
            for method_name in method_names
        ])

    for target_index in range(start_index, len(df)):
        training_df = df.iloc[:target_index]
        is_recent_review = (
            target_index >= len(df) - recent_review_draws
        )
        actual_numbers = set(
            int(df.iloc[target_index][column])
            for column in NUMBER_COLUMNS
        )
        score_maps = calculate_analysis_scores(training_df)

        for method_name, score_map in score_maps.items():
            ranked_numbers = sorted(
                range(1, 46),
                key=lambda number: (-score_map[number], number)
            )
            rank_map = {
                number: rank
                for rank, number in enumerate(ranked_numbers, start=1)
            }
            top6_hits = len(
                actual_numbers & set(ranked_numbers[:6])
            )
            top12_hits = len(
                actual_numbers & set(ranked_numbers[:12])
            )
            rank_quality = sum(
                (46 - rank_map[number]) / 45
                for number in actual_numbers
            ) / 6

            metrics[method_name]["top6_hits"] += top6_hits
            metrics[method_name]["top12_hits"] += top12_hits
            metrics[method_name]["rank_quality"] += rank_quality

            if is_recent_review:
                metrics[method_name]["recent_top6_hits"] += top6_hits
                metrics[method_name]["recent_top12_hits"] += top12_hits
                metrics[method_name]["recent_rank_quality"] += rank_quality

    expected_top6_hits = 6 * 6 / 45
    expected_top12_hits = 12 * 6 / 45
    expected_rank_quality = 23 / 45
    records = []

    for method_name in method_names:
        average_top6 = (
            metrics[method_name]["top6_hits"] / tested_draws
        )
        average_top12 = (
            metrics[method_name]["top12_hits"] / tested_draws
        )
        average_rank_quality = (
            metrics[method_name]["rank_quality"] / tested_draws
        )

        raw_performance = (
            0.50 * average_top6 / expected_top6_hits
            + 0.30 * average_top12 / expected_top12_hits
            + 0.20 * average_rank_quality / expected_rank_quality
        )

        recent_average_top6 = (
            metrics[method_name]["recent_top6_hits"]
            / recent_review_draws
        )
        recent_average_top12 = (
            metrics[method_name]["recent_top12_hits"]
            / recent_review_draws
        )
        recent_average_rank_quality = (
            metrics[method_name]["recent_rank_quality"]
            / recent_review_draws
        )
        recent_performance = (
            0.50 * recent_average_top6 / expected_top6_hits
            + 0.30 * recent_average_top12 / expected_top12_hits
            + 0.20
            * recent_average_rank_quality
            / expected_rank_quality
        )

        # 전체 120회에 포함된 최근 두 회차를 먼저 빼서 이중계수를 피합니다.
        long_term_draws = tested_draws - recent_review_draws

        if long_term_draws > 0:
            long_average_top6 = (
                metrics[method_name]["top6_hits"]
                - metrics[method_name]["recent_top6_hits"]
            ) / long_term_draws
            long_average_top12 = (
                metrics[method_name]["top12_hits"]
                - metrics[method_name]["recent_top12_hits"]
            ) / long_term_draws
            long_average_rank_quality = (
                metrics[method_name]["rank_quality"]
                - metrics[method_name]["recent_rank_quality"]
            ) / long_term_draws
            long_term_performance = (
                0.50 * long_average_top6 / expected_top6_hits
                + 0.30 * long_average_top12 / expected_top12_hits
                + 0.20
                * long_average_rank_quality
                / expected_rank_quality
            )
        else:
            long_term_performance = raw_performance

        # 2 / (2 + 18) = 10%만 최근 두 회차 성과에 배정합니다.
        recent_reliability = recent_review_draws / (
            recent_review_draws + RECENT_REVIEW_PRIOR_DRAWS
        )
        reinforced_performance = (
            long_term_performance * (1.0 - recent_reliability)
            + recent_performance * recent_reliability
        )

        # 기준(1.0)을 넘긴 검증 성과만 주 가중치로 사용합니다. 기준 이하인
        # 기법에도 10% 탐색 몫은 남겨 한 구간의 우연으로 완전히 제거되지
        # 않게 합니다. 3개 120회 구간에서 비교한 후보식 중 전체 손실이 가장
        # 작았던 보수적 게이트이며, 통계적 우위가 확정됐다는 뜻은 아닙니다.
        excess_skill = max(0.0, reinforced_performance - 1.0)
        gated_performance = 0.10 + 3.0 * excess_skill

        records.append({
            "분석기법": method_name,
            "백테스트회차수": tested_draws,
            "TOP6평균적중": round(average_top6, 4),
            "TOP12평균적중": round(average_top12, 4),
            "평균순위점수": round(average_rank_quality, 4),
            "성능지수": round(raw_performance, 4),
            "최근보강회차수": recent_review_draws,
            "최근2회성능지수": round(recent_performance, 4),
            "장기성능지수": round(long_term_performance, 4),
            "보강성능지수": round(reinforced_performance, 4),
            "초과성능": round(excess_skill, 4),
            "사전가중치": ANALYSIS_METHOD_PRIORS[method_name],
            "조정성능": (
                gated_performance * ANALYSIS_METHOD_PRIORS[method_name]
            )
        })

    total_performance = sum(
        record["조정성능"]
        for record in records
    )

    for record in records:
        record["가중치"] = record.pop("조정성능") / total_performance

    return pd.DataFrame(records)

def calculate_ensemble_scores(df, method_performance_df):
    """
    백테스트 성능 가중치로 분석기법별 점수를 결합합니다.

    개별 기법에서 이미 공통 범위로 만든 점수를 다시 min-max 처리하면 작은
    앙상블 차이가 0.05~1.00으로 과장됩니다. 가중평균을 그대로 유지해 특정
    고득점 번호가 모든 세트에 과도하게 반복되는 현상을 줄입니다.
    """

    score_maps = calculate_analysis_scores(df)
    weight_map = dict(zip(
        method_performance_df["분석기법"],
        method_performance_df["가중치"]
    ))
    method_weights = {
        method_name: max(
            0.0,
            float(weight_map.get(method_name, 0.0))
        )
        for method_name in score_maps
    }
    total_weight = sum(method_weights.values())

    if total_weight <= 0:
        method_weights = {
            method_name: 1.0 / len(score_maps)
            for method_name in score_maps
        }
    else:
        method_weights = {
            method_name: weight / total_weight
            for method_name, weight in method_weights.items()
        }

    ensemble_scores = {}

    for number in range(1, 46):
        ensemble_scores[number] = sum(
            method_weights[method_name]
            * score_map[number]
            for method_name, score_map in score_maps.items()
        )

    return {
        number: min(1.0, max(0.05, score))
        for number, score in ensemble_scores.items()
    }

def _calibrate_ensemble_scores(
        ensemble_scores,
        prediction_feedback=None
):
    """실전 Brier 피드백으로 점수의 과도한 분산만 완만하게 보정합니다."""

    feedback = prediction_feedback or {}
    spread_multiplier = float(
        feedback.get("score_spread_multiplier", 1.0)
    )
    spread_multiplier = min(1.05, max(0.95, spread_multiplier))
    center = sum(ensemble_scores.values()) / len(ensemble_scores)

    return {
        number: min(
            1.0,
            max(
                0.05,
                center
                + spread_multiplier
                * (score - center)
            )
        )
        for number, score in ensemble_scores.items()
    }

def _calculate_pair_affinity(df, lookback=500):
    """조합 안에서 번호 쌍의 과거 동반 출현 강도를 계산합니다."""

    pair_counts = Counter()
    draws = _get_draw_sets(df.tail(lookback))

    for age, draw in enumerate(reversed(draws)):
        weight = math.exp(-age / 300)

        for pair in combinations(sorted(draw), 2):
            pair_counts[pair] += weight

    maximum = max(pair_counts.values()) if pair_counts else 1.0

    return {
        pair: 0.05 + 0.95 * count / maximum
        for pair, count in pair_counts.items()
    }

def _calculate_bonus_pair_affinity(df, lookback=500):
    """과거 본번호-보너스 번호 관계를 약한 조합 보조 신호로 계산합니다."""

    pair_counts = Counter()
    sample = df.tail(lookback).reset_index(drop=True)

    for age, row in enumerate(reversed(list(sample.iterrows()))):
        _, draw = row
        weight = math.exp(-age / 300)
        bonus_number = int(draw["보너스"])

        for column in NUMBER_COLUMNS:
            main_number = int(draw[column])
            pair = tuple(sorted((main_number, bonus_number)))
            pair_counts[pair] += weight

    maximum = max(pair_counts.values()) if pair_counts else 1.0

    return {
        pair: 0.05 + 0.95 * count / maximum
        for pair, count in pair_counts.items()
    }

def _mean_and_standard_deviation(values):
    average = sum(values) / len(values)
    variance = sum(
        (value - average) ** 2
        for value in values
    ) / len(values)

    return average, max(math.sqrt(variance), 0.5)

def _consecutive_pair_count(numbers):
    ordered = sorted(numbers)

    return sum(
        ordered[index + 1] - ordered[index] == 1
        for index in range(len(ordered) - 1)
    )

def _relative_empirical_probabilities(values, possible_values):
    """Laplace 보정한 범주별 경험확률을 최빈 범주 대비 0~1로 변환합니다."""

    counts = Counter(values)
    possible_values = list(possible_values)
    denominator = len(values) + len(possible_values)
    probabilities = {
        value: (counts.get(value, 0) + 1) / denominator
        for value in possible_values
    }
    maximum = max(probabilities.values()) if probabilities else 1.0

    return {
        value: probability / maximum
        for value, probability in probabilities.items()
    }

def _learn_combination_structure(df):
    """고정 숫자 대신 과거 조합 분포의 중앙 90% 수준을 학습합니다."""

    draws = [sorted(draw) for draw in _get_draw_sets(df)]
    sums = [sum(draw) for draw in draws]
    odd_counts = [
        sum(number % 2 == 1 for number in draw)
        for draw in draws
    ]
    high_counts = [
        sum(number >= 23 for number in draw)
        for draw in draws
    ]
    spans = [draw[-1] - draw[0] for draw in draws]
    consecutive_counts = [
        _consecutive_pair_count(draw)
        for draw in draws
    ]
    adjacent_overlaps = [
        len(set(draws[index - 1]) & set(draws[index]))
        for index in range(1, len(draws))
    ] or [2]

    sum_series = pd.Series(sums)
    odd_series = pd.Series(odd_counts)
    high_series = pd.Series(high_counts)
    span_series = pd.Series(spans)
    consecutive_series = pd.Series(consecutive_counts)
    overlap_series = pd.Series(adjacent_overlaps)

    sum_mean, sum_std = _mean_and_standard_deviation(sums)
    odd_mean, odd_std = _mean_and_standard_deviation(odd_counts)
    high_mean, high_std = _mean_and_standard_deviation(high_counts)
    span_mean, span_std = _mean_and_standard_deviation(spans)
    consecutive_mean, consecutive_std = _mean_and_standard_deviation(
        consecutive_counts
    )

    return {
        "sum_low": int(math.floor(sum_series.quantile(0.05))),
        "sum_high": int(math.ceil(sum_series.quantile(0.95))),
        "odd_low": max(1, int(math.floor(odd_series.quantile(0.03)))),
        "odd_high": min(5, int(math.ceil(odd_series.quantile(0.97)))),
        "high_low": max(1, int(math.floor(high_series.quantile(0.03)))),
        "high_high": min(5, int(math.ceil(high_series.quantile(0.97)))),
        "span_low": int(math.floor(span_series.quantile(0.03))),
        "span_high": int(math.ceil(span_series.quantile(0.97))),
        "consecutive_high": max(
            1,
            int(math.ceil(consecutive_series.quantile(0.95)))
        ),
        "overlap_high": max(
            2,
            min(3, int(math.ceil(overlap_series.quantile(0.97))))
        ),
        "averages": {
            "sum": (sum_mean, sum_std),
            "odd": (odd_mean, odd_std),
            "high": (high_mean, high_std),
            "span": (span_mean, span_std),
            "consecutive": (consecutive_mean, consecutive_std)
        },
        "empirical": {
            "odd": _relative_empirical_probabilities(
                odd_counts,
                range(0, 7)
            ),
            "high": _relative_empirical_probabilities(
                high_counts,
                range(0, 7)
            ),
            "consecutive": _relative_empirical_probabilities(
                consecutive_counts,
                range(0, 6)
            )
        }
    }

def _is_balanced_combination(numbers, structure, latest_draw):
    total_sum = sum(numbers)
    odd_count = sum(number % 2 == 1 for number in numbers)
    high_count = sum(number >= 23 for number in numbers)
    span = max(numbers) - min(numbers)
    consecutive_count = _consecutive_pair_count(numbers)
    overlap_count = len(set(numbers) & latest_draw)

    zone_counts = [
        sum(low <= number <= high for number in numbers)
        for low, high in [
            (1, 10),
            (11, 20),
            (21, 30),
            (31, 40),
            (41, 45)
        ]
    ]

    return (
        structure["sum_low"] <= total_sum <= structure["sum_high"]
        and structure["odd_low"] <= odd_count <= structure["odd_high"]
        and structure["high_low"] <= high_count <= structure["high_high"]
        and structure["span_low"] <= span <= structure["span_high"]
        and consecutive_count <= structure["consecutive_high"]
        and overlap_count <= structure["overlap_high"]
        and max(zone_counts) <= 4
    )

def _combination_balance_score(numbers, structure):
    total_sum = sum(numbers)
    odd_count = sum(number % 2 == 1 for number in numbers)
    high_count = sum(number >= 23 for number in numbers)
    span = max(numbers) - min(numbers)
    consecutive_count = _consecutive_pair_count(numbers)

    sum_mean, sum_std = structure["averages"]["sum"]
    span_mean, span_std = structure["averages"]["span"]
    sum_z = (total_sum - sum_mean) / sum_std
    span_z = (span - span_mean) / span_std

    sum_score = math.exp(-0.5 * (sum_z ** 2))
    span_score = math.exp(-0.5 * (span_z ** 2))
    odd_score = structure["empirical"]["odd"].get(odd_count, 0.0)
    high_score = structure["empirical"]["high"].get(high_count, 0.0)
    consecutive_score = structure["empirical"]["consecutive"].get(
        consecutive_count,
        0.0
    )

    return (
        sum_score * 0.30
        + span_score * 0.20
        + odd_score * 0.20
        + high_score * 0.20
        + consecutive_score * 0.10
    )

def _weighted_sample_without_replacement(
        random_generator,
        numbers,
        weights,
        sample_size
):
    available_numbers = list(numbers)
    available_weights = list(weights)
    selected = []

    while len(selected) < sample_size:
        total_weight = sum(available_weights)
        point = random_generator.random() * total_weight
        cumulative = 0.0
        selected_index = len(available_numbers) - 1

        for index, weight in enumerate(available_weights):
            cumulative += weight

            if point <= cumulative:
                selected_index = index
                break

        selected.append(available_numbers.pop(selected_index))
        available_weights.pop(selected_index)

    return tuple(sorted(selected))

def generate_prediction_sets(
        df,
        set_count=PREDICTION_SET_COUNT,
        random_seed=None,
        target_draw=None,
        method_performance_df=None,
        prediction_feedback=None,
        llm_audit=None
):
    """
    다중 분석기법의 성능 가중 앙상블로 후보 조합을 만들고,
    과거 조합 분포와 세트 간 다양성을 반영해 최종 세트를 선택합니다.
    """

    if df.empty:
        return pd.DataFrame(columns=["세트"] + NUMBER_COLUMNS + ["조합점수"])

    target_draw = target_draw or int(df["회차"].max()) + 1

    if method_performance_df is None:
        method_performance_df = backtest_analysis_methods(df)

    ensemble_scores = calculate_ensemble_scores(
        df,
        method_performance_df
    )
    ensemble_scores = _calibrate_ensemble_scores(
        ensemble_scores,
        prediction_feedback
    )
    statistical_diversity_multiplier = float(
        (prediction_feedback or {}).get(
            "diversity_penalty_multiplier",
            1.0
        )
    )
    llm_diversity_multiplier = float(
        (llm_audit or {}).get(
            "effective_diversity_multiplier",
            1.0
        )
    )
    diversity_penalty_multiplier = min(
        1.20,
        max(
            0.90,
            statistical_diversity_multiplier
            * llm_diversity_multiplier
        )
    )
    pair_affinity = _calculate_pair_affinity(df)
    bonus_pair_affinity = _calculate_bonus_pair_affinity(df)
    structure = _learn_combination_structure(df)
    latest_draw = _get_draw_sets(df.tail(1))[0]

    seed = (
        int(random_seed)
        if random_seed is not None
        else target_draw * 10007 + len(df)
    )
    random_generator = random.Random(seed)
    numbers = list(range(1, 46))

    # 모든 번호에 탐색 확률을 남겨 특정 번호가 완전히 배제되지 않게 합니다.
    sampling_weights = [
        0.30 + 1.70 * ensemble_scores[number]
        for number in numbers
    ]

    candidate_scores = {}
    candidate_attempts = max(
        PREDICTION_CANDIDATE_ATTEMPTS,
        set_count * 800
    )

    for _ in range(candidate_attempts):
        candidate = _weighted_sample_without_replacement(
            random_generator,
            numbers,
            sampling_weights,
            6
        )

        if not _is_balanced_combination(
            candidate,
            structure,
            latest_draw
        ):
            continue

        number_score = sum(
            ensemble_scores[number]
            for number in candidate
        ) / 6
        main_pair_score = sum(
            pair_affinity.get(pair, 0.05)
            for pair in combinations(candidate, 2)
        ) / 15
        bonus_relation_score = sum(
            bonus_pair_affinity.get(pair, 0.05)
            for pair in combinations(candidate, 2)
        ) / 15
        pair_score = (
            main_pair_score * 0.85
            + bonus_relation_score * 0.15
        )
        balance_score = _combination_balance_score(
            candidate,
            structure
        )

        total_score = (
            number_score * 0.68
            + pair_score * 0.20
            + balance_score * 0.12
        )

        candidate_scores[candidate] = max(
            total_score,
            candidate_scores.get(candidate, 0.0)
        )

    # 학습 범위가 지나치게 좁거나 데이터가 적어 후보가 부족한 경우의 안전장치
    fallback_attempts = 0

    while len(candidate_scores) < set_count and fallback_attempts < 5000:
        fallback_attempts += 1
        candidate = tuple(sorted(random_generator.sample(numbers, 6)))

        if candidate in candidate_scores:
            continue

        number_score = sum(
            ensemble_scores[number]
            for number in candidate
        ) / 6
        main_pair_score = sum(
            pair_affinity.get(pair, 0.05)
            for pair in combinations(candidate, 2)
        ) / 15
        bonus_relation_score = sum(
            bonus_pair_affinity.get(pair, 0.05)
            for pair in combinations(candidate, 2)
        ) / 15
        pair_score = (
            main_pair_score * 0.85
            + bonus_relation_score * 0.15
        )
        balance_score = _combination_balance_score(
            candidate,
            structure
        )
        candidate_scores[candidate] = (
            number_score * 0.68
            + pair_score * 0.20
            + balance_score * 0.12
        )

    selected_candidates = []
    number_usage = Counter()
    remaining = dict(candidate_scores)

    while len(selected_candidates) < set_count and remaining:
        def diversity_adjusted_score(item):
            candidate, base_score = item
            candidate_set = set(candidate)
            maximum_overlap = max(
                (
                    len(candidate_set & set(selected))
                    for selected, _ in selected_candidates
                ),
                default=0
            )
            usage_penalty = sum(
                number_usage[number]
                for number in candidate
            )

            return (
                base_score
                - maximum_overlap
                * 0.035
                * diversity_penalty_multiplier
                - usage_penalty
                * 0.012
                * diversity_penalty_multiplier
            )

        # lotto.py의 장점을 반영해 두 세트가 5개 이상 겹치지 않게 합니다.
        diverse_items = [
            item
            for item in remaining.items()
            if all(
                len(set(item[0]) & set(selected)) <= 4
                for selected, _ in selected_candidates
            )
        ]
        candidate_pool = diverse_items or list(remaining.items())

        best_candidate, best_score = max(
            candidate_pool,
            key=diversity_adjusted_score
        )
        selected_candidates.append((best_candidate, best_score))
        number_usage.update(best_candidate)
        del remaining[best_candidate]

    rows = []

    for set_number, (candidate, score) in enumerate(
        selected_candidates,
        start=1
    ):
        row = {
            "세트": set_number,
            "조합점수": round(score * 100, 2)
        }

        for column, number in zip(NUMBER_COLUMNS, candidate):
            row[column] = number

        rows.append(row)

    return pd.DataFrame(rows)[
        ["세트"] + NUMBER_COLUMNS + ["조합점수"]
    ]
