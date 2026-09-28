# ============================================================
# 로또 6/45 고급 분석 & 예측 코어 엔진 (lotto_engine.py)
# ============================================================

import os
import math
import sys
import requests
import pandas as pd
import numpy as np
import time
import random
from collections import Counter
from datetime import datetime
from itertools import combinations

CSV_FILE = "과거로또 당첨번호.csv"
PREDICTION_FILE = "당첨예상번호.csv"

NUMBER_COLUMNS = ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"]
ALL_COLUMNS = ["회차", "날짜"] + NUMBER_COLUMNS + ["보너스"]

API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
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


# ============================================================
# 1. 공통 유틸리티 및 데이터 수집
# ============================================================

def get_lotto_ball_color(number):
    """로또 공식 번호대별 색상 반환 (1~10 노랑, 11~20 파랑, 21~30 빨강, 31~40 회색, 41~45 초록)"""
    if 1 <= number <= 10:
        return "#FBC400", "#000000"  # bg, text
    elif 11 <= number <= 20:
        return "#69C8F2", "#FFFFFF"
    elif 21 <= number <= 30:
        return "#FF7272", "#FFFFFF"
    elif 31 <= number <= 40:
        return "#AAAAAA", "#FFFFFF"
    elif 41 <= number <= 45:
        return "#B0D840", "#FFFFFF"
    return "#888888", "#FFFFFF"


def calculate_ac_value(numbers):
    """
    산술적 복잡도(Arithmetic Complexity, AC값) 계산
    6개 번호의 모든 차이값 조합의 고유 개수 - 5
    AC값 범위: 0 ~ 10 (실제 당첨번호의 90% 이상은 7 ~ 10)
    """
    ordered = sorted(numbers)
    diffs = set()
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            diffs.add(ordered[j] - ordered[i])
    return len(diffs) - (len(numbers) - 1)


def get_lotto_draw_from_api(draw_no, max_retries=3, retry_delay=1.0):
    """동행복권 API에서 특정 회차 데이터 조회"""
    params = {"srchLtEpsd": draw_no}
    for attempt in range(1, max_retries + 1):
        try:
            response = HTTP_SESSION.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "data" not in data:
                return None
            result_list = data["data"].get("list", [])
            if not result_list:
                return None
            item = result_list[0]
            returned_draw = int(item.get("ltEpsd", 0))
            if returned_draw != draw_no:
                return None
            return {
                "회차": returned_draw,
                "날짜": item.get("ltRflYmd"),
                "번호1": int(item.get("tm1WnNo")),
                "번호2": int(item.get("tm2WnNo")),
                "번호3": int(item.get("tm3WnNo")),
                "번호4": int(item.get("tm4WnNo")),
                "번호5": int(item.get("tm5WnNo")),
                "번호6": int(item.get("tm6WnNo")),
                "보너스": int(item.get("bnsWnNo")),
                "1등당첨금액": int(item.get("rnk1WnAmt", 0) or 0),
                "1등당첨자수": int(item.get("rnk1WnPsnCnt", 0) or 0),
                "총판매금액": int(item.get("wholSlAmt", 0) or 0)
            }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None
    return None


def get_latest_draw_no():
    """현재 기준 최신 회차 추정 및 조회"""
    base_date = datetime(2002, 12, 7)
    now = datetime.now()
    weeks = int((now - base_date).days / 7) + 1
    
    # 역순 탐색
    for draw in range(weeks + 2, max(1, weeks - 10), -1):
        info = get_lotto_draw_from_api(draw, max_retries=1)
        if info:
            return draw
    return weeks


def load_history_csv(filename=CSV_FILE):
    """과거 당첨번호 CSV 불러오기 및 전처리"""
    if not os.path.exists(filename):
        return pd.DataFrame(columns=ALL_COLUMNS)
    
    try:
        df = pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding="cp949")
        
    for col in NUMBER_COLUMNS + ["회차", "보너스"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.dropna(subset=NUMBER_COLUMNS + ["회차", "보너스"]).copy()
    df["회차"] = df["회차"].astype(int)
    for col in NUMBER_COLUMNS + ["보너스"]:
        df[col] = df[col].astype(int)
        
    df = df.sort_values("회차").reset_index(drop=True)
    return df


def save_history_csv(df, filename=CSV_FILE):
    """과거 당첨번호 CSV 저장"""
    df = df.sort_values("회차").drop_duplicates(subset=["회차"]).reset_index(drop=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    return df


def sync_lotto_history(progress_callback=None):
    """동행복권 API와 로컬 CSV 동기화 (최신 및 누락 회차 수집)"""
    df = load_history_csv()
    existing_draws = set(df["회차"].tolist()) if not df.empty else set()
    latest_draw = get_latest_draw_no()
    
    missing_draws = [d for d in range(1, latest_draw + 1) if d not in existing_draws]
    
    if not missing_draws:
        if progress_callback:
            progress_callback(100, f"최신 회차({latest_draw}회)까지 이미 동기화되어 있습니다.", df)
        return df, 0
        
    new_rows = []
    total = len(missing_draws)
    
    for i, draw_no in enumerate(missing_draws, 1):
        if progress_callback:
            progress_callback(int((i / total) * 95), f"{draw_no}회차 데이터 수집 중... ({i}/{total})", None)
        item = get_lotto_draw_from_api(draw_no)
        if item:
            new_rows.append(item)
        time.sleep(0.05)
        
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cols_to_keep = [c for c in ALL_COLUMNS if c in new_df.columns]
        new_df = new_df[cols_to_keep]
        combined_df = pd.concat([df, new_df], ignore_index=True)
        df = save_history_csv(combined_df)
        
    if progress_callback:
        progress_callback(100, f"동기화 완료! 총 {len(new_rows)}개 신규 회차가 추가되었습니다.", df)
        
    return df, len(new_rows)


# ============================================================
# 2. 파생 통계 변수 생성
# ============================================================

def create_rich_features(df):
    """분석 및 시각화용 풍부한 파생 변수 생성"""
    if df.empty:
        return df
        
    df = df.copy()
    
    # 합계
    df["총합"] = df[NUMBER_COLUMNS].sum(axis=1)
    
    # 홀수 / 짝수 개수
    df["홀수개수"] = df[NUMBER_COLUMNS].apply(lambda row: sum(n % 2 == 1 for n in row), axis=1)
    df["짝수개수"] = 6 - df["홀수개수"]
    df["홀짝비"] = df["홀수개수"].astype(str) + ":" + df["짝수개수"].astype(str)
    
    # 고저 개수 (고: 23~45, 저: 1~22)
    df["고번호개수"] = df[NUMBER_COLUMNS].apply(lambda row: sum(n >= 23 for n in row), axis=1)
    df["저번호개수"] = 6 - df["고번호개수"]
    df["고저비"] = df["고번호개수"].astype(str) + ":" + df["저번호개수"].astype(str)
    
    # 번호 범위 (Span)
    df["범위"] = df[NUMBER_COLUMNS].max(axis=1) - df[NUMBER_COLUMNS].min(axis=1)
    
    # 연속 번호 쌍 개수
    def count_consec(row):
        s = sorted(row)
        return sum(s[i+1] - s[i] == 1 for i in range(len(s) - 1))
    df["연속번호개수"] = df[NUMBER_COLUMNS].apply(count_consec, axis=1)
    
    # AC값 (산술적 복잡도)
    df["AC값"] = df[NUMBER_COLUMNS].apply(calculate_ac_value, axis=1)
    
    # 끝수(1의 자리) 합계 및 고유 개수
    df["끝수합"] = df[NUMBER_COLUMNS].apply(lambda row: sum(n % 10 for n in row), axis=1)
    df["고유끝수개수"] = df[NUMBER_COLUMNS].apply(lambda row: len(set(n % 10 for n in row)), axis=1)
    
    # 이전 회차 이월수 개수
    draw_sets = [set(row) for row in df[NUMBER_COLUMNS].values]
    overlaps = [0]
    for i in range(1, len(draw_sets)):
        overlaps.append(len(draw_sets[i] & draw_sets[i-1]))
    df["이월수개수"] = overlaps
    
    return df


# ============================================================
# 3. 번호별 12종 고도화 분석 기법 스코어링
# ============================================================

def _get_draw_sets(df):
    return [set(int(n) for n in row) for row in df[NUMBER_COLUMNS].values]

def _normalize_score_map(raw_scores):
    cleaned = {}
    for n in range(1, 46):
        v = float(raw_scores.get(n, 0.0))
        cleaned[n] = v if math.isfinite(v) else 0.0
    mn = min(cleaned.values())
    mx = max(cleaned.values())
    if math.isclose(mn, mx):
        return {n: 0.5 for n in range(1, 46)}
    rng = mx - mn
    return {n: 0.05 + 0.95 * (cleaned[n] - mn) / rng for n in range(1, 46)}

# 기법 1: 전체 출현 빈도
def _score_frequency(df, recent_count=None):
    sample = df if recent_count is None else df.tail(recent_count)
    counts = Counter(int(n) for n in sample[NUMBER_COLUMNS].values.flatten())
    return _normalize_score_map({n: counts.get(n, 0) for n in range(1, 46)})

# 기법 2: 지수 가중 빈도 (시간 감쇠)
def _score_exponential_frequency(df, half_life=35):
    draws = _get_draw_sets(df)
    decay = math.log(2) / max(half_life, 1)
    raw = {n: 0.0 for n in range(1, 46)}
    for age, draw in enumerate(reversed(draws)):
        w = math.exp(-decay * age)
        for n in draw:
            raw[n] += w
    return _normalize_score_map(raw)

# 기법 3: 보너스 번호 보조 신호
def _score_bonus_support(df, half_life=120):
    bonus_counts = Counter(int(n) for n in df["보너스"].values.flatten())
    bonus_decay = {n: 0.0 for n in range(1, 46)}
    decay = math.log(2) / max(half_life, 1)
    for age, n in enumerate(reversed(df["보너스"].astype(int).tolist())):
        bonus_decay[n] += math.exp(-decay * age)
    freq = _normalize_score_map(bonus_counts)
    dec = _normalize_score_map(bonus_decay)
    return _normalize_score_map({n: freq[n] * 0.5 + dec[n] * 0.5 for n in range(1, 46)})

# 기법 4: 최근 추세 모멘텀 (최근 20회 vs 직전 80회 변화율)
def _score_recent_trend(df, recent_count=20, comparison_count=80):
    total = len(df)
    if total <= recent_count:
        return _score_frequency(df)
    recent = df.tail(recent_count)
    c_start = max(0, total - recent_count - comparison_count)
    c_end = total - recent_count
    comparison = df.iloc[c_start:c_end]
    if comparison.empty:
        return _score_frequency(df, recent_count=recent_count)
    recent_counts = Counter(int(n) for n in recent[NUMBER_COLUMNS].values.flatten())
    comp_counts = Counter(int(n) for n in comparison[NUMBER_COLUMNS].values.flatten())
    raw = {}
    for n in range(1, 46):
        r_rate = recent_counts.get(n, 0) / len(recent)
        c_rate = comp_counts.get(n, 0) / len(comparison)
        raw[n] = r_rate - c_rate
    return _normalize_score_map(raw)

# 기법 5: 출현 간격 주기 및 미출현 초과도 (Gap Cycle)
def _score_gap_cycle(df):
    draws = _get_draw_sets(df)
    total = len(draws)
    raw = {}
    for n in range(1, 46):
        app = [idx for idx, draw in enumerate(draws) if n in draw]
        if len(app) < 2:
            raw[n] = 0.0
            continue
        intervals = [app[i] - app[i-1] for i in range(1, len(app))]
        avg_int = sum(intervals) / len(intervals)
        var = sum((iv - avg_int) ** 2 for iv in intervals) / len(intervals)
        std_dev = math.sqrt(var)
        cur_missing = total - 1 - app[-1]
        next_iv = cur_missing + 1
        proximity = math.exp(-abs(next_iv - avg_int) / max(avg_int, 1.0))
        overdue = min(next_iv / max(avg_int, 1.0), 2.5) / 2.5
        consistency = 1.0 / (1.0 + std_dev / max(avg_int, 1.0))
        raw[n] = proximity * 0.45 + overdue * 0.40 + consistency * 0.15
    return _normalize_score_map(raw)

# 기법 6: 직전 회차 유사도 전이 (Draw Transition)
def _score_draw_transition(df, lookback=500):
    draws = _get_draw_sets(df)
    if len(draws) < 2:
        return {n: 0.5 for n in range(1, 46)}
    latest = draws[-1]
    raw = {n: 0.0 for n in range(1, 46)}
    start_idx = max(1, len(draws) - lookback)
    for idx in range(start_idx, len(draws)):
        sim = len(draws[idx - 1] & latest)
        if sim == 0:
            continue
        age = len(draws) - 1 - idx
        w = math.exp(-age / 250)
        for n in draws[idx]:
            raw[n] += sim * w
    return _normalize_score_map(raw)

# 기법 7: 동반 출현 네트워크 (Pair Co-occurrence Network)
def _score_pair_network(df, lookback=500, anchor_count=10):
    draws = _get_draw_sets(df.tail(lookback))
    recent_draws = draws[-30:]
    recent_counts = Counter(n for draw in recent_draws for n in draw)
    anchors = [n for n, _ in sorted(recent_counts.items(), key=lambda x: (-x[1], x[0]))[:anchor_count]]
    pair_counts = Counter()
    for age, draw in enumerate(reversed(draws)):
        w = math.exp(-age / 250)
        for pair in combinations(sorted(draw), 2):
            pair_counts[pair] += w
    raw = {}
    for n in range(1, 46):
        score = 0.0
        for anchor in anchors:
            if anchor == n:
                continue
            pair = tuple(sorted((n, anchor)))
            anchor_w = 1.0 + recent_counts.get(anchor, 0) / 10
            score += pair_counts.get(pair, 0.0) * anchor_w
        raw[n] = score
    return _normalize_score_map(raw)

# 기법 8: 핫/콜드/웜 (Hot/Cold/Warm) 3분할 밸런스 점수
def _score_hot_cold_balance(df, lookback=10):
    draws = _get_draw_sets(df.tail(lookback))
    counts = Counter(n for draw in draws for n in draw)
    raw = {}
    for n in range(1, 46):
        c = counts.get(n, 0)
        if c >= 2:
            raw[n] = 0.85 + 0.15 * min(c / 4, 1.0)
        elif c == 1:
            raw[n] = 0.65
        else:
            raw[n] = 0.40
    return _normalize_score_map(raw)

# 기법 9: 마코프 1차 전이 행렬 (Markov Chain Transition)
def _score_markov_transition(df, lookback=400):
    draws = _get_draw_sets(df.tail(lookback))
    if len(draws) < 2:
        return {n: 0.5 for n in range(1, 46)}
    latest = draws[-1]
    
    trans_matrix = np.zeros((46, 46), dtype=float)
    for i in range(1, len(draws)):
        prev_draw = draws[i-1]
        curr_draw = draws[i]
        for p in prev_draw:
            for c in curr_draw:
                trans_matrix[p, c] += 1.0
                
    trans_matrix += 0.5
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_prob = trans_matrix / row_sums
    
    raw = {n: 0.0 for n in range(1, 46)}
    for p in latest:
        for c in range(1, 46):
            raw[c] += trans_prob[p, c]
            
    return _normalize_score_map(raw)

# 기법 10: 끝수(1의 자리) 패턴
def _score_last_digit(df, lookback=100):
    sample = df.tail(lookback)
    draws = _get_draw_sets(sample)
    digit_counts = Counter()
    for draw in draws:
        for n in draw:
            digit_counts[n % 10] += 1
            
    recent_5_digits = [n % 10 for draw in draws[-5:] for n in draw]
    rec_digit_counts = Counter(recent_5_digits)
    
    raw = {}
    for n in range(1, 46):
        d = n % 10
        overall_d_rate = digit_counts.get(d, 0) / (len(draws) * 6)
        rec_d_rate = rec_digit_counts.get(d, 0) / 30
        raw[n] = overall_d_rate * 0.4 + rec_d_rate * 0.6
        
    return _normalize_score_map(raw)


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
}

ANALYSIS_METHOD_PRIORS = {
    "전체빈도": 0.90,
    "최근30회빈도": 1.00,
    "최근100회빈도": 1.00,
    "지수가중빈도": 1.10,
    "보너스보조신호": 0.55,
    "최근추세": 1.05,
    "출현간격주기": 1.15,
    "직전회차전이": 1.05,
    "동반출현연결": 1.10,
    "핫콜드밸런스": 1.15,
    "마코프전이": 1.10,
    "끝수다이내믹스": 0.95,
}


def calculate_all_analysis_scores(df):
    """12개 모든 분석기법의 번호별 점수 산출"""
    return {name: fn(df) for name, fn in ANALYSIS_METHODS.items()}


# ============================================================
# 4. 백테스팅 및 성과 가중치 산출
# ============================================================

def run_backtest(df, backtest_draws=120, min_train_draws=150, progress_callback=None):
    """
    Walk-forward 순차 백테스트
    각 기법의 Top6, Top12 적중률 및 순위 점수 측정 후 수축(Shrinkage) 가중치 부여
    """
    method_names = list(ANALYSIS_METHODS.keys())
    metrics = {m: {"top6_hits": 0.0, "top12_hits": 0.0, "rank_quality": 0.0} for m in method_names}
    
    start_index = max(min_train_draws, len(df) - backtest_draws)
    tested_draws = max(0, len(df) - start_index)
    
    if tested_draws == 0:
        prior_total = sum(ANALYSIS_METHOD_PRIORS[m] for m in method_names)
        res_df = pd.DataFrame([
            {
                "분석기법": m,
                "백테스트회차수": 0,
                "TOP6평균적중": 0.0,
                "TOP12평균적중": 0.0,
                "평균순위점수": 0.0,
                "성능지수": 1.0,
                "사전가중치": ANALYSIS_METHOD_PRIORS[m],
                "가중치": ANALYSIS_METHOD_PRIORS[m] / prior_total
            }
            for m in method_names
        ])
        return res_df

    total_steps = len(df) - start_index
    for step_idx, target_index in enumerate(range(start_index, len(df))):
        if progress_callback and step_idx % 5 == 0:
            progress_callback(int((step_idx / total_steps) * 100), f"백테스팅 진행 중... ({step_idx + 1}/{total_steps})")
            
        training_df = df.iloc[:target_index]
        actual_numbers = set(int(df.iloc[target_index][c]) for c in NUMBER_COLUMNS)
        
        score_maps = calculate_all_analysis_scores(training_df)
        
        for m, score_map in score_maps.items():
            ranked = sorted(range(1, 46), key=lambda n: (-score_map[n], n))
            rank_map = {n: rank for rank, n in enumerate(ranked, start=1)}
            
            metrics[m]["top6_hits"] += len(actual_numbers & set(ranked[:6]))
            metrics[m]["top12_hits"] += len(actual_numbers & set(ranked[:12]))
            metrics[m]["rank_quality"] += sum((46 - rank_map[n]) / 45 for n in actual_numbers) / 6

    expected_top6 = 6 * 6 / 45
    expected_top12 = 12 * 6 / 45
    expected_rank = 23 / 45
    records = []
    
    for m in method_names:
        avg_top6 = metrics[m]["top6_hits"] / tested_draws
        avg_top12 = metrics[m]["top12_hits"] / tested_draws
        avg_rank = metrics[m]["rank_quality"] / tested_draws
        
        raw_perf = (
            0.50 * avg_top6 / expected_top6 +
            0.30 * avg_top12 / expected_top12 +
            0.20 * avg_rank / expected_rank
        )
        
        shrunk = 1.0 + 0.25 * (raw_perf - 1.0)
        shrunk = min(1.25, max(0.75, shrunk))
        
        records.append({
            "분석기법": m,
            "백테스트회차수": tested_draws,
            "TOP6평균적중": round(avg_top6, 4),
            "TOP12평균적중": round(avg_top12, 4),
            "평균순위점수": round(avg_rank, 4),
            "성능지수": round(raw_perf, 4),
            "사전가중치": ANALYSIS_METHOD_PRIORS[m],
            "조정성능": shrunk * ANALYSIS_METHOD_PRIORS[m]
        })
        
    tot_perf = sum(r["조정성능"] for r in records)
    for r in records:
        r["가중치"] = round(r.pop("조정성능") / tot_perf, 4)
        
    perf_df = pd.DataFrame(records)
    
    if progress_callback:
        progress_callback(100, "백테스트 완료!")
        
    return perf_df


# ============================================================
# 5. 추천 전략별 스코어 계산 및 조합 생성기
# ============================================================

def get_strategy_scores(df, strategy_name="AI_ENSEMBLE", perf_df=None):
    """선택한 전략에 따른 번호별 최종 추천 점수 계산"""
    all_scores = calculate_all_analysis_scores(df)
    
    if strategy_name == "MOMENTUM":
        weights = {"핫콜드밸런스": 0.35, "최근추세": 0.25, "최근30회빈도": 0.20, "지수가중빈도": 0.20}
    elif strategy_name == "CYCLE_REVERSION":
        weights = {"출현간격주기": 0.45, "마코프전이": 0.25, "전체빈도": 0.15, "직전회차전이": 0.15}
    elif strategy_name == "GOLDILOCKS":
        weights = {"동반출현연결": 0.30, "지수가중빈도": 0.25, "끝수다이내믹스": 0.25, "전체빈도": 0.20}
    else:
        if perf_df is None or perf_df.empty:
            weights = {k: 1.0 / len(all_scores) for k in all_scores}
        else:
            weights = dict(zip(perf_df["분석기법"], perf_df["가중치"]))
            
    final_scores = {n: 0.0 for n in range(1, 46)}
    for m_name, score_map in all_scores.items():
        w = weights.get(m_name, 0.0)
        for n in range(1, 46):
            final_scores[n] += w * score_map[n]
            
    return _normalize_score_map(final_scores)


def _learn_advanced_combination_structure(df):
    """과거 당첨 데이터로부터 이상적인 조합의 통계적 허용 범위 학습"""
    rich_df = create_rich_features(df)
    
    sums = rich_df["총합"].tolist()
    odds = rich_df["홀수개수"].tolist()
    highs = rich_df["고번호개수"].tolist()
    spans = rich_df["범위"].tolist()
    consecs = rich_df["연속번호개수"].tolist()
    acs = rich_df["AC값"].tolist()
    unique_digits = rich_df["고유끝수개수"].tolist()
    
    return {
        "sum_min": int(np.percentile(sums, 3)),
        "sum_max": int(np.percentile(sums, 97)),
        "sum_mean": float(np.mean(sums)),
        "sum_std": float(np.std(sums)),
        "odd_min": int(np.percentile(odds, 2)),
        "odd_max": int(np.percentile(odds, 98)),
        "high_min": int(np.percentile(highs, 2)),
        "high_max": int(np.percentile(highs, 98)),
        "span_min": int(np.percentile(spans, 2)),
        "span_max": int(np.percentile(spans, 98)),
        "span_mean": float(np.mean(spans)),
        "span_std": float(np.std(spans)),
        "consec_max": int(np.percentile(consecs, 95)),
        "ac_min": 6,
        "ac_max": 10,
        "unique_digits_min": 4,
    }


def is_statistically_valid_combination(numbers, structure, latest_draw_set):
    """조합이 통계적 골디락스 영역 내에 있는지 검증"""
    s = sorted(numbers)
    tot = sum(s)
    if not (structure["sum_min"] <= tot <= structure["sum_max"]):
        return False
        
    odd_c = sum(n % 2 == 1 for n in s)
    if not (structure["odd_min"] <= odd_c <= structure["odd_max"]):
        return False
        
    high_c = sum(n >= 23 for n in s)
    if not (structure["high_min"] <= high_c <= structure["high_max"]):
        return False
        
    span = s[-1] - s[0]
    if not (structure["span_min"] <= span <= structure["span_max"]):
        return False
        
    consec_c = sum(s[i+1] - s[i] == 1 for i in range(len(s) - 1))
    if consec_c > structure["consec_max"]:
        return False
        
    ac = calculate_ac_value(s)
    if not (structure["ac_min"] <= ac <= structure["ac_max"]):
        return False
        
    u_digits = len(set(n % 10 for n in s))
    if u_digits < structure["unique_digits_min"]:
        return False
        
    overlap = len(set(s) & latest_draw_set)
    if overlap > 2:
        return False
        
    zones = [
        sum(1 <= n <= 10 for n in s),
        sum(11 <= n <= 20 for n in s),
        sum(21 <= n <= 30 for n in s),
        sum(31 <= n <= 40 for n in s),
        sum(41 <= n <= 45 for n in s)
    ]
    if max(zones) >= 4:
        return False
        
    return True


def calculate_combination_score(numbers, score_map, pair_affinity, structure):
    """개별 조합의 종합 점수 평가 (0.0 ~ 100.0)"""
    base_score = sum(score_map.get(n, 0.5) for n in numbers) / 6.0
    pairs = list(combinations(sorted(numbers), 2))
    pair_score = sum(pair_affinity.get(p, 0.1) for p in pairs) / len(pairs)
    
    tot = sum(numbers)
    span = max(numbers) - min(numbers)
    sum_z = (tot - structure["sum_mean"]) / structure["sum_std"]
    span_z = (span - structure["span_mean"]) / structure["span_std"]
    dist_fit = math.exp(-0.5 * (sum_z**2)) * 0.5 + math.exp(-0.5 * (span_z**2)) * 0.5
    
    ac = calculate_ac_value(numbers)
    ac_bonus = 1.0 if ac in (8, 9) else (0.85 if ac in (7, 10) else 0.6)
    
    total = (base_score * 0.40 + pair_score * 0.25 + dist_fit * 0.25 + ac_bonus * 0.10) * 100.0
    return round(total, 2)


def generate_smart_prediction_sets(
    df,
    set_count=10,
    strategy="AI_ENSEMBLE",
    pinned_numbers=None,
    excluded_numbers=None,
    perf_df=None,
    progress_callback=None
):
    """
    고정수(Pinned) / 제외수(Excluded) 반영 및 스마트 앙상블 조합 생성
    15,000개 후보 생성 후 랭킹 및 상위 세트 도출
    """
    pinned = set(pinned_numbers) if pinned_numbers else set()
    excluded = set(excluded_numbers) if excluded_numbers else set()
    excluded = excluded - pinned
    
    if len(pinned) > 5:
        pinned = set(list(pinned)[:5])
        
    score_map = get_strategy_scores(df, strategy_name=strategy, perf_df=perf_df)
    
    for n in excluded:
        score_map[n] = 0.0001
        
    draws = _get_draw_sets(df.tail(400))
    pair_counts = Counter()
    for age, draw in enumerate(reversed(draws)):
        w = math.exp(-age / 250)
        for pair in combinations(sorted(draw), 2):
            pair_counts[pair] += w
    mx_pair = max(pair_counts.values()) if pair_counts else 1.0
    pair_affinity = {p: 0.05 + 0.95 * c / mx_pair for p, c in pair_counts.items()}
    
    structure = _learn_advanced_combination_structure(df)
    latest_draw_set = set(df.iloc[-1][NUMBER_COLUMNS].astype(int).tolist()) if not df.empty else set()
    
    candidate_pool = [n for n in range(1, 46) if n not in pinned and n not in excluded]
    weights = [score_map[n] ** 2 for n in candidate_pool]
    tot_w = sum(weights)
    probs = [w / tot_w for w in weights]
    
    num_to_sample = 6 - len(pinned)
    generated_candidates = []
    seen_combos = set()
    
    target_attempts = 15000
    for i in range(target_attempts):
        if progress_callback and i % 1500 == 0:
            progress_callback(int((i / target_attempts) * 90), f"후보 조합 탐색 및 필터링 중... ({i}/{target_attempts})")
            
        sampled = np.random.choice(candidate_pool, size=num_to_sample, replace=False, p=probs)
        full_combo = tuple(sorted(list(pinned) + list(sampled)))
        
        if full_combo in seen_combos:
            continue
        seen_combos.add(full_combo)
        
        if is_statistically_valid_combination(full_combo, structure, latest_draw_set):
            score = calculate_combination_score(full_combo, score_map, pair_affinity, structure)
            generated_candidates.append((score, full_combo))
            
    if not generated_candidates:
        for _ in range(set_count * 2):
            sampled = np.random.choice(candidate_pool, size=num_to_sample, replace=False, p=probs)
            full_combo = tuple(sorted(list(pinned) + list(sampled)))
            score = calculate_combination_score(full_combo, score_map, pair_affinity, structure)
            generated_candidates.append((score, full_combo))
            
    generated_candidates.sort(key=lambda x: x[0], reverse=True)
    
    selected_sets = []
    for score, combo in generated_candidates:
        if len(selected_sets) >= set_count:
            break
        if any(len(set(combo) & set(s["numbers"])) >= 5 for s in selected_sets):
            continue
            
        s = sorted(combo)
        odd_c = sum(n % 2 == 1 for n in s)
        high_c = sum(n >= 23 for n in s)
        ac = calculate_ac_value(s)
        
        selected_sets.append({
            "set_index": len(selected_sets) + 1,
            "numbers": s,
            "score": score,
            "sum": sum(s),
            "odd_even": f"{odd_c}:{6 - odd_c}",
            "high_low": f"{high_c}:{6 - high_c}",
            "ac": ac,
            "span": s[-1] - s[0]
        })
        
    idx = 0
    while len(selected_sets) < set_count and idx < len(generated_candidates):
        score, combo = generated_candidates[idx]
        idx += 1
        if any(s["numbers"] == list(combo) for s in selected_sets):
            continue
        s = sorted(combo)
        odd_c = sum(n % 2 == 1 for n in s)
        high_c = sum(n >= 23 for n in s)
        ac = calculate_ac_value(s)
        selected_sets.append({
            "set_index": len(selected_sets) + 1,
            "numbers": s,
            "score": score,
            "sum": sum(s),
            "odd_even": f"{odd_c}:{6 - odd_c}",
            "high_low": f"{high_c}:{6 - high_c}",
            "ac": ac,
            "span": s[-1] - s[0]
        })
        
    if progress_callback:
        progress_callback(100, f"최적의 {len(selected_sets)}개 예상 조합 추출 완료!")
        
    return selected_sets


# ============================================================
# 6. 예측 결과 저장 및 과거 예측 성적 평가
# ============================================================

def save_predictions_to_csv(prediction_sets, target_draw, filename=PREDICTION_FILE):
    """예측 번호 CSV 저장 및 누적 관리"""
    rows = []
    for p in prediction_sets:
        s = p["numbers"]
        rows.append({
            "세트": p["set_index"],
            "번호1": s[0],
            "번호2": s[1],
            "번호3": s[2],
            "번호4": s[3],
            "번호5": s[4],
            "번호6": s[5],
            "예측회차": target_draw,
            "종합점수": p["score"],
            "합계": p["sum"],
            "홀짝": p["odd_even"],
            "AC값": p["ac"]
        })
        
    new_df = pd.DataFrame(rows)
    
    if os.path.exists(filename):
        try:
            old_df = pd.read_csv(filename, encoding="utf-8")
        except:
            old_df = pd.read_csv(filename, encoding="cp949")
        old_df = old_df[old_df["예측회차"] != target_draw]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
        
    combined.to_csv(filename, index=False, encoding="utf-8-sig")
    return combined


def evaluate_past_predictions(history_df, pred_filename=PREDICTION_FILE):
    """과거 예측한 번호와 실제 당첨번호 비교 평가 및 당첨 등수 산출"""
    if not os.path.exists(pred_filename):
        return pd.DataFrame(), {}
        
    try:
        pred_df = pd.read_csv(pred_filename, encoding="utf-8")
    except:
        pred_df = pd.read_csv(pred_filename, encoding="cp949")
        
    if pred_df.empty or "예측회차" not in pred_df.columns:
        return pd.DataFrame(), {}
        
    history_map = {}
    for _, row in history_df.iterrows():
        draw = int(row["회차"])
        nums = set(int(row[c]) for c in NUMBER_COLUMNS)
        bonus = int(row["보너스"])
        history_map[draw] = (nums, bonus)
        
    eval_rows = []
    prizes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "낙첨": 0}
    
    for _, row in pred_df.iterrows():
        target = int(row["예측회차"])
        pred_nums = set(int(row[c]) for c in NUMBER_COLUMNS if c in row)
        
        if target in history_map:
            actual_nums, actual_bonus = history_map[target]
            main_hits = len(pred_nums & actual_nums)
            bonus_hit = 1 if actual_bonus in pred_nums else 0
            
            if main_hits == 6:
                rank = "1등"
                prizes[1] += 1
            elif main_hits == 5 and bonus_hit == 1:
                rank = "2등"
                prizes[2] += 1
            elif main_hits == 5:
                rank = "3등"
                prizes[3] += 1
            elif main_hits == 4:
                rank = "4등"
                prizes[4] += 1
            elif main_hits == 3:
                rank = "5등"
                prizes[5] += 1
            else:
                rank = "낙첨"
                prizes["낙첨"] += 1
                
            eval_rows.append({
                "회차": target,
                "세트": row.get("세트", 1),
                "예측번호": sorted(list(pred_nums)),
                "실제번호": sorted(list(actual_nums)),
                "보너스": actual_bonus,
                "본번호적중": main_hits,
                "보너스적중": bonus_hit,
                "당첨결과": rank,
                "종합점수": row.get("종합점수", 0.0)
            })
            
    eval_df = pd.DataFrame(eval_rows)
    return eval_df, prizes


# ============================================================
# 7. 심층 통계 및 차트용 데이터 집계 함수들
# ============================================================

def get_number_frequency_stats(df, lookbacks=[0, 100, 30]):
    """1~45번 번호별 출현 빈도 통계 (전체, 최근 100회, 최근 30회)"""
    stats = []
    total_draws = len(df)
    
    counts_all = Counter(int(n) for n in df[NUMBER_COLUMNS].values.flatten())
    counts_100 = Counter(int(n) for n in df.tail(100)[NUMBER_COLUMNS].values.flatten())
    counts_30 = Counter(int(n) for n in df.tail(30)[NUMBER_COLUMNS].values.flatten())
    
    # 미출현 기간(Gap) 계산
    draws = _get_draw_sets(df)
    last_appearance = {}
    for n in range(1, 46):
        for idx in range(len(draws) - 1, -1, -1):
            if n in draws[idx]:
                last_appearance[n] = len(draws) - 1 - idx
                break
        if n not in last_appearance:
            last_appearance[n] = len(draws)
            
    for n in range(1, 46):
        stats.append({
            "번호": n,
            "전체빈도": counts_all.get(n, 0),
            "최근100회": counts_100.get(n, 0),
            "최근30회": counts_30.get(n, 0),
            "미출현회차": last_appearance.get(n, 0)
        })
        
    return pd.DataFrame(stats)


def get_cooccurrence_matrix(df, lookback=300):
    """45x45 번호 동반 출현 빈도 매트릭스 계산"""
    sample = df.tail(lookback)
    matrix = np.zeros((45, 45), dtype=int)
    
    for row in sample[NUMBER_COLUMNS].values:
        nums = [int(n) - 1 for n in row]  # 0-indexed
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                u, v = nums[i], nums[j]
                matrix[u, v] += 1
                matrix[v, u] += 1
                
    return matrix


def get_hot_cold_warm_numbers(df, lookback=10):
    """핫(2회 이상), 웜(1회), 콜드(0회) 번호 분류"""
    sample = df.tail(lookback)
    counts = Counter(int(n) for n in sample[NUMBER_COLUMNS].values.flatten())
    
    hot = []
    warm = []
    cold = []
    
    for n in range(1, 46):
        c = counts.get(n, 0)
        if c >= 2:
            hot.append((n, c))
        elif c == 1:
            warm.append((n, c))
        else:
            cold.append((n, c))
            
    hot.sort(key=lambda x: x[1], reverse=True)
    warm.sort(key=lambda x: x[0])
    cold.sort(key=lambda x: x[0])
    
    return hot, warm, cold
