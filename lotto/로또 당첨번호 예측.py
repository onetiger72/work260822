# 로또 결과 수집·기법 검증·번호 생성. 영구 저장은 당첨 이력과 예상번호 CSV 두 개만 사용합니다.

import os

import json

from lotto_validation import strict_int, validate_history, final_review, print_review

from lotto_weekly import latest_completed_draw, verify_latest_result

from lotto_portfolio import generate_prediction_sets as generate_main_hit_portfolio

import math

import sys

import requests

import pandas as pd

import time


from collections import Counter


from itertools import combinations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "과거로또 당첨번호.csv")

PREDICTION_FILE = os.path.join(BASE_DIR, "당첨예상번호.csv")

QUERY_ROUND = 0

PREDICTION_SET_COUNT = 10

BACKTEST_DRAWS = 120

MIN_BACKTEST_TRAIN_DRAWS = 150

RECENT_REVIEW_DRAWS = 2

RECENT_REVIEW_PRIOR_DRAWS = 18

PREDICTION_FEEDBACK_PRIOR_DRAWS = 38

NUMBER_COLUMNS = [
    "번호1",
    "번호2",
    "번호3",
    "번호4",
    "번호5",
    "번호6"
]

API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"

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

def get_lotto_draw(draw_no, max_retries=3, retry_delay=1.0):
    """
    특정 회차를 세션 재사용과 제한적 재시도로 조회합니다.
    미래 회차, 빈 응답, 잘못된 JSON 및 번호 오류는 None으로 처리합니다.
    """

    try:
        draw_no = strict_int(draw_no)

        if draw_no < 1:
            print(f"[조회 제외] 유효하지 않은 회차 : {draw_no}")
            return None
    except (TypeError, ValueError):
        print(f"[조회 제외] 회차 값이 숫자가 아닙니다 : {draw_no}")
        return None

    params = {"srchLtEpsd": draw_no}

    try:
        response = None

        for attempt in range(1, max_retries + 1):
            try:
                response = HTTP_SESSION.get(
                    API_URL,
                    params=params,
                    timeout=10
                )

                if (
                    response.status_code != 429
                    and response.status_code < 500
                ):
                    break

                if attempt < max_retries:
                    print(
                        f"[조회 재시도] {draw_no}회 HTTP "
                        f"{response.status_code} ({attempt}/{max_retries})"
                    )
                    time.sleep(retry_delay * attempt)
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError
            ) as error:
                if attempt >= max_retries:
                    raise

                print(
                    f"[조회 재시도] {draw_no}회 "
                    f"{type(error).__name__} ({attempt}/{max_retries})"
                )
                time.sleep(retry_delay * attempt)

        if response is None:
            return None

        if response.status_code != 200:
            print(
                f"[조회 실패] {draw_no}회 HTTP 상태코드 : "
                f"{response.status_code}"
            )
            return None

        if not response.text or not response.text.strip():
            print(f"[조회 없음] {draw_no}회 응답이 비어 있습니다.")
            return None

        try:
            data = response.json()
        except ValueError:
            print(f"[조회 없음] {draw_no}회 응답이 JSON 형식이 아닙니다.")
            return None

        if not isinstance(data, dict):
            print(f"[조회 없음] {draw_no}회 응답 형식이 올바르지 않습니다.")
            return None

        data_block = data.get("data")

        if not isinstance(data_block, dict):
            print(f"[조회 없음] {draw_no}회 데이터가 존재하지 않습니다.")
            return None

        result_list = data_block.get("list", [])

        if not isinstance(result_list, list) or not result_list:
            print(
                f"[조회 없음] {draw_no}회는 아직 존재하지 않거나 "
                "당첨결과가 발표되지 않았습니다."
            )
            return None

        item = result_list[0]

        if not isinstance(item, dict):
            print(f"[조회 없음] {draw_no}회 결과 형식이 올바르지 않습니다.")
            return None

        required_number_fields = [
            "tm1WnNo",
            "tm2WnNo",
            "tm3WnNo",
            "tm4WnNo",
            "tm5WnNo",
            "tm6WnNo",
            "bnsWnNo"
        ]

        if any(
            item.get(field) in (None, "")
            for field in required_number_fields
        ):
            print(f"[조회 없음] {draw_no}회 당첨번호 필드가 비어 있습니다.")
            return None

        try:
            returned_draw = strict_int(item.get("ltEpsd", 0))

            result = {
                "회차": returned_draw,
                "날짜": item.get("ltRflYmd"),
                "번호1": strict_int(item.get("tm1WnNo")),
                "번호2": strict_int(item.get("tm2WnNo")),
                "번호3": strict_int(item.get("tm3WnNo")),
                "번호4": strict_int(item.get("tm4WnNo")),
                "번호5": strict_int(item.get("tm5WnNo")),
                "번호6": strict_int(item.get("tm6WnNo")),
                "보너스": strict_int(item.get("bnsWnNo")),
                "1등당첨자수": item.get("rnk1WnNope"),
                "1등당첨금": item.get("rnk1WnAmt"),
                "1등총당첨금": item.get("rnk1SumWnAmt"),
                "2등당첨자수": item.get("rnk2WnNope"),
                "2등당첨금": item.get("rnk2WnAmt"),
                "3등당첨자수": item.get("rnk3WnNope"),
                "3등당첨금": item.get("rnk3WnAmt"),
                "4등당첨금": item.get("rnk4WnAmt"),
                "5등당첨금": item.get("rnk5WnAmt"),
                "총당첨자수": item.get("sumWnNope"),
                "판매금액": item.get("rlvtEpsdSumNtslAmt")
            }
        except (TypeError, ValueError) as error:
            print(f"[파싱 오류] {draw_no}회 당첨번호 변환 실패 : {error}")
            return None

        if returned_draw != draw_no:
            print(
                f"[조회 없음] 요청 {draw_no}회 / 응답 {returned_draw}회 불일치"
            )
            return None

        main_numbers = [
            result[column]
            for column in NUMBER_COLUMNS
        ]

        if (
            len(set(main_numbers)) != 6
            or any(number < 1 or number > 45 for number in main_numbers)
            or not 1 <= result["보너스"] <= 45
            or result["보너스"] in main_numbers
        ):
            print(f"[검증 실패] {draw_no}회 당첨번호 값이 유효하지 않습니다.")
            return None

        return result

    except requests.exceptions.Timeout:
        print(f"[네트워크 오류] {draw_no}회 조회 시간 초과")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[네트워크 오류] {draw_no}회 서버 연결 실패")
        return None
    except requests.exceptions.RequestException as error:
        print(f"[네트워크 오류] {draw_no}회 : {error}")
        return None
    except Exception as error:
        print(f"[예외 처리] {draw_no}회 조회 중 오류 : {error}")
        return None

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

def collect_all_lotto(
        query_round,
        start_draw=1,
        sleep_time=0.10
):
    """
    사용자가 조회한 회차(query_round)까지 기존 CSV와 비교하여
    마지막 회차 이후의 신규 데이터와 중간 누락 회차를 수집합니다.

    예)
    CSV 마지막 회차 = 1180
    조회 회차 = 1184
    -> 1181, 1182, 1183, 1184를 추가

    임의의 최대 회차 상한값은 사용하지 않습니다.
    """

    print()
    print("=" * 70)
    print("조회 회차와 CSV 마지막 회차 비교")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 조회 회차 및 기존 CSV 확인
    # --------------------------------------------------------
    query_round = int(query_round)
    existing_df = load_history_csv()

    if query_round <= 0:
        print("QUERY_ROUND가 0이므로 홈페이지의 다음 회차를 자동 확인합니다.")
        print(f"기존 CSV 사용 : {CSV_FILE}")

        # 0을 '네트워크 완전 생략'으로 처리하면 CSV가 1237회에 머문 뒤
        # 홈페이지에 이미 발표된 1238회를 놓칠 수 있습니다. 현재 CSV의
        # 마지막 회차 다음 회차만 먼저 조회하여 누락분만 보충합니다.
        if not existing_df.empty:
            latest_existing_round = int(existing_df["회차"].max())
            next_round = latest_existing_round + 1
            next_result = get_lotto_draw(next_round)
            if next_result is not None:
                print(f"홈페이지에서 {next_round}회 데이터를 확인했습니다.")
                query_round = next_round
            else:
                print("CSV 이후 발표된 새 회차가 없어 기존 CSV를 사용합니다.")
                return existing_df
        else:
            print("기존 CSV가 비어 있어 자동 수집을 진행할 수 없습니다.")
            return existing_df

    # --------------------------------------------------------
    # 2. 기존 CSV 로드 및 마지막 회차 확인
    # --------------------------------------------------------
    if existing_df.empty:
        csv_last_round = 0
        print(f"기존 CSV 없음 : {CSV_FILE}")
    else:
        csv_last_round = int(existing_df["회차"].max())
        print(f"CSV 마지막 회차 : {csv_last_round}회")

    print(f"조회 회차          : {query_round}회")

    # --------------------------------------------------------
    # 3. 조회 범위에서 신규 및 중간 누락 회차 확인
    # --------------------------------------------------------
    existing_rounds = (
        set(existing_df["회차"].astype(int))
        if not existing_df.empty
        else set()
    )
    target_rounds = [
        draw_no
        for draw_no in range(start_draw, query_round + 1)
        if draw_no not in existing_rounds
    ]

    if not target_rounds:
        print()
        print(f"1회부터 {query_round}회까지 누락된 데이터가 없습니다.")
        return existing_df

    # --------------------------------------------------------
    # 4. 수집 대상 출력
    # --------------------------------------------------------
    print()
    target_text = ", ".join(str(draw_no) for draw_no in target_rounds)

    if len(target_rounds) > 20:
        target_text = (
            ", ".join(str(draw_no) for draw_no in target_rounds[:10])
            + " ... "
            + ", ".join(str(draw_no) for draw_no in target_rounds[-5:])
        )

    print(f"수집 대상 회차 : {target_text}")
    print(
        f"수집 대상 건수 : "
        f"{len(target_rounds)}개"
    )

    # --------------------------------------------------------
    # 5. 차이 나는 회차만 조회
    # --------------------------------------------------------
    new_data = []

    for index, draw_no in enumerate(
        target_rounds,
        start=1
    ):

        result = get_lotto_draw(draw_no)

        if result is not None:
            new_data.append(result)

            print(
                f"[{index}/{len(target_rounds)}] "
                f"{draw_no}회 추가 완료"
            )

        else:
            print(
                f"[{index}/{len(target_rounds)}] "
                f"{draw_no}회 조회 실패"
            )

        time.sleep(sleep_time)

    if not new_data:
        print("추가할 당첨번호 데이터를 가져오지 못했습니다.")
        return existing_df

    # --------------------------------------------------------
    # 6. 기존 데이터 뒤에 신규 회차 추가
    # --------------------------------------------------------
    new_df = pd.DataFrame(new_data)

    if existing_df.empty:
        combined_df = new_df

    else:
        combined_df = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True
        )

    combined_df = (
        combined_df
        .drop_duplicates(subset=["회차"], keep="last")
        .sort_values("회차")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 7. CSV 저장
    # --------------------------------------------------------
    combined_df.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print(f"CSV 업데이트 완료 : {CSV_FILE}")
    print(f"기존 마지막 회차   : {csv_last_round}회")
    print(f"조회 회차          : {query_round}회")
    print(f"누락/신규 추가 수  : {len(new_df)}개")
    print(f"현재 마지막 회차   : {int(combined_df['회차'].max())}회")
    print("=" * 70)

    return combined_df

def create_features(df):

    # 호출자가 가진 원본 당첨 이력을 변경하지 않습니다.
    df = df.copy()

    number_cols = [
        "번호1",
        "번호2",
        "번호3",
        "번호4",
        "번호5",
        "번호6"
    ]

    # ----------------------------
    # 합계
    # ----------------------------

    df["합계"] = (
        df[number_cols]
        .sum(axis=1)
    )

    # ----------------------------
    # 평균
    # ----------------------------

    df["평균"] = (
        df[number_cols]
        .mean(axis=1)
        .round(2)
    )

    # ----------------------------
    # 홀수 개수
    # ----------------------------

    df["홀수개수"] = df[number_cols].apply(

        lambda row:
        sum(
            number % 2 == 1
            for number in row
        ),

        axis=1
    )

    # ----------------------------
    # 짝수 개수
    # ----------------------------

    df["짝수개수"] = (
        6 - df["홀수개수"]
    )

    # ----------------------------
    # 최소값 / 최대값
    # ----------------------------

    df["최소값"] = (
        df[number_cols]
        .min(axis=1)
    )

    df["최대값"] = (
        df[number_cols]
        .max(axis=1)
    )

    # ----------------------------
    # 번호 범위
    # ----------------------------

    df["범위"] = (
        df["최대값"]
        - df["최소값"]
    )

    # ----------------------------
    # 연속번호 Pair 개수
    # ----------------------------

    def count_consecutive(row):

        numbers = sorted(
            row.tolist()
        )

        count = 0

        for i in range(
            len(numbers) - 1
        ):

            if (
                numbers[i + 1]
                - numbers[i]
                == 1
            ):

                count += 1

        return count

    df["연속번호개수"] = (

        df[number_cols]
        .apply(
            count_consecutive,
            axis=1
        )
    )

    # ----------------------------
    # 구간별 개수
    # ----------------------------

    df["1~10"] = df[number_cols].apply(

        lambda row:
        sum(
            1 <= number <= 10
            for number in row
        ),

        axis=1
    )

    df["11~20"] = df[number_cols].apply(

        lambda row:
        sum(
            11 <= number <= 20
            for number in row
        ),

        axis=1
    )

    df["21~30"] = df[number_cols].apply(

        lambda row:
        sum(
            21 <= number <= 30
            for number in row
        ),

        axis=1
    )

    df["31~40"] = df[number_cols].apply(

        lambda row:
        sum(
            31 <= number <= 40
            for number in row
        ),

        axis=1
    )

    df["41~45"] = df[number_cols].apply(

        lambda row:
        sum(
            41 <= number <= 45
            for number in row
        ),

        axis=1
    )

    return df

def calculate_frequency(df):
    """
    본번호와 보너스 번호를 역할별로 집계합니다.
    보너스는 화면용 통합지표에서만 30%의 보조 가중치로 반영합니다.
    """

    number_cols = [
        "번호1",
        "번호2",
        "번호3",
        "번호4",
        "번호5",
        "번호6"
    ]

    all_numbers = (

        df[number_cols]
        .values
        .flatten()
    )

    bonus_numbers = df["보너스"].values.flatten()

    counts = Counter(all_numbers)
    bonus_counts = Counter(bonus_numbers)

    frequency_df = pd.DataFrame({

        "번호": range(1, 46),

        "출현횟수": [
            counts.get(number, 0)
            for number in range(1, 46)
        ],

        "보너스출현횟수": [
            bonus_counts.get(number, 0)
            for number in range(1, 46)
        ]
    })

    total = max(len(df), 1)

    frequency_df[
        "회차대비출현율(%)"
    ] = (

        frequency_df["출현횟수"]
        / total
        * 100

    ).round(2)

    frequency_df["보너스출현율(%)"] = (
        frequency_df["보너스출현횟수"]
        / total
        * 100
    ).round(2)

    frequency_df["통합가중출현점수"] = (
        frequency_df["출현횟수"]
        + frequency_df["보너스출현횟수"] * 0.30
    ).round(2)

    return frequency_df

def calculate_recent_frequency(
        df,
        recent_count=100
):

    recent_df = df.tail(
        recent_count
    )

    return calculate_frequency(
        recent_df
    )

def calculate_missing_draws(df):

    number_cols = [
        "번호1",
        "번호2",
        "번호3",
        "번호4",
        "번호5",
        "번호6"
    ]

    ordered_df = (
        df
        .sort_values("회차")
        .drop_duplicates(subset=["회차"], keep="last")
        .reset_index(drop=True)
    )
    latest_draw = int(ordered_df["회차"].iloc[-1])

    result = []

    for number in range(
        1,
        46
    ):

        mask = (

            ordered_df[number_cols]
            .eq(number)
            .any(axis=1)

        )

        appeared = ordered_df[
            mask
        ]

        if len(appeared) > 0:

            last_draw = int(
                appeared[
                    "회차"
                ].max()
            )

            last_position = int(appeared.index[-1])
            missing = len(ordered_df) - last_position - 1

        else:

            last_draw = None
            missing = len(ordered_df)

        result.append({

            "번호": number,

            "마지막출현회차":
                last_draw,

            "미출현회차":
                missing
        })

    return pd.DataFrame(
        result
    )

PREDICTION_FILE_COLUMNS = (
    ["예측회차", "세트"]
    + NUMBER_COLUMNS
    + ["종합점수"]
    + [f"실제번호{index}" for index in range(1, 7)]
    + ["본번호일치수", "보너스일치", "총일치수"]
)

PREDICTION_FEEDBACK_COLUMNS = [
    "평가회차",
    "세트수",
    "평균본번호적중",
    "최고본번호적중",
    "번호커버리지",
    "기대커버리지",
    "최대번호노출률",
    "브라이어점수",
    "균등기준브라이어",
    "브라이어스킬",
    "누적평가회차수",
    "누적평균브라이어스킬",
    "피드백신뢰도",
    "점수분산계수",
    "다양성보강계수"
]

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

def calculate_prediction_feedback(prediction_df):
    """
    실제로 발행한 10세트의 번호별 노출확률과 당첨 여부를 비교합니다.

    Brier skill은 균등 포함확률(6/45)보다 나았는지를 측정합니다. 평가 회차가
    적을 때는 38회 사전표본으로 강하게 수축해 우연한 1~2회 결과가 다음
    예측을 크게 흔들지 않게 합니다.
    """

    empty_summary = {
        "evaluated_draws": 0,
        "mean_brier_skill": 0.0,
        "mean_main_hits": 0.0,
        "best_main_hits": 0,
        "mean_coverage": 0.0,
        "mean_max_exposure": 0.0,
        "reliability": 0.0,
        "score_spread_multiplier": 1.0,
        "diversity_penalty_multiplier": 1.0
    }

    if prediction_df.empty:
        return empty_summary, pd.DataFrame(
            columns=PREDICTION_FEEDBACK_COLUMNS
        )

    actual_columns = [
        f"실제번호{index}"
        for index in range(1, 7)
    ]

    if not set(actual_columns).issubset(prediction_df.columns):
        return empty_summary, pd.DataFrame(
            columns=PREDICTION_FEEDBACK_COLUMNS
        )

    base_probability = 6 / 45
    uniform_brier = (
        6 * (1.0 - base_probability) ** 2
        + 39 * base_probability ** 2
    ) / 45
    records = []

    for target_draw, group in prediction_df.groupby("예측회차"):
        group = group.copy()

        if group.empty or not group[actual_columns].notna().all().all():
            continue

        actual_numbers = {
            int(group.iloc[0][column])
            for column in actual_columns
        }
        set_count = len(group)
        exposure_counts = Counter(
            int(number)
            for number in group[NUMBER_COLUMNS].values.flatten()
        )
        exposures = {
            number: exposure_counts.get(number, 0) / set_count
            for number in range(1, 46)
        }
        brier_score = sum(
            (
                exposures[number]
                - int(number in actual_numbers)
            ) ** 2
            for number in range(1, 46)
        ) / 45
        brier_skill = 1.0 - brier_score / uniform_brier
        hit_counts = [
            len(
                {
                    int(row[column])
                    for column in NUMBER_COLUMNS
                }
                & actual_numbers
            )
            for _, row in group.iterrows()
        ]
        expected_coverage = 45 * (
            1.0 - (39 / 45) ** set_count
        )

        records.append({
            "평가회차": int(target_draw),
            "세트수": set_count,
            "평균본번호적중": round(sum(hit_counts) / set_count, 6),
            "최고본번호적중": max(hit_counts),
            "번호커버리지": len(exposure_counts),
            "기대커버리지": round(expected_coverage, 6),
            "최대번호노출률": round(max(exposures.values()), 6),
            "브라이어점수": round(brier_score, 8),
            "균등기준브라이어": round(uniform_brier, 8),
            "브라이어스킬": round(brier_skill, 8)
        })

    if not records:
        return empty_summary, pd.DataFrame(
            columns=PREDICTION_FEEDBACK_COLUMNS
        )

    detail_df = pd.DataFrame(records).sort_values(
        "평가회차"
    ).reset_index(drop=True)
    evaluated_draws = len(detail_df)
    reliability = evaluated_draws / (
        evaluated_draws + PREDICTION_FEEDBACK_PRIOR_DRAWS
    )
    mean_brier_skill = float(detail_df["브라이어스킬"].mean())
    bounded_skill = min(0.25, max(-0.25, mean_brier_skill))
    score_spread_multiplier = min(
        1.05,
        max(0.95, 1.0 + reliability * bounded_skill)
    )
    mean_coverage = float(detail_df["번호커버리지"].mean())
    mean_expected_coverage = float(
        detail_df["기대커버리지"].mean()
    )
    mean_max_exposure = float(
        detail_df["최대번호노출률"].mean()
    )
    coverage_deficit = max(
        0.0,
        (mean_expected_coverage - mean_coverage)
        / max(mean_expected_coverage, 1.0)
    )
    exposure_excess = max(
        0.0,
        (mean_max_exposure - 0.40) / 0.60
    )
    diversity_issue = (
        coverage_deficit * 0.60
        + exposure_excess * 0.40
    )
    diversity_penalty_multiplier = min(
        1.15,
        1.0 + reliability * diversity_issue
    )
    summary = {
        "evaluated_draws": evaluated_draws,
        "mean_brier_skill": mean_brier_skill,
        "mean_main_hits": float(
            detail_df["평균본번호적중"].mean()
        ),
        "best_main_hits": int(
            detail_df["최고본번호적중"].max()
        ),
        "mean_coverage": mean_coverage,
        "mean_max_exposure": mean_max_exposure,
        "reliability": reliability,
        "score_spread_multiplier": score_spread_multiplier,
        "diversity_penalty_multiplier": diversity_penalty_multiplier
    }

    detail_df["누적평가회차수"] = evaluated_draws
    detail_df["누적평균브라이어스킬"] = round(
        mean_brier_skill,
        8
    )
    detail_df["피드백신뢰도"] = round(reliability, 8)
    detail_df["점수분산계수"] = round(
        score_spread_multiplier,
        8
    )
    detail_df["다양성보강계수"] = round(
        diversity_penalty_multiplier,
        8
    )

    return summary, detail_df[PREDICTION_FEEDBACK_COLUMNS]

def generate_prediction_sets(
        df, set_count=PREDICTION_SET_COUNT, random_seed=None, target_draw=None,
        method_performance_df=None, prediction_feedback=None, llm_audit=None
):
    """기법별 순차 검증 후 번호 노출을 분산한 6개 번호 조합을 생성합니다.

    과거 호환 인수인 가중치/노출 피드백/LLM 제안은 이 모델에 적용하지 않습니다.
    조합점수는 번호 포함 추정치 합이며 당첨확률이 아닙니다.
    """
    return generate_main_hit_portfolio(
        df, calculate_analysis_scores, set_count=set_count,
        random_seed=random_seed, target_draw=target_draw
    )

def _lotto_prize_name(main_hit_count, bonus_hit):
    if main_hit_count == 6:
        return "1등"
    if main_hit_count == 5 and bonus_hit:
        return "2등"
    if main_hit_count == 5:
        return "3등"
    if main_hit_count == 4:
        return "4등"
    if main_hit_count == 3:
        return "5등"

    return "낙첨"

def _empty_prediction_file_history():
    return pd.DataFrame(columns=PREDICTION_FILE_COLUMNS)

def _parse_actual_number_text(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    for character in "[]()":
        text = text.replace(character, " ")

    numbers = []

    for token in text.replace(",", " ").split():
        try:
            number = int(float(token))
        except ValueError:
            continue

        if 1 <= number <= 45:
            numbers.append(number)

    return sorted(dict.fromkeys(numbers))[:6]

def _bonus_hit_to_integer(value):
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()

    if normalized in {"y", "yes", "true", "1", "1.0"}:
        return 1
    if normalized in {"n", "no", "false", "0", "0.0"}:
        return 0

    return pd.NA

def load_prediction_history(
        latest_history_round,
        filename=PREDICTION_FILE
):
    """
    현재 당첨예상번호.csv의 기존 예측을 읽어 고정 폼으로 유지합니다.
    이전 실행에서 확장된 열이 있더라도 기존 폼으로 안전하게 되돌립니다.
    """

    if not os.path.exists(filename):
        return _empty_prediction_file_history()

    try:
        source_df = pd.read_csv(
            filename,
            encoding="utf-8-sig"
        )
    except Exception as error:
        print(f"[예측 이력 로드 오류] {error}")
        return _empty_prediction_file_history()

    if source_df.empty:
        return _empty_prediction_file_history()

    result_df = pd.DataFrame(index=source_df.index)

    if "세트" in source_df.columns:
        result_df["세트"] = source_df["세트"]
    else:
        result_df["세트"] = range(1, len(source_df) + 1)

    for column in NUMBER_COLUMNS:
        result_df[column] = (
            source_df[column]
            if column in source_df.columns
            else pd.NA
        )

    if "예측회차" in source_df.columns:
        result_df["예측회차"] = source_df["예측회차"]
    elif "예측대상회차" in source_df.columns:
        result_df["예측회차"] = source_df["예측대상회차"]
    else:
        result_df["예측회차"] = int(latest_history_round) + 1

    if "종합점수" in source_df.columns:
        combined_score = pd.to_numeric(
            source_df["종합점수"],
            errors="coerce"
        )
    elif "조합점수" in source_df.columns:
        combined_score = pd.to_numeric(
            source_df["조합점수"],
            errors="coerce"
        )

        if combined_score.notna().any() and combined_score.max() > 1.5:
            combined_score = combined_score / 100
    else:
        combined_score = pd.Series(
            pd.NA,
            index=source_df.index,
            dtype="Float64"
        )

    result_df["종합점수"] = combined_score

    parsed_actual_numbers = (
        source_df["실제당첨번호"].map(_parse_actual_number_text)
        if "실제당첨번호" in source_df.columns
        else pd.Series(
            [[] for _ in range(len(source_df))],
            index=source_df.index
        )
    )

    actual_number_columns = [
        f"실제번호{index}"
        for index in range(1, 7)
    ]

    for position, column in enumerate(actual_number_columns):
        if column in source_df.columns:
            direct_values = pd.to_numeric(
                source_df[column],
                errors="coerce"
            )
        else:
            direct_values = pd.Series(
                pd.NA,
                index=source_df.index,
                dtype="Float64"
            )

        fallback_values = pd.Series(
            [
                numbers[position]
                if len(numbers) > position
                else pd.NA
                for numbers in parsed_actual_numbers
            ],
            index=source_df.index,
            dtype="Float64"
        )
        result_df[column] = direct_values.fillna(fallback_values)

    if "본번호일치수" in source_df.columns:
        main_hit_count = pd.to_numeric(
            source_df["본번호일치수"],
            errors="coerce"
        )
    elif "적중개수" in source_df.columns:
        main_hit_count = pd.to_numeric(
            source_df["적중개수"],
            errors="coerce"
        )
    else:
        main_hit_count = pd.Series(
            pd.NA,
            index=source_df.index,
            dtype="Float64"
        )

    if "보너스일치" in source_df.columns:
        bonus_hit = source_df["보너스일치"].map(
            _bonus_hit_to_integer
        )
    elif "보너스적중" in source_df.columns:
        bonus_hit = source_df["보너스적중"].map(
            _bonus_hit_to_integer
        )
    else:
        bonus_hit = pd.Series(
            pd.NA,
            index=source_df.index,
            dtype="Int64"
        )

    main_hit_count = pd.to_numeric(
        main_hit_count,
        errors="coerce"
    ).astype("Int64")
    bonus_hit = pd.to_numeric(
        bonus_hit,
        errors="coerce"
    ).astype("Int64")

    if "총일치수" in source_df.columns:
        total_hit_count = pd.to_numeric(
            source_df["총일치수"],
            errors="coerce"
        )
    else:
        total_hit_count = pd.Series(
            pd.NA,
            index=source_df.index,
            dtype="Float64"
        )

    evaluated_rows = main_hit_count.notna() | bonus_hit.notna()
    calculated_total = (
        main_hit_count.fillna(0) + bonus_hit.fillna(0)
    )
    total_hit_count = pd.to_numeric(
        total_hit_count,
        errors="coerce"
    ).fillna(
        calculated_total.where(evaluated_rows, pd.NA)
    ).astype("Int64")

    result_df["본번호일치수"] = main_hit_count
    result_df["보너스일치"] = bonus_hit
    result_df["총일치수"] = total_hit_count

    required_numeric_columns = [
        "세트",
        "예측회차"
    ] + NUMBER_COLUMNS

    for column in required_numeric_columns:
        result_df[column] = pd.to_numeric(
            result_df[column],
            errors="coerce"
        )

    result_df = result_df.dropna(
        subset=required_numeric_columns
    ).copy()

    for column in required_numeric_columns:
        result_df[column] = result_df[column].astype(int)

    valid_predictions = (
        result_df[NUMBER_COLUMNS]
        .apply(lambda column: column.between(1, 45))
        .all(axis=1)
        & (result_df[NUMBER_COLUMNS].nunique(axis=1) == 6)
    )
    result_df = result_df[valid_predictions].copy()
    result_df["종합점수"] = pd.to_numeric(
        result_df["종합점수"],
        errors="coerce"
    ).round(6)

    for column in actual_number_columns:
        result_df[column] = pd.to_numeric(
            result_df[column],
            errors="coerce"
        ).astype("Int64")

    result_df = result_df.drop_duplicates(
        subset=["예측회차", "세트"],
        keep="first"
    ).sort_values(
        ["예측회차", "세트"]
    ).reset_index(drop=True)

    if list(source_df.columns) != PREDICTION_FILE_COLUMNS:
        print("기존 예측 내용은 유지하고 당첨예상번호.csv 폼을 통일합니다.")

    return result_df[PREDICTION_FILE_COLUMNS]

def evaluate_prediction_history(prediction_df, history_df):
    """새 당첨 회차가 있으면 기존 폼의 실제번호·일치수 열을 채웁니다."""

    if prediction_df.empty:
        return prediction_df, set()

    evaluated_df = prediction_df.copy()
    actual_draws = {}

    for _, row in history_df.iterrows():
        draw_number = int(row["회차"])
        actual_draws[draw_number] = {
            "numbers": sorted(
                int(row[column])
                for column in NUMBER_COLUMNS
            ),
            "bonus": int(row["보너스"])
        }

    actual_number_columns = [
        f"실제번호{index}"
        for index in range(1, 7)
    ]
    newly_evaluated_draws = set()

    for index, row in evaluated_df.iterrows():
        target_draw = int(row["예측회차"])

        if target_draw not in actual_draws:
            continue

        was_evaluated = all(
            pd.notna(row[column])
            for column in actual_number_columns
        )
        actual_numbers = actual_draws[target_draw]["numbers"]
        actual_number_set = set(actual_numbers)
        predicted_numbers = {
            int(row[column])
            for column in NUMBER_COLUMNS
        }
        main_hit_count = len(predicted_numbers & actual_number_set)
        bonus_hit = int(
            actual_draws[target_draw]["bonus"] in predicted_numbers
        )

        for column, number in zip(
            actual_number_columns,
            actual_numbers
        ):
            evaluated_df.at[index, column] = number

        evaluated_df.at[index, "본번호일치수"] = main_hit_count
        evaluated_df.at[index, "보너스일치"] = bonus_hit
        evaluated_df.at[index, "총일치수"] = (
            main_hit_count + bonus_hit
        )

        if not was_evaluated:
            newly_evaluated_draws.add(target_draw)

    return evaluated_df[PREDICTION_FILE_COLUMNS], newly_evaluated_draws

def _print_new_evaluation_summary(
        prediction_df,
        target_draws,
        history_df
):
    for target_draw in sorted(target_draws):
        result = prediction_df[
            prediction_df["예측회차"] == target_draw
        ].copy()

        if result.empty:
            continue

        hit_counts = pd.to_numeric(
            result["본번호일치수"],
            errors="coerce"
        ).fillna(0)
        bonus_hits = pd.to_numeric(
            result["보너스일치"],
            errors="coerce"
        ).fillna(0)
        prize_names = [
            _lotto_prize_name(int(main_hits), bool(bonus_hit))
            for main_hits, bonus_hit in zip(hit_counts, bonus_hits)
        ]
        prize_priority = {
            "1등": 1,
            "2등": 2,
            "3등": 3,
            "4등": 4,
            "5등": 5,
            "낙첨": 6
        }
        best_prize = min(
            prize_names,
            key=lambda prize: prize_priority[prize]
        )
        actual_numbers = ",".join(
            str(int(result.iloc[0][f"실제번호{index}"]))
            for index in range(1, 7)
        )
        actual_row = history_df[
            history_df["회차"] == target_draw
        ]
        actual_bonus = (
            int(actual_row.iloc[0]["보너스"])
            if not actual_row.empty
            else ""
        )

        print()
        print("=" * 70)
        print(f"{target_draw}회 기존 예측 자동 비교 결과")
        print("=" * 70)
        print(f"실제 당첨번호 : {actual_numbers} + {actual_bonus}")
        print(f"예측 세트 중 최고 본번호 적중수 : {int(hit_counts.max())}개")
        print(f"최고 당첨 등수 : {best_prize}")
        print(
            "5등 이상 당첨 세트 수 : "
            f"{sum(prize != '낙첨' for prize in prize_names)}개"
        )

def save_prediction_sets(df, filename=PREDICTION_FILE, manual_review=None):
    """Evaluate existing tickets and save only the prediction CSV; reviews stay in memory."""
    validate_history(df)
    verify_latest_result(df, get_lotto_draw, expected_round=latest_completed_draw())
    latest_round = int(df["회차"].max())
    target_draw = latest_round + 1
    prediction_history = load_prediction_history(latest_history_round=latest_round, filename=filename)
    prediction_history, newly_evaluated_draws = evaluate_prediction_history(prediction_history, df)
    prediction_feedback, _ = calculate_prediction_feedback(prediction_history)
    current_prediction = prediction_history[prediction_history["예측회차"] == target_draw].copy()
    generated_now = current_prediction.empty
    weekly_report = None

    if generated_now:
        generated = generate_prediction_sets(df, set_count=PREDICTION_SET_COUNT, target_draw=target_draw)
        weekly_report = generated.attrs["weekly_review"]
        previous = prediction_history[prediction_history["예측회차"] == latest_round]
        weekly_report["previous_prediction_evaluation"] = {
            "draw": latest_round, "available": not previous.empty, "sets": len(previous),
            "main_hits": pd.to_numeric(previous["본번호일치수"], errors="coerce").dropna().astype(int).tolist(),
            "limits": "저장 시점이 확인되지 않는 기록은 사전 예측 성과를 입증하지 않습니다.",
        }
        current_prediction = generated[["세트"] + NUMBER_COLUMNS].copy()
        current_prediction["예측회차"] = target_draw
        current_prediction["종합점수"] = generated["조합점수"].round(6)
        for column in [f"실제번호{i}" for i in range(1, 7)] + ["본번호일치수", "보너스일치", "총일치수"]:
            current_prediction[column] = pd.NA
        current_prediction = current_prediction[PREDICTION_FILE_COLUMNS]
        prediction_history = pd.concat([prediction_history, current_prediction], ignore_index=True)
        print("주간 분석 검토:")
        print(json.dumps(weekly_report, ensure_ascii=False, indent=2))

    final_audit = final_review(df, current_prediction, prediction_history, get_lotto_draw,
                               expected_count=PREDICTION_SET_COUNT, manual_review=manual_review)
    print_review(final_audit)
    if final_audit["status"] in {"validation_error", "source_mismatch", "reject"}:
        raise ValueError(f"최종 번호 검증 실패: {final_audit['status']}")
    if not final_audit["finalized"]:
        print("아래 번호는 검증 대기 후보이며 GPT 최종 확정 번호가 아닙니다.")

    prediction_history = prediction_history.sort_values(["예측회차", "세트"]).reset_index(drop=True)
    prediction_history[PREDICTION_FILE_COLUMNS].to_csv(filename, index=False, encoding="utf-8-sig")
    _print_new_evaluation_summary(prediction_history, newly_evaluated_draws, df)
    print(f"{target_draw}회 당첨 예상번호 {len(current_prediction)}세트")
    print(current_prediction[["예측회차", "세트"] + NUMBER_COLUMNS + ["종합점수"]].to_string(index=False))
    if not generated_now:
        print("이미 저장된 예측이 있어 새 세트를 중복 생성하지 않았습니다.")
    else:
        print("기법별 사전 검증·반영 강도 검증·번호 노출 분산 방식으로 생성했습니다.")
    print("종합점수는 모델의 본번호 포함 추정치 합이며 1등 당첨확률이 아닙니다.")
    print(f"기존 예측 평가: {prediction_feedback['evaluated_draws']}회, "
          f"평균 본번호 적중 {prediction_feedback['mean_main_hits']:.3f}개")
    print(f"회차별 예상번호/평가 이력 저장 완료: {filename}")
    current_prediction.attrs["weekly_review"] = weekly_report
    current_prediction.attrs["final_review"] = final_audit
    current_prediction.attrs["prediction_feedback"] = prediction_feedback
    return current_prediction

def print_summary(
        df,
        frequency_df,
        recent_frequency_df,
        missing_df
):

    print()
    print("=" * 70)
    print("데이터 수집 결과")
    print("=" * 70)

    print(
        f"총 회차 : "
        f"{len(df):,}회"
    )

    print(
        f"최초 회차 : "
        f"{df['회차'].min()}회"
    )

    print(
        f"최신 회차 : "
        f"{df['회차'].max()}회"
    )

    print()

    print(
        "최근 당첨번호"
    )

    latest = df.iloc[-1]

    print(
        f"{int(latest['회차'])}회 : "
        f"{int(latest['번호1'])}, "
        f"{int(latest['번호2'])}, "
        f"{int(latest['번호3'])}, "
        f"{int(latest['번호4'])}, "
        f"{int(latest['번호5'])}, "
        f"{int(latest['번호6'])} "
        f"+ {int(latest['보너스'])}"
    )

    print()
    print("=" * 70)
    print("전체 기간 많이 나온 번호 TOP 10")
    print("=" * 70)

    print(

        frequency_df
        .sort_values(
            "출현횟수",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )

    )

    print()
    print("=" * 70)
    print("최근 100회 많이 나온 번호 TOP 10")
    print("=" * 70)

    print(

        recent_frequency_df
        .sort_values(
            "출현횟수",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )

    )

    print()
    print("=" * 70)
    print("오랫동안 나오지 않은 번호 TOP 10")
    print("=" * 70)

    print(

        missing_df
        .sort_values(
            "미출현회차",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )

    )

def main():

    if "--audit-only" in sys.argv:
        from audit_predictions import audit
        audit(sys.modules[__name__], offline="--offline" in sys.argv)
        return

    # --------------------------------------------------------
    # STEP 1
    # 데이터 수집
    # --------------------------------------------------------

    query_round = QUERY_ROUND
    if query_round <= 0:
        query_round = latest_completed_draw()

    arguments = [arg for arg in sys.argv[1:] if arg != "--manual-response"]
    if arguments:
        try:
            query_round = int(arguments[0])
        except ValueError:
            print(
                f"[실행 인수 오류] 회차는 정수여야 합니다 : {arguments[0]}"
            )
            return

    df = collect_all_lotto(
        query_round=query_round,
        start_draw=1,
        sleep_time=0.10
    )

    if df.empty:
        print("데이터를 가져오지 못했습니다.")
        return

    df = (
        df
        .sort_values("회차")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # STEP 2
    # 파생변수 생성
    # --------------------------------------------------------

    df = create_features(df)

    # --------------------------------------------------------
    # STEP 3
    # 전체 빈도 분석
    # --------------------------------------------------------

    frequency_df = (
        calculate_frequency(df)
    )

    # --------------------------------------------------------
    # STEP 4
    # 최근 100회 분석
    # --------------------------------------------------------

    recent_frequency_df = (
        calculate_recent_frequency(
            df,
            recent_count=100
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # 미출현 기간 분석
    # --------------------------------------------------------

    missing_df = (
        calculate_missing_draws(df)
    )


    # --------------------------------------------------------
    # STEP 6
    # 기존 예측 평가 + 다음 회차 예상번호 생성 및 누적 저장
    # --------------------------------------------------------

    proposal = json.load(sys.stdin) if "--manual-response" in sys.argv else None
    prediction_df = save_prediction_sets(
        df, filename=PREDICTION_FILE, manual_review=proposal
    )

    # --------------------------------------------------------
    # STEP 7
    # 화면 출력
    # --------------------------------------------------------

    print_summary(

        df,

        frequency_df,

        recent_frequency_df,

        missing_df
    )

    print()
    print("=" * 70)
    print("데이터 처리 완료")
    print("=" * 70)
    print(f"당첨 이력 CSV : {CSV_FILE}")
    print(f"예측/평가 이력 CSV : {PREDICTION_FILE}")


    print("기본 빈도 요약은 화면에 출력합니다.")

if __name__ == "__main__":

    main()
