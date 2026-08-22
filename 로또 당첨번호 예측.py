# ============================================================
# 로또 6/45 과거 데이터 자동 수집 + 분석 통합 프로그램
#
# 기능
# 1. 동행복권에서 세션 재사용·재시도로 과거 당첨번호 수집
# 2. 신규 및 중간 누락 회차 자동 수집
# 3. 당첨 이력 CSV 검증·저장
# 4. 분석용 파생변수 생성
# 5. 전체 번호 출현빈도 분석
# 6. 최근 100회 출현빈도 분석
# 7. 번호별 마지막 출현 이후 경과회차 계산
# 8. 기존 예상번호와 실제 당첨번호 자동 비교
# 9. 9개 분석기법의 순차 백테스트·성과 가중 앙상블
# 10. 본번호/보너스 관계와 경험분포 기반 15,000개 후보 조합 평가
# 11. 회차별 예상번호 및 분석기법 성과 누적 저장
#
# 실행 예시
# python "로또 당첨번호 예측.py" 1238
# 회차 인수를 생략하면 QUERY_ROUND 설정값을 사용합니다.
#
# 필요 라이브러리
# pip install requests pandas
# ============================================================

import os
import math
import sys
import requests
import pandas as pd
import time
import random
from collections import Counter
from datetime import datetime
from itertools import combinations


# ============================================================
# 1. 기본 설정
# ============================================================
CSV_FILE = "과거로또 당첨번호.csv"
PREDICTION_FILE = "당첨예상번호.csv"
METHOD_PERFORMANCE_FILE = "분석기법성과.csv"

# 0이면 기존 CSV만 사용합니다. 새 회차를 자동 수집하려면 회차를 입력하세요.
# 예: QUERY_ROUND = 1238
QUERY_ROUND = 0

PREDICTION_SET_COUNT = 10
PREDICTION_CANDIDATE_ATTEMPTS = 15000
BACKTEST_DRAWS = 120
MIN_BACKTEST_TRAIN_DRAWS = 150

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

# 회차별 요청 사이에서 연결과 쿠키를 재사용합니다.
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update(HEADERS)


# ============================================================
# 2. 특정 회차 조회
# ============================================================

def _get_lotto_draw_basic(draw_no):
    """
    특정 회차의 로또 데이터를 조회합니다.

    Parameters
    ----------
    draw_no : int
        조회할 로또 회차

    Returns
    -------
    dict 또는 None
    """

    params = {
        "srchLtEpsd": draw_no
    }

    try:

        response = requests.get(
            API_URL,
            headers=HEADERS,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # API 응답 구조 확인
        if "data" not in data:
            return None

        result_list = data["data"].get("list", [])

        if not result_list:
            return None

        item = result_list[0]

        # 회차 확인
        returned_draw = int(item.get("ltEpsd", 0))

        if returned_draw != draw_no:
            return None

        result = {

            "회차": returned_draw,

            "날짜": item.get("ltRflYmd"),

            "번호1": int(item.get("tm1WnNo")),
            "번호2": int(item.get("tm2WnNo")),
            "번호3": int(item.get("tm3WnNo")),
            "번호4": int(item.get("tm4WnNo")),
            "번호5": int(item.get("tm5WnNo")),
            "번호6": int(item.get("tm6WnNo")),

            "보너스": int(item.get("bnsWnNo")),

            # 추가 정보
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

        return result

    except requests.exceptions.RequestException as e:

        print(
            f"[네트워크 오류] {draw_no}회 : {e}"
        )

        return None

    except Exception as e:

        print(
            f"[파싱 오류] {draw_no}회 : {e}"
        )

        return None


def get_lotto_draw(draw_no, max_retries=3, retry_delay=1.0):
    """
    특정 회차를 세션 재사용과 제한적 재시도로 조회합니다.
    미래 회차, 빈 응답, 잘못된 JSON 및 번호 오류는 None으로 처리합니다.
    """

    try:
        draw_no = int(draw_no)

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
            returned_draw = int(item.get("ltEpsd", 0))

            result = {
                "회차": returned_draw,
                "날짜": item.get("ltRflYmd"),
                "번호1": int(item.get("tm1WnNo")),
                "번호2": int(item.get("tm2WnNo")),
                "번호3": int(item.get("tm3WnNo")),
                "번호4": int(item.get("tm4WnNo")),
                "번호5": int(item.get("tm5WnNo")),
                "번호6": int(item.get("tm6WnNo")),
                "보너스": int(item.get("bnsWnNo")),
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



# ============================================================
# 기존 CSV 로드
# ============================================================

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


# ============================================================
# 3. 전체 회차 자동 수집
# ============================================================

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
        print("QUERY_ROUND가 0이므로 네트워크 수집을 생략합니다.")
        print(f"기존 CSV 사용 : {CSV_FILE}")

        if not existing_df.empty:
            existing_rounds = set(existing_df["회차"].astype(int))
            latest_existing_round = max(existing_rounds)
            missing_rounds = [
                draw_no
                for draw_no in range(start_draw, latest_existing_round + 1)
                if draw_no not in existing_rounds
            ]

            if missing_rounds:
                print(
                    f"[주의] CSV 중간 누락 회차 {len(missing_rounds)}개 : "
                    + ", ".join(str(draw_no) for draw_no in missing_rounds)
                )
                print(
                    "QUERY_ROUND를 최신 회차로 지정하면 누락 회차도 함께 보충합니다."
                )

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


# ============================================================
# 4. 분석용 파생변수 생성
# ============================================================

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


# ============================================================
# 5. 번호 출현빈도 계산
# ============================================================

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


# ============================================================
# 6. 최근 N회 빈도 계산
# ============================================================

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


# ============================================================
# 7. 번호별 마지막 출현 이후 경과회차
# ============================================================

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


# ============================================================
# 8. 이전 예측 방식 (구형 파일 호환용, 현재 실행되지 않음)
# ============================================================

def _generate_prediction_sets_legacy(df, set_count=10, random_seed=None):
    """
    보너스 번호를 제외하고 1회부터 최신회차까지
    전 회차의 본 당첨번호 6개만 분석하여 예상번호를 생성합니다.

    전 회차 분석 항목:
    - 전체 출현빈도
    - 전체 회차 기준 마지막 출현 이후 미출현 회차
    - 전체 회차의 번호별 평균 출현 간격

    주의:
    로또 추첨은 무작위이므로 이 결과는 과거 전 회차 데이터를
    이용한 통계적 휴리스틱이며 실제 당첨을 보장하지 않습니다.
    """

    if random_seed is not None:
        random.seed(random_seed)

    number_cols = [
        "번호1",
        "번호2",
        "번호3",
        "번호4",
        "번호5",
        "번호6"
    ]

    # --------------------------------------------------------
    # 1. 전 회차 전체 출현빈도
    # --------------------------------------------------------
    all_numbers = df[number_cols].values.flatten()
    all_counts = Counter(all_numbers)

    # --------------------------------------------------------
    # 2. 전 회차 기준 미출현 회차
    # --------------------------------------------------------
    missing_df = calculate_missing_draws(df)

    missing_map = dict(
        zip(
            missing_df["번호"],
            missing_df["미출현회차"]
        )
    )

    # --------------------------------------------------------
    # 3. 전 회차 번호별 평균 출현 간격 계산
    # --------------------------------------------------------
    average_gap_map = {}

    for number in range(1, 46):

        appeared_rounds = df.loc[
            df[number_cols].eq(number).any(axis=1),
            "회차"
        ].astype(int).tolist()

        if len(appeared_rounds) >= 2:
            gaps = [
                appeared_rounds[i] - appeared_rounds[i - 1]
                for i in range(1, len(appeared_rounds))
            ]

            average_gap_map[number] = sum(gaps) / len(gaps)

        else:
            average_gap_map[number] = 0

    # --------------------------------------------------------
    # 4. 정규화 기준값
    # --------------------------------------------------------
    max_frequency = max(all_counts.values()) if all_counts else 1
    max_missing = max(missing_map.values()) if missing_map else 1
    max_average_gap = max(average_gap_map.values()) if average_gap_map else 1

    # --------------------------------------------------------
    # 5. 전 회차 기반 번호별 종합 점수
    #
    # 전체 출현빈도        60%
    # 현재 미출현 회차     25%
    # 전체 평균 출현 간격  15%
    # --------------------------------------------------------
    scores = {}

    for number in range(1, 46):

        frequency_score = (
            all_counts.get(number, 0)
            / max_frequency
        )

        missing_score = (
            missing_map.get(number, 0)
            / max_missing
        )

        average_gap_score = (
            average_gap_map.get(number, 0)
            / max_average_gap
        )

        score = (
            frequency_score * 0.60
            + missing_score * 0.25
            + average_gap_score * 0.15
        )

        # 동일한 조합 반복을 줄이기 위한 매우 작은 무작위 보정
        score += random.uniform(0, 0.02)

        scores[number] = score

    # --------------------------------------------------------
    # 6. 가중치 기반 6개 번호 추출
    # --------------------------------------------------------
    numbers = list(range(1, 46))
    weights = [
        max(scores[number], 0.0001)
        for number in numbers
    ]

    prediction_sets = []
    used_sets = set()

    attempts = 0
    max_attempts = 10000

    while (
        len(prediction_sets) < set_count
        and attempts < max_attempts
    ):

        attempts += 1

        selected = []

        available_numbers = numbers.copy()
        available_weights = weights.copy()

        # 중복 없이 6개 추출
        while len(selected) < 6:

            chosen = random.choices(
                available_numbers,
                weights=available_weights,
                k=1
            )[0]

            idx = available_numbers.index(chosen)

            selected.append(chosen)
            available_numbers.pop(idx)
            available_weights.pop(idx)

        selected = sorted(selected)
        selected_tuple = tuple(selected)

        if selected_tuple in used_sets:
            continue

        # ----------------------------------------------------
        # 과거 전 회차 분포에서 지나치게 극단적인 조합 제외
        # ----------------------------------------------------
        odd_count = sum(
            number % 2 == 1
            for number in selected
        )

        total_sum = sum(selected)

        # 홀수 2~4개 범위
        if odd_count < 2 or odd_count > 4:
            continue

        # 일반적인 합계 범위
        if total_sum < 100 or total_sum > 190:
            continue

        used_sets.add(selected_tuple)
        prediction_sets.append(selected)

    # 충분한 세트를 만들지 못한 경우 보완
    while len(prediction_sets) < set_count:

        selected = sorted(
            random.sample(numbers, 6)
        )

        selected_tuple = tuple(selected)

        if selected_tuple not in used_sets:
            used_sets.add(selected_tuple)
            prediction_sets.append(selected)

    result_df = pd.DataFrame(
        prediction_sets,
        columns=[
            "번호1",
            "번호2",
            "번호3",
            "번호4",
            "번호5",
            "번호6"
        ]
    )

    result_df.insert(
        0,
        "세트",
        range(1, len(result_df) + 1)
    )

    return result_df

def _save_prediction_sets_legacy(df, filename="당첨예상번호.csv"):
    """
    전 회차 분석 기반 예상번호 10세트를 생성하고
    반드시 CSV 파일로 저장합니다.

    저장 파일명 기본값:
    당첨예상번호.csv
    """

    prediction_df = _generate_prediction_sets_legacy(
        df,
        set_count=10
    )

    # UTF-8-SIG로 저장하여 Excel에서 한글 파일/컬럼명이 깨지지 않도록 처리
    prediction_df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("전 회차 분석 기반 이번주 당첨 예상번호 10세트")
    print("=" * 70)
    print(
        prediction_df.to_string(
            index=False
        )
    )

    print()
    print(f"예상번호 CSV 저장 완료 : {filename}")

    return prediction_df


# ============================================================
# 9. 회차별 예측 이력 + 다중 분석기법 앙상블
# ============================================================

PREDICTION_HISTORY_COLUMNS = [
    "예측대상회차",
    "기준회차",
    "생성일시",
    "모델버전",
    "예측방식",
    "세트",
    "번호1",
    "번호2",
    "번호3",
    "번호4",
    "번호5",
    "번호6",
    "조합점수",
    "평가상태",
    "적중개수",
    "적중번호",
    "보너스적중",
    "당첨등수",
    "실제당첨번호",
    "실제보너스"
]

# 사용자가 기존부터 사용해 온 당첨예상번호.csv의 고정 양식입니다.
# 이 파일에는 아래 열만 같은 순서로 누적하고, 모델 성과는 별도 CSV에 저장합니다.
PREDICTION_FILE_COLUMNS = (
    ["세트"]
    + NUMBER_COLUMNS
    + ["예측회차", "종합점수"]
    + [f"실제번호{index}" for index in range(1, 7)]
    + ["본번호일치수", "보너스일치", "총일치수"]
)


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


# 새 분석기법은 아래 사전에 함수만 추가하면 백테스트와 앙상블에 자동 포함됩니다.
ANALYSIS_METHODS = {
    "전체빈도": lambda df: _score_frequency(df),
    "최근30회빈도": lambda df: _score_frequency(df, recent_count=30),
    "최근100회빈도": lambda df: _score_frequency(df, recent_count=100),
    "지수가중빈도": _score_exponential_frequency,
    "보너스보조신호": _score_bonus_support,
    "최근추세": _score_recent_trend,
    "출현간격주기": _score_gap_cycle,
    "직전회차전이": _score_draw_transition,
    "동반출현연결": _score_pair_network
}

# 보너스 신호는 본번호 분석보다 낮은 사전가중치를 부여합니다.
ANALYSIS_METHOD_PRIORS = {
    method_name: (0.55 if method_name == "보너스보조신호" else 1.0)
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
    성능 차이는 20% 범위로 제한하여 우연한 과적합을 완화합니다.
    """

    method_names = list(ANALYSIS_METHODS)
    metrics = {
        method_name: {
            "top6_hits": 0.0,
            "top12_hits": 0.0,
            "rank_quality": 0.0
        }
        for method_name in method_names
    }

    start_index = max(
        min_train_draws,
        len(df) - backtest_draws
    )
    tested_draws = max(0, len(df) - start_index)

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
                "사전가중치": ANALYSIS_METHOD_PRIORS[method_name],
                "가중치": (
                    ANALYSIS_METHOD_PRIORS[method_name] / prior_total
                )
            }
            for method_name in method_names
        ])

    for target_index in range(start_index, len(df)):
        training_df = df.iloc[:target_index]
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

            metrics[method_name]["top6_hits"] += len(
                actual_numbers & set(ranked_numbers[:6])
            )
            metrics[method_name]["top12_hits"] += len(
                actual_numbers & set(ranked_numbers[:12])
            )
            metrics[method_name]["rank_quality"] += sum(
                (46 - rank_map[number]) / 45
                for number in actual_numbers
            ) / 6

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

        # 무작위 자료의 일시적인 성능 차이를 과신하지 않도록 75% 축소합니다.
        shrunk_performance = 1.0 + 0.25 * (raw_performance - 1.0)
        shrunk_performance = min(1.20, max(0.80, shrunk_performance))

        records.append({
            "분석기법": method_name,
            "백테스트회차수": tested_draws,
            "TOP6평균적중": round(average_top6, 4),
            "TOP12평균적중": round(average_top12, 4),
            "평균순위점수": round(average_rank_quality, 4),
            "성능지수": round(raw_performance, 4),
            "사전가중치": ANALYSIS_METHOD_PRIORS[method_name],
            "조정성능": (
                shrunk_performance * ANALYSIS_METHOD_PRIORS[method_name]
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
    """백테스트 성능 가중치로 분석기법별 점수를 결합합니다."""

    score_maps = calculate_analysis_scores(df)
    weight_map = dict(zip(
        method_performance_df["분석기법"],
        method_performance_df["가중치"]
    ))
    fallback_weight = 1.0 / len(score_maps)

    ensemble_scores = {}

    for number in range(1, 46):
        ensemble_scores[number] = sum(
            float(weight_map.get(method_name, fallback_weight))
            * score_map[number]
            for method_name, score_map in score_maps.items()
        )

    return _normalize_score_map(ensemble_scores)


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
        method_performance_df=None
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
                - maximum_overlap * 0.035
                - usage_penalty * 0.012
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


def _empty_prediction_history():
    return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)


def _load_prediction_history_expanded(
        latest_history_round,
        filename=PREDICTION_FILE
):
    """기존 예측 파일을 읽고 구형 10세트 파일도 회차 이력으로 이관합니다."""

    if not os.path.exists(filename):
        return _empty_prediction_history()

    try:
        prediction_df = pd.read_csv(
            filename,
            encoding="utf-8-sig"
        )
    except Exception as error:
        print(f"[예측 이력 로드 오류] {error}")
        return _empty_prediction_history()

    if prediction_df.empty:
        return _empty_prediction_history()

    legacy_file = "예측대상회차" not in prediction_df.columns

    if legacy_file:
        if "예측회차" in prediction_df.columns:
            legacy_target = pd.to_numeric(
                prediction_df["예측회차"],
                errors="coerce"
            ).fillna(int(latest_history_round) + 1)
        else:
            legacy_target = pd.Series(
                [int(latest_history_round) + 1] * len(prediction_df),
                index=prediction_df.index
            )

        target_draw = int(legacy_target.iloc[0])
        created_at = datetime.fromtimestamp(
            os.path.getmtime(filename)
        ).strftime("%Y-%m-%d %H:%M:%S")

        prediction_df["예측대상회차"] = legacy_target.astype(int)
        prediction_df["기준회차"] = (
            prediction_df["예측대상회차"] - 1
        )
        prediction_df["생성일시"] = created_at
        prediction_df["모델버전"] = "legacy-v1"
        prediction_df["예측방식"] = "기존방식(이관)"
        prediction_df["조합점수"] = pd.NA
        prediction_df["평가상태"] = "대기"
        prediction_df["적중개수"] = pd.NA
        prediction_df["적중번호"] = ""
        prediction_df["보너스적중"] = ""
        prediction_df["당첨등수"] = ""
        prediction_df["실제당첨번호"] = ""
        prediction_df["실제보너스"] = pd.NA

        print(
            f"기존 예상번호 10세트를 {target_draw}회 예측 이력으로 이관합니다."
        )

    if "세트" not in prediction_df.columns:
        prediction_df["세트"] = range(1, len(prediction_df) + 1)

    default_values = {
        "기준회차": "",
        "생성일시": "",
        "모델버전": "unknown",
        "예측방식": "unknown",
        "조합점수": pd.NA,
        "평가상태": "대기",
        "적중개수": pd.NA,
        "적중번호": "",
        "보너스적중": "",
        "당첨등수": "",
        "실제당첨번호": "",
        "실제보너스": pd.NA
    }

    for column, default_value in default_values.items():
        if column not in prediction_df.columns:
            prediction_df[column] = default_value

    # 이전 버전에서 사용하던 열이 있으면 새 형식으로 값을 옮깁니다.
    if "종합점수" in prediction_df.columns:
        legacy_scores = pd.to_numeric(
            prediction_df["종합점수"],
            errors="coerce"
        )

        if legacy_scores.notna().any() and legacy_scores.max() <= 1.5:
            legacy_scores = legacy_scores * 100

        current_scores = pd.to_numeric(
            prediction_df["조합점수"],
            errors="coerce"
        )
        prediction_df["조합점수"] = current_scores.fillna(
            legacy_scores
        ).round(2)

    legacy_actual_columns = [
        f"실제번호{index}"
        for index in range(1, 7)
    ]

    if all(
        column in prediction_df.columns
        for column in legacy_actual_columns
    ):
        legacy_actual = prediction_df[legacy_actual_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )
        actual_available = legacy_actual.notna().all(axis=1)
        actual_text = legacy_actual.apply(
            lambda row: ",".join(
                str(int(number))
                for number in sorted(row.dropna().tolist())
            ),
            axis=1
        )
        current_actual_text = (
            prediction_df["실제당첨번호"]
            .fillna("")
            .astype(str)
        )
        prediction_df.loc[
            actual_available & (current_actual_text == ""),
            "실제당첨번호"
        ] = actual_text

    if "본번호일치수" in prediction_df.columns:
        old_hit_count = pd.to_numeric(
            prediction_df["본번호일치수"],
            errors="coerce"
        )
        current_hit_count = pd.to_numeric(
            prediction_df["적중개수"],
            errors="coerce"
        )
        prediction_df["적중개수"] = current_hit_count.fillna(
            old_hit_count
        )

    if "보너스일치" in prediction_df.columns:
        old_bonus_hit = (
            prediction_df["보너스일치"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map({
                "true": "Y",
                "1": "Y",
                "y": "Y",
                "false": "N",
                "0": "N",
                "n": "N"
            })
        )
        current_bonus_hit = (
            prediction_df["보너스적중"]
            .fillna("")
            .astype(str)
        )
        prediction_df["보너스적중"] = current_bonus_hit.where(
            current_bonus_hit != "",
            old_bonus_hit
        )

    legacy_compatibility_columns = [
        "예측회차",
        "종합점수",
        "본번호일치수",
        "보너스일치",
        "총일치수"
    ] + legacy_actual_columns
    prediction_df = prediction_df.drop(
        columns=[
            column
            for column in legacy_compatibility_columns
            if column in prediction_df.columns
        ]
    )

    required_numeric_columns = [
        "예측대상회차",
        "세트"
    ] + NUMBER_COLUMNS

    for column in required_numeric_columns:
        prediction_df[column] = pd.to_numeric(
            prediction_df[column],
            errors="coerce"
        )

    prediction_df = prediction_df.dropna(
        subset=required_numeric_columns
    ).copy()

    for column in required_numeric_columns:
        prediction_df[column] = prediction_df[column].astype(int)

    prediction_df["기준회차"] = pd.to_numeric(
        prediction_df["기준회차"],
        errors="coerce"
    ).fillna(
        prediction_df["예측대상회차"] - 1
    ).astype(int)
    prediction_df["조합점수"] = pd.to_numeric(
        prediction_df["조합점수"],
        errors="coerce"
    ).round(2)
    prediction_df["적중개수"] = pd.to_numeric(
        prediction_df["적중개수"],
        errors="coerce"
    ).astype("Int64")
    prediction_df["실제보너스"] = pd.to_numeric(
        prediction_df["실제보너스"],
        errors="coerce"
    ).astype("Int64")

    text_columns = [
        "생성일시",
        "모델버전",
        "예측방식",
        "평가상태",
        "적중번호",
        "보너스적중",
        "당첨등수",
        "실제당첨번호"
    ]

    for column in text_columns:
        prediction_df[column] = (
            prediction_df[column]
            .fillna("")
            .astype(str)
        )

    prediction_df.loc[
        prediction_df["평가상태"] == "",
        "평가상태"
    ] = "대기"

    prediction_df = prediction_df.drop_duplicates(
        subset=["예측대상회차", "세트"],
        keep="first"
    ).sort_values(
        ["예측대상회차", "세트"]
    ).reset_index(drop=True)

    extra_columns = [
        column
        for column in prediction_df.columns
        if column not in PREDICTION_HISTORY_COLUMNS
    ]

    return prediction_df[
        PREDICTION_HISTORY_COLUMNS + extra_columns
    ]


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


def _evaluate_prediction_history_expanded(prediction_df, history_df):
    """당첨 결과가 추가된 회차의 기존 예측을 자동 채점합니다."""

    if prediction_df.empty:
        return prediction_df, set()

    evaluated_df = prediction_df.copy()
    actual_draws = {}

    for _, row in history_df.iterrows():
        draw_number = int(row["회차"])
        actual_draws[draw_number] = {
            "numbers": set(
                int(row[column])
                for column in NUMBER_COLUMNS
            ),
            "bonus": int(row["보너스"])
        }

    newly_evaluated_draws = set()

    for index, row in evaluated_df.iterrows():
        target_draw = int(row["예측대상회차"])

        if target_draw not in actual_draws:
            continue

        previous_status = str(row.get("평가상태", ""))
        actual = actual_draws[target_draw]
        predicted_numbers = set(
            int(row[column])
            for column in NUMBER_COLUMNS
        )
        hit_numbers = sorted(predicted_numbers & actual["numbers"])
        bonus_hit = (
            actual["bonus"] in predicted_numbers
            and actual["bonus"] not in actual["numbers"]
        )

        evaluated_df.at[index, "평가상태"] = "평가완료"
        evaluated_df.at[index, "적중개수"] = len(hit_numbers)
        evaluated_df.at[index, "적중번호"] = ",".join(
            str(number) for number in hit_numbers
        )
        evaluated_df.at[index, "보너스적중"] = "Y" if bonus_hit else "N"
        evaluated_df.at[index, "당첨등수"] = _lotto_prize_name(
            len(hit_numbers),
            bonus_hit
        )
        evaluated_df.at[index, "실제당첨번호"] = ",".join(
            str(number) for number in sorted(actual["numbers"])
        )
        evaluated_df.at[index, "실제보너스"] = actual["bonus"]

        if previous_status != "평가완료":
            newly_evaluated_draws.add(target_draw)

    return evaluated_df, newly_evaluated_draws


def _save_method_performance(
        method_performance_df,
        target_draw,
        base_draw,
        created_at,
        filename=METHOD_PERFORMANCE_FILE
):
    new_records = method_performance_df.copy()
    new_records.insert(0, "예측대상회차", int(target_draw))
    new_records.insert(1, "기준회차", int(base_draw))
    new_records.insert(2, "생성일시", created_at)

    if os.path.exists(filename):
        try:
            existing_records = pd.read_csv(
                filename,
                encoding="utf-8-sig"
            )
        except Exception:
            existing_records = pd.DataFrame()
    else:
        existing_records = pd.DataFrame()

    combined_records = pd.concat(
        [existing_records, new_records],
        ignore_index=True
    )
    combined_records = combined_records.drop_duplicates(
        subset=["예측대상회차", "분석기법"],
        keep="first"
    ).sort_values(
        ["예측대상회차", "분석기법"]
    )
    combined_records.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )


def _print_new_evaluation_summary_expanded(prediction_df, target_draws):
    for target_draw in sorted(target_draws):
        result = prediction_df[
            prediction_df["예측대상회차"] == target_draw
        ].copy()

        if result.empty:
            continue

        hit_counts = pd.to_numeric(
            result["적중개수"],
            errors="coerce"
        ).fillna(0)
        prize_count = int(
            (~result["당첨등수"].isin(["", "낙첨"])).sum()
        )
        actual_numbers = result.iloc[0]["실제당첨번호"]
        actual_bonus = result.iloc[0]["실제보너스"]

        print()
        print("=" * 70)
        print(f"{target_draw}회 기존 예측 자동 비교 결과")
        print("=" * 70)
        print(f"실제 당첨번호 : {actual_numbers} + {actual_bonus}")
        print(f"10세트 중 최고 적중수 : {int(hit_counts.max())}개")
        print(f"3개 이상 적중 세트 수 : {prize_count}개")


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


def save_prediction_sets(df, filename=PREDICTION_FILE):
    """
    기존 CSV 폼을 그대로 유지하면서 실제 당첨번호와 비교하고,
    최신 회차 다음 회차의 예측이 없을 때만 새 10세트를 추가합니다.
    """

    latest_round = int(df["회차"].max())
    target_draw = latest_round + 1
    prediction_history = load_prediction_history(
        latest_history_round=latest_round,
        filename=filename
    )
    prediction_history, newly_evaluated_draws = (
        evaluate_prediction_history(prediction_history, df)
    )

    current_prediction = prediction_history[
        prediction_history["예측회차"] == target_draw
    ].copy()
    generated_now = current_prediction.empty
    method_performance_df = None

    if generated_now:
        method_performance_df = backtest_analysis_methods(df)
        generated_prediction = generate_prediction_sets(
            df,
            set_count=PREDICTION_SET_COUNT,
            target_draw=target_draw,
            method_performance_df=method_performance_df
        )
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        current_prediction = generated_prediction[
            ["세트"] + NUMBER_COLUMNS
        ].copy()
        current_prediction["예측회차"] = target_draw
        current_prediction["종합점수"] = (
            generated_prediction["조합점수"] / 100
        ).round(6)

        for index in range(1, 7):
            current_prediction[f"실제번호{index}"] = pd.NA

        current_prediction["본번호일치수"] = pd.NA
        current_prediction["보너스일치"] = pd.NA
        current_prediction["총일치수"] = pd.NA
        current_prediction = current_prediction[
            PREDICTION_FILE_COLUMNS
        ]
        prediction_history = pd.concat(
            [prediction_history, current_prediction],
            ignore_index=True
        )

        _save_method_performance(
            method_performance_df,
            target_draw=target_draw,
            base_draw=latest_round,
            created_at=created_at
        )

    prediction_history = prediction_history.sort_values(
        ["예측회차", "세트"]
    ).reset_index(drop=True)
    prediction_history[PREDICTION_FILE_COLUMNS].to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    _print_new_evaluation_summary(
        prediction_history,
        newly_evaluated_draws,
        df
    )

    print()
    print("=" * 70)
    print(f"{target_draw}회 당첨 예상번호 {len(current_prediction)}세트")
    print("=" * 70)

    display_columns = (
        ["세트"]
        + NUMBER_COLUMNS
        + ["예측회차", "종합점수"]
    )

    print(
        current_prediction[display_columns].to_string(index=False)
    )

    if generated_now and method_performance_df is not None:
        print()
        print("분석기법별 최근 백테스트 성능 및 앙상블 가중치")
        performance_display = method_performance_df[
            [
                "분석기법",
                "백테스트회차수",
                "TOP6평균적중",
                "TOP12평균적중",
                "가중치"
            ]
        ].copy()
        performance_display["가중치"] = (
            performance_display["가중치"] * 100
        ).round(2)
        print(performance_display.to_string(index=False))
    else:
        print("이미 저장된 예측이 있어 새 세트를 중복 생성하지 않았습니다.")

    print()
    print(f"회차별 예상번호/평가 이력 저장 완료 : {filename}")

    if generated_now:
        print(f"분석기법 성과 이력 저장 완료 : {METHOD_PERFORMANCE_FILE}")

    return current_prediction


# ============================================================
# 10. 분석 결과 출력
# ============================================================

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


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # 데이터 수집
    # --------------------------------------------------------

    query_round = QUERY_ROUND

    if len(sys.argv) >= 2:
        try:
            query_round = int(sys.argv[1])
        except ValueError:
            print(
                f"[실행 인수 오류] 회차는 정수여야 합니다 : {sys.argv[1]}"
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

    prediction_df = save_prediction_sets(
        df,
        filename="당첨예상번호.csv"
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

    if os.path.exists(METHOD_PERFORMANCE_FILE):
        print(f"분석기법 성과 CSV : {METHOD_PERFORMANCE_FILE}")

    print("기본 빈도 요약은 화면에 출력합니다.")


# ============================================================
# 프로그램 실행
# ============================================================

if __name__ == "__main__":

    main()
