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
# 9. 13개 분석기법의 순차 백테스트·최근 2회 보강·성과 가중 앙상블
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
import json
from lotto_validation import strict_int, validate_history, final_review
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
# 실행한 현재 폴더가 아니라 이 스크립트가 있는 폴더를 데이터 폴더로 사용합니다.
# 따라서 C:\work에서 실행하거나 IDE에서 실행해도 lotto 폴더의 CSV를 읽습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "과거로또 당첨번호.csv")
PREDICTION_FILE = os.path.join(BASE_DIR, "당첨예상번호.csv")
METHOD_PERFORMANCE_FILE = os.path.join(BASE_DIR, "분석기법성과.csv")
PREDICTION_FEEDBACK_FILE = os.path.join(BASE_DIR, "예측피드백성과.csv")
METHOD_FORECAST_FILE = os.path.join(BASE_DIR, "분석기법예측기록.csv")
LLM_AUDIT_FILE = os.path.join(BASE_DIR, "LLM분석감사.json")
LLM_AUDIT_REQUEST_FILE = os.path.join(BASE_DIR, "LLM감사요청.json")
LLM_AUDIT_PROPOSAL_FILE = os.path.join(BASE_DIR, "LLM감사제안.json")

# 0이면 기존 CSV만 사용합니다. 새 회차를 자동 수집하려면 회차를 입력하세요.
# 예: QUERY_ROUND = 1238
QUERY_ROUND = 0

PREDICTION_SET_COUNT = 10
PREDICTION_CANDIDATE_ATTEMPTS = 15000
BACKTEST_DRAWS = 120
MIN_BACKTEST_TRAIN_DRAWS = 150
RECENT_REVIEW_DRAWS = 2
RECENT_REVIEW_PRIOR_DRAWS = 18
PREDICTION_FEEDBACK_PRIOR_DRAWS = 38
METHOD_LIVE_FEEDBACK_PRIOR_DRAWS = 30
LLM_MIN_COMPLETED_DRAWS = 10
LLM_MAX_WEIGHT_DELTA = 0.05

# LLM 설정은 환경변수로만 받습니다.
# LOTTO_LLM_MODE=off|manual|audit|capped (기본 off)
# OPENAI_API_KEY는 audit/capped에서만 필요하며 코드나 CSV에 저장하지 않습니다.

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

METHOD_FORECAST_COLUMNS = [
    "예측대상회차",
    "기준회차",
    "생성일시",
    "분석기법",
    "예측TOP6",
    "예측TOP12",
    "전체순위",
    "평가상태",
    "실제당첨번호",
    "TOP6적중",
    "TOP12적중",
    "순위점수",
    "실전성능지수"
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
    "동반출현연결": _score_pair_network,
    "핫콜드밸런스": _score_hot_cold_balance,
    "마코프전이": _score_markov_transition,
    "끝수다이내믹스": _score_last_digit,
    "최근2회문맥": _score_recent_two_draw_context
}

# 보너스와 최근 2회 문맥은 본번호 장기 분석보다 낮은 사전가중치를 부여합니다.
# 최근 2회 문맥은 독립 120회 구간 검증에서 안정적인 초과성과가 확인되지 않아
# 탐색용 보조 신호(대략 전체 가중치의 2~3%)로만 사용합니다.
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


def save_prediction_feedback(
        feedback_df,
        filename=PREDICTION_FEEDBACK_FILE
):
    """회차별 실전 예측 보정 근거를 별도 CSV에 저장합니다."""

    feedback_df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )


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


def _parse_ranked_numbers(value, maximum=45):
    if pd.isna(value):
        return []

    numbers = []

    for token in str(value).replace(" ", "").split(","):
        try:
            number = int(token)
        except ValueError:
            continue

        if 1 <= number <= 45 and number not in numbers:
            numbers.append(number)

    return numbers[:maximum]


def create_method_forecast_records(
        df,
        target_draw,
        created_at=None
):
    """각 분석기법의 예측 당시 순위를 저장해 향후 실전 성과를 귀속합니다."""

    created_at = created_at or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    score_maps = calculate_analysis_scores(df)
    records = []

    for method_name, score_map in score_maps.items():
        ranked_numbers = sorted(
            range(1, 46),
            key=lambda number: (-score_map[number], number)
        )
        records.append({
            "예측대상회차": int(target_draw),
            "기준회차": int(df["회차"].max()),
            "생성일시": created_at,
            "분석기법": method_name,
            "예측TOP6": ",".join(
                str(number)
                for number in ranked_numbers[:6]
            ),
            "예측TOP12": ",".join(
                str(number)
                for number in ranked_numbers[:12]
            ),
            "전체순위": ",".join(
                str(number)
                for number in ranked_numbers
            ),
            "평가상태": "대기",
            "실제당첨번호": "",
            "TOP6적중": pd.NA,
            "TOP12적중": pd.NA,
            "순위점수": pd.NA,
            "실전성능지수": pd.NA
        })

    return pd.DataFrame(records)[METHOD_FORECAST_COLUMNS]


def load_method_forecast_history(filename=METHOD_FORECAST_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame(columns=METHOD_FORECAST_COLUMNS)

    try:
        forecast_df = pd.read_csv(
            filename,
            encoding="utf-8-sig"
        )
    except Exception as error:
        print(f"[분석기법 예측기록 로드 오류] {error}")
        return pd.DataFrame(columns=METHOD_FORECAST_COLUMNS)

    for column in METHOD_FORECAST_COLUMNS:
        if column not in forecast_df.columns:
            forecast_df[column] = pd.NA

    text_columns = [
        "생성일시",
        "분석기법",
        "예측TOP6",
        "예측TOP12",
        "전체순위",
        "평가상태",
        "실제당첨번호"
    ]

    for column in text_columns:
        forecast_df[column] = forecast_df[column].astype("object")

    return forecast_df[METHOD_FORECAST_COLUMNS]


def save_method_forecast_records(
        new_records,
        filename=METHOD_FORECAST_FILE
):
    """동일 회차의 기법별 예측 스냅샷은 현재 로직 결과로 교체합니다."""

    existing_records = load_method_forecast_history(filename)

    if not new_records.empty:
        target_draws = set(
            pd.to_numeric(
                new_records["예측대상회차"],
                errors="coerce"
            ).dropna().astype(int)
        )
        existing_draws = pd.to_numeric(
            existing_records["예측대상회차"],
            errors="coerce"
        )
        existing_records = existing_records[
            ~existing_draws.isin(target_draws)
        ].copy()

    combined_records = pd.concat(
        [existing_records, new_records],
        ignore_index=True
    )
    combined_records = combined_records.drop_duplicates(
        subset=["예측대상회차", "분석기법"],
        keep="last"
    ).sort_values(
        ["예측대상회차", "분석기법"]
    ).reset_index(drop=True)
    combined_records[METHOD_FORECAST_COLUMNS].to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    return combined_records[METHOD_FORECAST_COLUMNS]


def evaluate_method_forecast_history(
        history_df,
        filename=METHOD_FORECAST_FILE
):
    """예측 당시 저장한 기법별 순위를 실제 당첨번호로 사후 평가합니다."""

    forecast_df = load_method_forecast_history(filename)

    if forecast_df.empty:
        return forecast_df

    actual_draws = {
        int(row["회차"]): {
            int(row[column])
            for column in NUMBER_COLUMNS
        }
        for _, row in history_df.iterrows()
    }
    expected_top6_hits = 6 * 6 / 45
    expected_top12_hits = 12 * 6 / 45
    expected_rank_quality = 23 / 45
    changed = False

    for index, row in forecast_df.iterrows():
        try:
            target_draw = int(row["예측대상회차"])
        except (TypeError, ValueError):
            continue

        if target_draw not in actual_draws:
            continue

        ranked_numbers = _parse_ranked_numbers(
            row["전체순위"],
            maximum=45
        )

        if len(ranked_numbers) != 45:
            continue

        actual_numbers = actual_draws[target_draw]
        top6_hits = len(actual_numbers & set(ranked_numbers[:6]))
        top12_hits = len(actual_numbers & set(ranked_numbers[:12]))
        rank_map = {
            number: rank
            for rank, number in enumerate(ranked_numbers, start=1)
        }
        rank_quality = sum(
            (46 - rank_map[number]) / 45
            for number in actual_numbers
        ) / 6
        live_performance = (
            0.50 * top6_hits / expected_top6_hits
            + 0.30 * top12_hits / expected_top12_hits
            + 0.20 * rank_quality / expected_rank_quality
        )
        forecast_df.at[index, "평가상태"] = "평가완료"
        forecast_df.at[index, "실제당첨번호"] = ",".join(
            str(number)
            for number in sorted(actual_numbers)
        )
        forecast_df.at[index, "TOP6적중"] = top6_hits
        forecast_df.at[index, "TOP12적중"] = top12_hits
        forecast_df.at[index, "순위점수"] = round(rank_quality, 6)
        forecast_df.at[index, "실전성능지수"] = round(
            live_performance,
            6
        )
        changed = True

    if changed:
        forecast_df[METHOD_FORECAST_COLUMNS].to_csv(
            filename,
            index=False,
            encoding="utf-8-sig"
        )

    return forecast_df[METHOD_FORECAST_COLUMNS]


def calculate_live_method_feedback(method_forecast_df):
    """저장된 실전 기법 성과를 강하게 수축한 보정계수로 변환합니다."""

    feedback = {}

    if method_forecast_df.empty:
        return feedback

    evaluated = method_forecast_df[
        method_forecast_df["평가상태"] == "평가완료"
    ].copy()
    evaluated["실전성능지수"] = pd.to_numeric(
        evaluated["실전성능지수"],
        errors="coerce"
    )
    evaluated = evaluated.dropna(subset=["실전성능지수"])

    for method_name, group in evaluated.groupby("분석기법"):
        group = group.sort_values("예측대상회차").tail(30)
        draw_count = len(group)
        live_performance = float(group["실전성능지수"].mean())
        reliability = draw_count / (
            draw_count + METHOD_LIVE_FEEDBACK_PRIOR_DRAWS
        )
        bounded_skill = min(
            0.50,
            max(-0.50, live_performance - 1.0)
        )
        multiplier = min(
            1.10,
            max(0.90, 1.0 + reliability * bounded_skill)
        )
        feedback[method_name] = {
            "draw_count": draw_count,
            "performance": live_performance,
            "reliability": reliability,
            "multiplier": multiplier
        }

    return feedback


def apply_live_method_feedback(
        method_performance_df,
        live_feedback
):
    """백테스트 가중치에 실전 스냅샷 성과를 곱하고 다시 정규화합니다."""

    adjusted_df = method_performance_df.copy()
    adjusted_df["실전피드백회차수"] = adjusted_df["분석기법"].map(
        lambda name: live_feedback.get(name, {}).get("draw_count", 0)
    )
    adjusted_df["실전성능지수"] = adjusted_df["분석기법"].map(
        lambda name: live_feedback.get(name, {}).get("performance", 1.0)
    )
    adjusted_df["실전보정계수"] = adjusted_df["분석기법"].map(
        lambda name: live_feedback.get(name, {}).get("multiplier", 1.0)
    )
    adjusted_df["가중치"] = (
        pd.to_numeric(adjusted_df["가중치"], errors="coerce").fillna(0.0)
        * adjusted_df["실전보정계수"]
    )
    total_weight = float(adjusted_df["가중치"].sum())

    if total_weight > 0:
        adjusted_df["가중치"] = adjusted_df["가중치"] / total_weight

    return adjusted_df


def _safe_environment_float(name, default, minimum, maximum):
    """환경변수 실수값을 안전한 범위 안에서 읽습니다."""

    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = float(default)

    if not math.isfinite(value):
        value = float(default)

    return min(maximum, max(minimum, value))


def build_llm_audit_payload(
        target_draw,
        method_performance_df,
        live_feedback,
        prediction_feedback
):
    """
    LLM에는 번호나 원본 CSV가 아닌 집계 성과만 전달합니다.

    번호 선택은 통계 로직이 담당하고 LLM은 분석기법 가중치와 데이터 품질을
    감사하는 보조자 역할만 맡습니다.
    """

    methods = []

    for _, row in method_performance_df.iterrows():
        method_name = str(row["분석기법"])
        live = live_feedback.get(method_name, {})
        methods.append({
            "method": method_name,
            "backtest_draws": int(row.get("백테스트회차수", 0)),
            "top6_average_hits": round(
                float(row.get("TOP6평균적중", 0.0)),
                6
            ),
            "top12_average_hits": round(
                float(row.get("TOP12평균적중", 0.0)),
                6
            ),
            "reinforced_performance": round(
                float(row.get("보강성능지수", 1.0)),
                6
            ),
            "live_draws": int(live.get("draw_count", 0)),
            "live_performance": round(
                float(live.get("performance", 1.0)),
                6
            ),
            "live_multiplier": round(
                float(live.get("multiplier", 1.0)),
                6
            ),
            "statistical_weight": round(
                float(row.get("가중치", 0.0)),
                8
            )
        })

    completed_draws = max(
        (
            int(item.get("draw_count", 0))
            for item in live_feedback.values()
        ),
        default=0
    )
    feedback_keys = (
        "evaluated_draws",
        "mean_brier_skill",
        "mean_main_hits",
        "best_main_hits",
        "mean_coverage",
        "mean_max_exposure",
        "reliability",
        "score_spread_multiplier",
        "diversity_penalty_multiplier"
    )

    return {
        "schema_version": "1.0",
        "purpose": "method_weight_and_data_quality_audit_only",
        "target_draw": int(target_draw),
        "completed_live_draws": completed_draws,
        "minimum_draws_for_application": LLM_MIN_COMPLETED_DRAWS,
        "maximum_relative_delta": LLM_MAX_WEIGHT_DELTA,
        "selection_feedback": {
            key: prediction_feedback.get(key, 0)
            for key in feedback_keys
        },
        "methods": methods
    }


def _write_json_file(filename, value):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _validate_llm_audit_proposal(proposal, allowed_methods):
    """LLM 출력 전체를 검증하고 허용된 작은 조정만 반환합니다."""

    if not isinstance(proposal, dict):
        raise ValueError("감사 제안은 JSON 객체여야 합니다.")

    action = proposal.get("action")

    if action not in {"keep", "review", "adjust"}:
        raise ValueError("허용되지 않은 action입니다.")

    summary = str(proposal.get("summary", ""))[:600]
    warnings = proposal.get("warnings", [])

    if not isinstance(warnings, list) or len(warnings) > 8:
        raise ValueError("warnings 형식이 올바르지 않습니다.")

    clean_warnings = [str(item)[:240] for item in warnings]
    adjustments = proposal.get("adjustments", [])

    if not isinstance(adjustments, list) or len(adjustments) > 13:
        raise ValueError("adjustments 형식이 올바르지 않습니다.")

    allowed_methods = set(allowed_methods)
    clean_adjustments = []
    seen_methods = set()

    for adjustment in adjustments:
        if not isinstance(adjustment, dict):
            raise ValueError("개별 조정값은 JSON 객체여야 합니다.")

        method_name = str(adjustment.get("method", ""))

        if method_name not in allowed_methods:
            raise ValueError("등록되지 않은 분석기법이 포함되었습니다.")

        if method_name in seen_methods:
            raise ValueError("중복 분석기법 조정이 포함되었습니다.")

        relative_delta = float(adjustment.get("relative_delta", 0.0))
        confidence = float(adjustment.get("confidence", 0.0))

        if not math.isfinite(relative_delta) or not math.isfinite(confidence):
            raise ValueError("유한하지 않은 조정값이 포함되었습니다.")

        if abs(relative_delta) > LLM_MAX_WEIGHT_DELTA:
            raise ValueError("분석기법 조정폭이 안전 상한을 넘었습니다.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence가 허용 범위를 벗어났습니다.")

        evidence_keys = adjustment.get("evidence_keys", [])

        if not isinstance(evidence_keys, list) or len(evidence_keys) > 5:
            raise ValueError("evidence_keys 형식이 올바르지 않습니다.")

        clean_adjustments.append({
            "method": method_name,
            "relative_delta": relative_delta,
            "confidence": confidence,
            "evidence_keys": [str(item)[:80] for item in evidence_keys],
            "reason": str(adjustment.get("reason", ""))[:240]
        })
        seen_methods.add(method_name)

    diversity_multiplier = float(
        proposal.get("diversity_multiplier", 1.0)
    )

    if not math.isfinite(diversity_multiplier):
        raise ValueError("다양성 조정값이 유한하지 않습니다.")

    if not 0.95 <= diversity_multiplier <= 1.05:
        raise ValueError("다양성 조정값이 안전 범위를 벗어났습니다.")

    if action != "adjust" and clean_adjustments:
        raise ValueError("adjust가 아닌 응답에는 가중치 조정이 없어야 합니다.")

    return {
        "schema_version": "1.0",
        "action": action,
        "adjustments": clean_adjustments,
        "diversity_multiplier": diversity_multiplier,
        "warnings": clean_warnings,
        "summary": summary
    }


def _llm_audit_json_schema():
    adjustment_schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string"},
            "relative_delta": {
                "type": "number",
                "minimum": -LLM_MAX_WEIGHT_DELTA,
                "maximum": LLM_MAX_WEIGHT_DELTA
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "evidence_keys": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5
            },
            "reason": {"type": "string", "maxLength": 240}
        },
        "required": [
            "method",
            "relative_delta",
            "confidence",
            "evidence_keys",
            "reason"
        ],
        "additionalProperties": False
    }

    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["1.0"]
            },
            "action": {
                "type": "string",
                "enum": ["keep", "review", "adjust"]
            },
            "adjustments": {
                "type": "array",
                "items": adjustment_schema,
                "maxItems": 13
            },
            "diversity_multiplier": {
                "type": "number",
                "minimum": 0.95,
                "maximum": 1.05
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 8
            },
            "summary": {"type": "string", "maxLength": 600}
        },
        "required": [
            "schema_version",
            "action",
            "adjustments",
            "diversity_multiplier",
            "warnings",
            "summary"
        ],
        "additionalProperties": False
    }


def request_llm_weight_audit(
        payload,
        allowed_methods,
        audit_filename=LLM_AUDIT_FILE,
        request_filename=LLM_AUDIT_REQUEST_FILE,
        proposal_filename=LLM_AUDIT_PROPOSAL_FILE
):
    """
    선택적 LLM 감사를 실행합니다.

    off는 완전 비활성화, manual은 현재 ChatGPT/Codex에서 검토할 집계 JSON을
    내보내며, audit는 API 결과를 기록만 합니다. capped는 실전 평가가 10회
    이상일 때에만 검증된 제안을 최대 5% 범위로 적용합니다.
    """

    mode = os.getenv("LOTTO_LLM_MODE", "off").strip().lower()

    if mode not in {"off", "manual", "audit", "capped"}:
        mode = "off"

    result = {
        "schema_version": "1.0",
        "mode": mode,
        "status": "disabled" if mode == "off" else "pending",
        "model": "",
        "applied": False,
        "effective_diversity_multiplier": 1.0,
        "proposal": None
    }

    if mode == "off":
        return result

    if mode == "manual":
        manual_request = {
            "instructions": (
                "번호를 예측하지 말고 집계지표만으로 분석기법과 데이터 품질을 "
                "감사하세요. 표본이 약하면 keep을 선택하세요."
            ),
            "response_schema": _llm_audit_json_schema(),
            "payload": payload
        }
        _write_json_file(request_filename, manual_request)
        result["status"] = "manual_request_saved"

        if os.path.exists(proposal_filename):
            try:
                with open(proposal_filename, "r", encoding="utf-8") as file:
                    raw_proposal = json.load(file)

                result["proposal"] = _validate_llm_audit_proposal(
                    raw_proposal,
                    allowed_methods
                )
                result["status"] = "manual_proposal_validated"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                result["status"] = "manual_proposal_rejected"

        _write_json_file(audit_filename, result)
        return result

    if not os.getenv("OPENAI_API_KEY"):
        result["status"] = "missing_api_key"
        _write_json_file(audit_filename, result)
        return result

    model = os.getenv("LOTTO_LLM_MODEL", "gpt-5.6-luna").strip()
    timeout_seconds = _safe_environment_float(
        "LOTTO_LLM_TIMEOUT",
        20.0,
        5.0,
        60.0
    )
    result["model"] = model

    try:
        from openai import OpenAI

        client = OpenAI(
            timeout=timeout_seconds,
            max_retries=1
        )
        response = client.responses.create(
            model=model,
            instructions=(
                "당신은 통계 앙상블 감사자입니다. 로또 번호를 예측하지 말고 "
                "제공된 집계지표만 근거로 분석기법 가중치와 데이터 품질을 "
                "감사하세요. 표본이 약하면 keep을 선택하세요."
            ),
            input=json.dumps(payload, ensure_ascii=False),
            max_output_tokens=1000,
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "lotto_method_audit",
                    "schema": _llm_audit_json_schema(),
                    "strict": True
                }
            },
            store=False
        )
        raw_proposal = json.loads(response.output_text)
        proposal = _validate_llm_audit_proposal(
            raw_proposal,
            allowed_methods
        )
        result["proposal"] = proposal
        result["status"] = "audit_completed"
        completed_draws = int(payload.get("completed_live_draws", 0))

        if (
            mode == "capped"
            and completed_draws >= LLM_MIN_COMPLETED_DRAWS
            and proposal["action"] == "adjust"
        ):
            result["applied"] = True
            result["status"] = "capped_adjustment_applied"
            result["effective_diversity_multiplier"] = proposal[
                "diversity_multiplier"
            ]
        elif mode == "capped":
            result["status"] = "capped_audit_only_insufficient_data"
    except Exception as error:
        # 오류 세부문에는 요청/자격증명 정보가 섞일 수 있어 유형만 기록합니다.
        result["status"] = "api_error_statistical_fallback"
        result["error_type"] = type(error).__name__

    _write_json_file(audit_filename, result)
    return result


def apply_llm_method_audit(method_performance_df, llm_audit):
    """검증·승인된 capped 제안만 통계 가중치에 최대 5% 반영합니다."""

    adjusted_df = method_performance_df.copy()
    adjusted_df["LLM상대조정"] = 0.0
    adjusted_df["LLM신뢰도"] = 0.0
    adjusted_df["LLM보정계수"] = 1.0

    if not llm_audit or not llm_audit.get("applied"):
        return adjusted_df

    proposal = llm_audit.get("proposal") or {}
    adjustment_map = {
        item["method"]: item
        for item in proposal.get("adjustments", [])
    }

    for index, row in adjusted_df.iterrows():
        adjustment = adjustment_map.get(str(row["분석기법"]))

        if not adjustment:
            continue

        delta = float(adjustment["relative_delta"])
        confidence = float(adjustment["confidence"])
        factor = 1.0 + delta * confidence
        adjusted_df.at[index, "LLM상대조정"] = delta
        adjusted_df.at[index, "LLM신뢰도"] = confidence
        adjusted_df.at[index, "LLM보정계수"] = factor

    adjusted_df["가중치"] = (
        pd.to_numeric(adjusted_df["가중치"], errors="coerce").fillna(0.0)
        * adjusted_df["LLM보정계수"]
    )
    total_weight = float(adjusted_df["가중치"].sum())

    if total_weight > 0:
        adjusted_df["가중치"] = adjusted_df["가중치"] / total_weight

    return adjusted_df


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

    # 같은 회차를 로직 변경 후 재생성하면 과거 가중치와 새 가중치가
    # 섞이지 않도록 해당 회차의 성과 묶음 전체를 새 결과로 교체합니다.
    if (
        not existing_records.empty
        and "예측대상회차" in existing_records.columns
    ):
        existing_target_draws = pd.to_numeric(
            existing_records["예측대상회차"],
            errors="coerce"
        )
        existing_records = existing_records[
            existing_target_draws != int(target_draw)
        ].copy()

    combined_records = pd.concat(
        [existing_records, new_records],
        ignore_index=True
    )
    combined_records = combined_records.drop_duplicates(
        subset=["예측대상회차", "분석기법"],
        keep="last"
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

    validate_history(df)
    latest_round = int(df["회차"].max())
    target_draw = latest_round + 1
    prediction_history = load_prediction_history(
        latest_history_round=latest_round,
        filename=filename
    )
    prediction_history, newly_evaluated_draws = (
        evaluate_prediction_history(prediction_history, df)
    )
    prediction_feedback, prediction_feedback_df = (
        calculate_prediction_feedback(prediction_history)
    )
    save_prediction_feedback(prediction_feedback_df)

    method_forecast_history = evaluate_method_forecast_history(df)
    live_method_feedback = calculate_live_method_feedback(
        method_forecast_history
    )

    current_prediction = prediction_history[
        prediction_history["예측회차"] == target_draw
    ].copy()
    generated_now = current_prediction.empty
    method_performance_df = None
    llm_audit = {
        "mode": "off",
        "status": "disabled",
        "applied": False,
        "effective_diversity_multiplier": 1.0
    }
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if generated_now:
        method_performance_df = backtest_analysis_methods(df)
        method_performance_df = apply_live_method_feedback(
            method_performance_df,
            live_method_feedback
        )
        llm_payload = build_llm_audit_payload(
            target_draw,
            method_performance_df,
            live_method_feedback,
            prediction_feedback
        )
        llm_audit = request_llm_weight_audit(
            llm_payload,
            allowed_methods=method_performance_df["분석기법"]
        )
        method_performance_df = apply_llm_method_audit(
            method_performance_df,
            llm_audit
        )
        generated_prediction = generate_prediction_sets(
            df,
            set_count=PREDICTION_SET_COUNT,
            target_draw=target_draw,
            method_performance_df=method_performance_df,
            prediction_feedback=prediction_feedback,
            llm_audit=llm_audit
        )

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

    target_forecasts = method_forecast_history[
        pd.to_numeric(
            method_forecast_history["예측대상회차"],
            errors="coerce"
        ) == target_draw
    ]
    stored_target_methods = set(
        target_forecasts["분석기법"].dropna().astype(str)
    )

    if stored_target_methods != set(ANALYSIS_METHODS):
        new_method_forecasts = create_method_forecast_records(
            df,
            target_draw=target_draw,
            created_at=created_at
        )
        method_forecast_history = save_method_forecast_records(
            new_method_forecasts
        )

    # API 키 없이 현재 ChatGPT/Codex에서 감사하려는 경우에는 기존 예측이
    # 있더라도 집계 JSON을 내보냅니다. 이 결과는 번호나 가중치에 자동 적용하지
    # 않으며, API 모드는 새 회차 생성 시에만 호출해 중복 비용을 막습니다.
    llm_mode = os.getenv("LOTTO_LLM_MODE", "off").strip().lower()

    if not generated_now and llm_mode == "manual":
        manual_performance_df = apply_live_method_feedback(
            backtest_analysis_methods(df),
            live_method_feedback
        )
        manual_payload = build_llm_audit_payload(
            target_draw,
            manual_performance_df,
            live_method_feedback,
            prediction_feedback
        )
        llm_audit = request_llm_weight_audit(
            manual_payload,
            allowed_methods=manual_performance_df["분석기법"]
        )

    final_audit = final_review(
        df, current_prediction, prediction_history, get_lotto_draw,
        os.path.dirname(os.path.abspath(filename)),
        expected_count=PREDICTION_SET_COUNT
    )
    if final_audit["status"] in {"validation_error", "source_mismatch", "reject"}:
        raise ValueError("최종 번호 검증 실패: GPT최종검증결과.json을 확인하세요.")
    print(f"GPT 최종 번호 검증: {final_audit['status']}")
    if not final_audit["finalized"]:
        print("아래 번호는 검증 대기 후보이며 GPT 최종 확정 번호가 아닙니다.")

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
        ["예측회차", "세트"]
        + NUMBER_COLUMNS
        + ["종합점수"]
    )

    print(
        current_prediction[display_columns].to_string(index=False)
    )

    if generated_now and method_performance_df is not None:
        print()
        print("분석기법별 백테스트·실전 피드백 및 앙상블 가중치")
        performance_columns = [
            "분석기법",
            "백테스트회차수",
            "TOP6평균적중",
            "TOP12평균적중",
            "최근2회성능지수",
            "보강성능지수",
            "실전피드백회차수",
            "실전보정계수",
            "LLM보정계수",
            "가중치"
        ]
        performance_display = method_performance_df[
            performance_columns
        ].copy()
        performance_display["가중치"] = (
            performance_display["가중치"] * 100
        ).round(2)
        performance_display = performance_display.rename(
            columns={"가중치": "가중치(%)"}
        )
        print(performance_display.to_string(index=False))
    else:
        print("이미 저장된 예측이 있어 새 세트를 중복 생성하지 않았습니다.")

    print()
    print(
        "실전 예측 피드백 : "
        f"평가 {prediction_feedback['evaluated_draws']}회, "
        f"평균 적중 {prediction_feedback['mean_main_hits']:.3f}개, "
        f"Brier skill {prediction_feedback['mean_brier_skill']:.4f}, "
        "다양성 보강계수 "
        f"{prediction_feedback['diversity_penalty_multiplier']:.4f}"
    )

    if llm_audit.get("mode") != "off":
        llm_status = llm_audit.get("status", "unknown")
        print(
            "LLM 분석 감사 : "
            f"{llm_status}"
        )

        if llm_status.startswith("manual_"):
            print(f"수동 감사 요청 파일 : {LLM_AUDIT_REQUEST_FILE}")
        elif llm_status == "missing_api_key":
            print(
                "OPENAI_API_KEY가 없어 통계 로직만 사용했습니다. "
                "키는 로컬 환경변수에만 설정하세요."
            )

    print()
    print(f"회차별 예상번호/평가 이력 저장 완료 : {filename}")

    if generated_now:
        print(f"분석기법 성과 이력 저장 완료 : {METHOD_PERFORMANCE_FILE}")

    print(f"실전 예측 피드백 저장 완료 : {PREDICTION_FEEDBACK_FILE}")
    print(f"분석기법 예측 스냅샷 저장 완료 : {METHOD_FORECAST_FILE}")

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
        filename=PREDICTION_FILE
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
