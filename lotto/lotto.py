# ============================================================
# 로또 6/45 과거 당첨번호 관리 + 고급 앙상블 분석 + 예상번호 생성
#
# 주요 기능
# 1. 사용자가 지정한 조회 회차의 당첨번호 조회
# 2. '과거로또 당첨번호.csv'의 마지막 회차 확인
# 3. 조회 회차가 CSV 마지막 회차보다 큰 경우 차이 회차만 추가
# 4. 본번호 6개 + 보너스 번호를 전 회차 분석에 활용
# 5. 본번호와 보너스 번호의 역할을 구분해 서로 다른 가중치 적용
# 6. 전 회차 빈도 + 시간감쇠 최근성 + 미출현 간격 분석
# 7. 번호 간 동시출현(pair co-occurrence) 및 시간가중 동시출현 분석
# 8. 본번호-보너스 번호 연관성까지 보조 신호로 반영
# 9. 홀짝/합계/고저/연속번호/번호범위 등 1등 조합 구조 분석
# 10. 수만 개 후보 조합을 Monte Carlo 방식으로 생성 후 조합 자체를 재점수화
# 11. 서로 지나치게 겹치지 않는 상위 10개 조합을 선택
# 12. 현재 최종 회차 + 1을 예측회차로 부여해 '당첨예상번호.csv'에 누적 저장
# 13. 다음 회차 실제 결과가 들어오면 기존 예상번호와 자동 비교/등수 평가
# 14. 존재하지 않거나 아직 발표되지 않은 회차 조회 시 오류 없이 안전하게 종료
#
# 중요:
# - 로또 추첨은 독립적인 무작위 추첨을 전제로 하므로,
#   과거 데이터 분석이 실제 1등 당첨 확률을 높인다고 보장할 수 없습니다.
# - 본 코드는 단순 빈도 추출보다 다양한 통계적/시계열/관계 특징을
#   결합하여 후보 조합을 체계적으로 순위화하는 휴리스틱 분석 도구입니다.
#
# 사용 방법
# - 아래 QUERY_ROUND에 실제 조회할 로또 회차를 입력합니다.
#
# 필요 라이브러리
# pip install requests pandas
# ============================================================

import os
import requests
import pandas as pd
import time
import random
import math
from collections import Counter
from itertools import combinations


# ============================================================
# 1. 기본 설정
# ============================================================
CSV_FILE = "과거로또 당첨번호.csv"
PREDICTION_FILE = "당첨예상번호.csv"
# 실제 조회할 로또 회차를 입력하세요.
# 예: CSV 마지막 회차가 1182회이고 QUERY_ROUND가 1185이면
#     1183회, 1184회, 1185회만 추가합니다.
QUERY_ROUND = 0  # 예: 1185
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
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin"
}

# 연결과 쿠키를 회차별 요청 사이에서 재사용합니다.
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update(HEADERS)


# ============================================================
# 2. 특정 회차 조회
# ============================================================

def get_lotto_draw(draw_no, max_retries=3, retry_delay=1.0):
    """
    특정 회차의 로또 데이터를 조회합니다.

    존재하지 않는 회차, 아직 발표되지 않은 미래 회차,
    빈 응답, JSON 오류, 필드 누락이 발생해도 예외를 밖으로
    던지지 않고 None을 반환합니다.
    """

    try:
        draw_no = int(draw_no)

        if draw_no < 1:
            print(f"[조회 제외] 유효하지 않은 회차: {draw_no}")
            return None

    except (TypeError, ValueError):
        print(f"[조회 제외] 회차 값이 숫자가 아닙니다: {draw_no}")
        return None

    params = {
        "srchLtEpsd": draw_no
    }

    try:
        response = None

        for attempt in range(1, max_retries + 1):
            try:
                response = HTTP_SESSION.get(
                    API_URL,
                    params=params,
                    timeout=10
                )

                # 요청 제한과 서버 오류만 재시도합니다. 4xx는 재시도해도
                # 결과가 달라질 가능성이 낮으므로 즉시 아래에서 처리합니다.
                if response.status_code != 429 and response.status_code < 500:
                    break

                if attempt < max_retries:
                    print(
                        f"[조회 재시도] {draw_no}회 "
                        f"HTTP {response.status_code} ({attempt}/{max_retries})"
                    )
                    time.sleep(retry_delay * attempt)

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError
            ) as e:
                if attempt >= max_retries:
                    raise

                print(
                    f"[조회 재시도] {draw_no}회 {type(e).__name__} "
                    f"({attempt}/{max_retries})"
                )
                time.sleep(retry_delay * attempt)

        if response is None:
            return None

        # 404/500 등 HTTP 오류도 프로그램 중단 없이 처리
        if response.status_code != 200:
            print(
                f"[조회 실패] {draw_no}회 HTTP 상태코드: "
                f"{response.status_code}"
            )
            return None

        # 응답 내용이 비어 있으면 존재하지 않는 회차로 처리
        if not response.text or not response.text.strip():
            print(
                f"[조회 없음] {draw_no}회 응답이 비어 있습니다."
            )
            return None

        try:
            data = response.json()
        except ValueError:
            print(
                f"[조회 없음] {draw_no}회 응답이 JSON 형식이 아닙니다."
            )
            return None

        if not isinstance(data, dict):
            print(
                f"[조회 없음] {draw_no}회 응답 형식이 올바르지 않습니다."
            )
            return None

        data_block = data.get("data")

        if not isinstance(data_block, dict):
            print(
                f"[조회 없음] {draw_no}회 데이터가 존재하지 않습니다."
            )
            return None

        result_list = data_block.get("list", [])

        if not isinstance(result_list, list) or not result_list:
            print(
                f"[조회 없음] {draw_no}회는 아직 존재하지 않거나 "
                f"당첨결과가 발표되지 않았습니다."
            )
            return None

        item = result_list[0]

        if not isinstance(item, dict):
            print(
                f"[조회 없음] {draw_no}회 결과 형식이 올바르지 않습니다."
            )
            return None

        try:
            returned_draw = int(item.get("ltEpsd", 0))
        except (TypeError, ValueError):
            print(
                f"[조회 없음] {draw_no}회 회차 정보를 확인할 수 없습니다."
            )
            return None

        # 요청 회차와 응답 회차가 다르면 잘못된 응답으로 처리
        if returned_draw != draw_no:
            print(
                f"[조회 없음] 요청 {draw_no}회 / 응답 {returned_draw}회 "
                f"불일치"
            )
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

        for field in required_number_fields:
            if item.get(field) in (None, ""):
                print(
                    f"[조회 없음] {draw_no}회 당첨번호 필드 "
                    f"({field})가 비어 있습니다."
                )
                return None

        try:
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

        except (TypeError, ValueError) as e:
            print(
                f"[파싱 오류] {draw_no}회 당첨번호 변환 실패: {e}"
            )
            return None

        # 번호 범위 검증
        main_numbers = [
            result["번호1"],
            result["번호2"],
            result["번호3"],
            result["번호4"],
            result["번호5"],
            result["번호6"]
        ]

        if (
            len(set(main_numbers)) != 6
            or any(n < 1 or n > 45 for n in main_numbers)
            or result["보너스"] < 1
            or result["보너스"] > 45
            or result["보너스"] in main_numbers
        ):
            print(
                f"[검증 실패] {draw_no}회 당첨번호 값이 유효하지 않습니다."
            )
            return None

        return result

    except requests.exceptions.Timeout:
        print(
            f"[네트워크 오류] {draw_no}회 조회 시간 초과"
        )
        return None

    except requests.exceptions.ConnectionError:
        print(
            f"[네트워크 오류] {draw_no}회 서버 연결 실패"
        )
        return None

    except requests.exceptions.RequestException as e:
        print(
            f"[네트워크 오류] {draw_no}회 : {e}"
        )
        return None

    except Exception as e:
        print(
            f"[예외 처리] {draw_no}회 조회 중 예상치 못한 오류: {e}"
        )
        return None


# ============================================================
# 기존 CSV 로드
# ============================================================

def load_history_csv():
    """
    기존 lotto_history.csv를 읽어옵니다.
    필수 열과 번호 범위를 검증하고 중복 회차를 제거합니다.
    """

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            CSV_FILE,
            encoding="utf-8-sig"
        )

        required_columns = [
            "회차", "번호1", "번호2", "번호3",
            "번호4", "번호5", "번호6", "보너스"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if df.empty:
            return pd.DataFrame()

        if missing_columns:
            print(
                "[CSV 검증 오류] 필수 열이 없습니다: "
                + ", ".join(missing_columns)
            )
            return pd.DataFrame()

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        original_count = len(df)
        df = df.dropna(subset=required_columns).copy()

        for col in required_columns:
            df[col] = df[col].astype(int)

        number_cols = required_columns[1:7]
        valid_mask = (
            df["회차"].ge(1)
            & df[number_cols].ge(1).all(axis=1)
            & df[number_cols].le(45).all(axis=1)
            & df["보너스"].between(1, 45)
            & df[number_cols].nunique(axis=1).eq(6)
            & ~df.apply(
                lambda row: row["보너스"] in row[number_cols].values,
                axis=1
            )
        )
        df = df.loc[valid_mask].copy()

        invalid_count = original_count - len(df)
        if invalid_count:
            print(f"[CSV 검증] 잘못된 행 {invalid_count}개를 제외했습니다.")

        duplicate_count = int(df.duplicated("회차", keep="last").sum())
        if duplicate_count:
            print(f"[CSV 검증] 중복 회차 {duplicate_count}개를 제거했습니다.")

        df = (
            df.drop_duplicates("회차", keep="last")
              .sort_values("회차")
              .reset_index(drop=True)
        )

        return df

    except Exception as e:
        print(f"[CSV 로드 오류] {e}")
        return pd.DataFrame()


# ============================================================
# 3. 조회 회차와 CSV 마지막 회차 비교 및 신규 회차 추가
# ============================================================

def collect_all_lotto(
        query_round,
        start_draw=1,
        sleep_time=0.10
):
    """
    사용자가 조회한 회차(query_round)와
    '과거로또 당첨번호.csv'의 마지막 회차를 비교합니다.

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
    # 1. 조회 회차 유효성 확인
    # --------------------------------------------------------
    query_round = int(query_round)

    if query_round < 1:
        print("조회 회차를 1 이상의 값으로 설정해야 합니다.")
        return load_history_csv()

    # --------------------------------------------------------
    # 2. 기존 CSV 로드 및 마지막 회차 확인
    # --------------------------------------------------------
    existing_df = load_history_csv()

    if existing_df.empty:
        csv_last_round = 0
        print(f"기존 CSV 없음 : {CSV_FILE}")
    else:
        csv_last_round = int(existing_df["회차"].max())
        print(f"CSV 마지막 회차 : {csv_last_round}회")

    print(f"조회 회차          : {query_round}회")

    # --------------------------------------------------------
    # 3. 조회 회차와 CSV 마지막 회차 비교
    # --------------------------------------------------------
    if query_round <= csv_last_round:
        print()
        print(
            f"조회 회차({query_round}회)가 "
            f"CSV 마지막 회차({csv_last_round}회)보다 크지 않습니다."
        )
        print("추가할 회차가 없습니다.")
        return existing_df

    # --------------------------------------------------------
    # 4. 차이 구간 계산
    # --------------------------------------------------------
    first_new_round = (
        start_draw
        if csv_last_round == 0
        else csv_last_round + 1
    )

    target_rounds = list(
        range(
            first_new_round,
            query_round + 1
        )
    )

    print()
    print(
        f"추가 대상 회차 : "
        f"{first_new_round}회 ~ {query_round}회"
    )
    print(
        f"추가 대상 건수 : "
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
                f"{draw_no}회 조회 결과 없음"
            )
            print(
                f"{draw_no}회가 아직 존재하지 않거나 "
                f"당첨결과가 발표되지 않은 것으로 판단하여 "
                f"이후 회차 조회를 중단합니다."
            )
            break

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
    print(f"요청 조회 회차     : {query_round}회")
    print(f"실제 추가 마지막회차 : {int(new_df['회차'].max())}회")
    print(f"신규 추가 회차 수  : {len(new_df)}개")
    print(f"현재 마지막 회차   : {int(combined_df['회차'].max())}회")
    print("=" * 70)

    return combined_df


# ============================================================
# 4. 전 회차 분석용 파생변수 생성
# ============================================================

def create_features(df):

    # 호출자가 보유한 원본 DataFrame을 변경하지 않습니다.
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
# 5. 보너스 제외 본번호 6개의 전체 출현빈도 계산
# ============================================================

def calculate_frequency(df):
    """
    본번호 6개와 보너스 번호의 전 회차 출현빈도를 함께 계산합니다.

    본번호와 보너스는 역할이 다르므로 별도 집계하고,
    화면 참고용 통합가중점수에서는 보너스 빈도를 30%만 반영합니다.
    """

    number_cols = [
        "번호1", "번호2", "번호3",
        "번호4", "번호5", "번호6"
    ]

    main_numbers = df[number_cols].values.flatten()
    bonus_numbers = df["보너스"].values.flatten()

    main_counts = Counter(main_numbers)
    bonus_counts = Counter(bonus_numbers)

    total = max(len(df), 1)

    frequency_df = pd.DataFrame({
        "번호": range(1, 46),
        "출현횟수": [
            main_counts.get(number, 0)
            for number in range(1, 46)
        ],
        "보너스출현횟수": [
            bonus_counts.get(number, 0)
            for number in range(1, 46)
        ]
    })

    frequency_df["본번호출현율(%)"] = (
        frequency_df["출현횟수"] / total * 100
    ).round(2)

    frequency_df["보너스출현율(%)"] = (
        frequency_df["보너스출현횟수"] / total * 100
    ).round(2)

    # 화면 참고용 통합지표
    frequency_df["통합가중출현점수"] = (
        frequency_df["출현횟수"]
        + frequency_df["보너스출현횟수"] * 0.30
    ).round(2)

    return frequency_df


# ============================================================
# 6. 지정 구간 출현빈도 계산 (화면 분석 보조용)
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
# 7. 전 회차 기준 번호별 마지막 출현 이후 경과회차
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
        df.sort_values("회차")
          .drop_duplicates("회차", keep="last")
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

            # CSV에 중간 회차가 빠진 경우에도 실제 보유 데이터 행을
            # 기준으로 경과 횟수를 계산합니다.
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
# 8. 전 회차 분석 기반 이번 주 예상번호 10세트 생성
# ============================================================

def generate_prediction_sets(
        df,
        set_count=10,
        random_seed=None,
        candidate_count=30000
):
    """
    본번호 6개와 보너스 번호까지 전 회차 분석에 포함하여
    1등 후보 조합 10세트를 생성합니다.

    단순 최대 빈도 번호 추출이 아니라 다음 신호를 앙상블합니다.

    [번호 단위 분석]
    1. 본번호 전 회차 출현빈도
    2. 보너스 번호 전 회차 출현빈도
    3. 시간감쇠(Exponential Decay) 최근성
    4. 현재 미출현 간격 / 평균 출현간격
    5. 번호 간 동시출현 그래프 중심성

    [번호 관계 분석]
    6. 본번호 6개 간 pair co-occurrence
    7. 시간가중 pair co-occurrence
    8. 본번호-보너스 번호 연관성

    [조합 구조 분석]
    9. 합계 분포
    10. 홀짝 개수
    11. 고번호(23~45) 개수
    12. 연속번호 pair 개수
    13. 최소~최대 번호 범위

    이후 candidate_count개의 후보 조합을 생성하고
    각 조합 자체를 다시 평가한 뒤 서로 지나치게 겹치지 않는
    상위 10개 조합을 선택합니다.

    주의:
    무작위 추첨의 실제 미래 당첨 확률을 높인다고 보장하는 모델은 아닙니다.
    """

    if df.empty:
        return pd.DataFrame()

    df = (
        df.sort_values("회차")
          .reset_index(drop=True)
          .copy()
    )

    main_cols = [
        "번호1", "번호2", "번호3",
        "번호4", "번호5", "번호6"
    ]

    latest_draw = int(df["회차"].max())

    # 결과 재현성을 위해 별도 seed가 없으면 최신 회차 기반 seed 사용
    if random_seed is None:
        random_seed = latest_draw * 1009 + len(df)

    rng = random.Random(random_seed)

    # ========================================================
    # A. 전 회차 기본 빈도
    # ========================================================
    main_counts = Counter(
        df[main_cols].values.flatten()
    )

    bonus_counts = Counter(
        df["보너스"].values.flatten()
    )

    # ========================================================
    # B. 시간 감쇠 점수
    # 최근 회차에 더 높은 가중치
    # half-life = 120회
    # ========================================================
    half_life = 120.0
    decay_lambda = math.log(2.0) / half_life

    main_decay = Counter()
    bonus_decay = Counter()

    # pair 관련 통계
    pair_counts = Counter()
    pair_decay = Counter()
    main_bonus_counts = Counter()
    main_bonus_decay = Counter()

    total_rows = len(df)

    for row_idx, row in df.iterrows():
        age = total_rows - 1 - row_idx
        decay_weight = math.exp(-decay_lambda * age)

        main_numbers = sorted(
            int(row[col])
            for col in main_cols
        )
        bonus = int(row["보너스"])

        for number in main_numbers:
            main_decay[number] += decay_weight

        bonus_decay[bonus] += decay_weight

        # 본번호끼리의 pair 동시출현
        for a, b in combinations(main_numbers, 2):
            pair_counts[(a, b)] += 1
            pair_decay[(a, b)] += decay_weight

        # 본번호-보너스 관계
        for number in main_numbers:
            key = tuple(sorted((number, bonus)))
            main_bonus_counts[key] += 1
            main_bonus_decay[key] += decay_weight

    # ========================================================
    # C. 번호별 출현 간격 / 미출현 압력
    # ========================================================
    current_gap = {}
    mean_gap = {}

    for number in range(1, 46):
        rounds = df.loc[
            df[main_cols].eq(number).any(axis=1),
            "회차"
        ].astype(int).tolist()

        if rounds:
            current_gap[number] = latest_draw - rounds[-1]
        else:
            current_gap[number] = latest_draw

        if len(rounds) >= 2:
            gaps = [
                rounds[i] - rounds[i - 1]
                for i in range(1, len(rounds))
            ]
            mean_gap[number] = sum(gaps) / len(gaps)
        else:
            mean_gap[number] = max(current_gap[number], 1)

    # gap ratio는 너무 큰 값이 지배하지 않게 cap
    gap_ratio = {
        n: min(
            current_gap[n] / max(mean_gap[n], 1e-9),
            3.0
        )
        for n in range(1, 46)
    }

    # ========================================================
    # D. 번호 관계 그래프 중심성
    # 별도 graph 라이브러리 없이 weighted degree 사용
    # ========================================================
    pair_degree = Counter()
    pair_decay_degree = Counter()

    for (a, b), value in pair_counts.items():
        pair_degree[a] += value
        pair_degree[b] += value

    for (a, b), value in pair_decay.items():
        pair_decay_degree[a] += value
        pair_decay_degree[b] += value

    # ========================================================
    # E. 정규화 함수
    # ========================================================
    def normalize_dict(data, numbers=range(1, 46)):
        values = [float(data.get(n, 0.0)) for n in numbers]
        lo = min(values) if values else 0.0
        hi = max(values) if values else 1.0

        if hi == lo:
            return {n: 0.5 for n in numbers}

        return {
            n: (float(data.get(n, 0.0)) - lo) / (hi - lo)
            for n in numbers
        }

    main_freq_n = normalize_dict(main_counts)
    bonus_freq_n = normalize_dict(bonus_counts)
    main_decay_n = normalize_dict(main_decay)
    bonus_decay_n = normalize_dict(bonus_decay)
    gap_ratio_n = normalize_dict(gap_ratio)
    pair_degree_n = normalize_dict(pair_degree)
    pair_decay_degree_n = normalize_dict(pair_decay_degree)

    # ========================================================
    # F. 번호 단위 앙상블 점수
    #
    # 보너스 번호는 최종 6개 당첨 본번호와 역할이 다르므로
    # 본번호보다 낮은 비중으로만 보조 신호에 반영합니다.
    # ========================================================
    number_scores = {}

    for number in range(1, 46):
        score = (
            main_freq_n[number] * 0.24
            + main_decay_n[number] * 0.24
            + gap_ratio_n[number] * 0.16
            + pair_degree_n[number] * 0.12
            + pair_decay_degree_n[number] * 0.12
            + bonus_freq_n[number] * 0.06
            + bonus_decay_n[number] * 0.06
        )

        number_scores[number] = score

    # ========================================================
    # G. pair 정규화
    # ========================================================
    def normalize_pair_counter(counter):
        if not counter:
            return {}

        max_value = max(counter.values())

        if max_value <= 0:
            return {
                key: 0.0
                for key in counter
            }

        return {
            key: float(value) / float(max_value)
            for key, value in counter.items()
        }

    pair_counts_n = normalize_pair_counter(pair_counts)
    pair_decay_n = normalize_pair_counter(pair_decay)
    main_bonus_counts_n = normalize_pair_counter(main_bonus_counts)
    main_bonus_decay_n = normalize_pair_counter(main_bonus_decay)

    # ========================================================
    # H. 과거 1등 조합 구조 분포 학습
    # ========================================================
    historical_sums = []
    historical_ranges = []
    odd_counter = Counter()
    high_counter = Counter()
    consecutive_counter = Counter()

    for _, row in df.iterrows():
        nums = sorted(
            int(row[col])
            for col in main_cols
        )

        total_sum = sum(nums)
        number_range = nums[-1] - nums[0]
        odd_count = sum(n % 2 == 1 for n in nums)
        high_count = sum(n >= 23 for n in nums)
        consecutive_count = sum(
            nums[i + 1] - nums[i] == 1
            for i in range(5)
        )

        historical_sums.append(total_sum)
        historical_ranges.append(number_range)
        odd_counter[odd_count] += 1
        high_counter[high_count] += 1
        consecutive_counter[consecutive_count] += 1

    def mean_std(values):
        if not values:
            return 0.0, 1.0

        mean_value = sum(values) / len(values)

        if len(values) < 2:
            return mean_value, 1.0

        variance = sum(
            (v - mean_value) ** 2
            for v in values
        ) / (len(values) - 1)

        std = math.sqrt(max(variance, 1e-9))
        return mean_value, std

    sum_mean, sum_std = mean_std(historical_sums)
    range_mean, range_std = mean_std(historical_ranges)

    def empirical_probability(counter, value):
        # Laplace smoothing
        total = sum(counter.values())
        category_count = max(len(counter), 1)

        return (
            counter.get(value, 0) + 1
        ) / (
            total + category_count
        )

    max_odd_p = max(
        empirical_probability(odd_counter, k)
        for k in range(0, 7)
    )
    max_high_p = max(
        empirical_probability(high_counter, k)
        for k in range(0, 7)
    )
    max_consecutive_p = max(
        empirical_probability(consecutive_counter, k)
        for k in range(0, 6)
    )

    # ========================================================
    # I. 조합 점수 함수
    # ========================================================
    def candidate_score(selected):
        selected = sorted(selected)

        # 1) 번호 단위 점수
        individual_score = sum(
            number_scores[n]
            for n in selected
        ) / 6.0

        # 2) 번호간 관계 점수
        pair_values = []
        bonus_relation_values = []

        for a, b in combinations(selected, 2):
            key = (min(a, b), max(a, b))

            main_pair_score = (
                pair_counts_n.get(key, 0.0) * 0.45
                + pair_decay_n.get(key, 0.0) * 0.45
            )

            # 두 번호 중 하나가 과거 보너스였던 관계도 낮은 비중 반영
            bonus_relation_score = (
                main_bonus_counts_n.get(key, 0.0) * 0.05
                + main_bonus_decay_n.get(key, 0.0) * 0.05
            )

            pair_values.append(main_pair_score)
            bonus_relation_values.append(bonus_relation_score)

        pair_score = (
            sum(pair_values) / len(pair_values)
            if pair_values else 0.0
        )

        bonus_relation_score = (
            sum(bonus_relation_values) / len(bonus_relation_values)
            if bonus_relation_values else 0.0
        )

        relation_score = pair_score + bonus_relation_score

        # 3) 과거 1등 조합 구조 유사도
        total_sum = sum(selected)
        number_range = selected[-1] - selected[0]
        odd_count = sum(n % 2 == 1 for n in selected)
        high_count = sum(n >= 23 for n in selected)
        consecutive_count = sum(
            selected[i + 1] - selected[i] == 1
            for i in range(5)
        )

        sum_z = (total_sum - sum_mean) / max(sum_std, 1e-9)
        range_z = (number_range - range_mean) / max(range_std, 1e-9)

        # 정규분포 중심에 가까울수록 높은 점수
        sum_score = math.exp(-0.5 * (sum_z ** 2))
        range_score = math.exp(-0.5 * (range_z ** 2))

        odd_score = (
            empirical_probability(odd_counter, odd_count)
            / max_odd_p
        )

        high_score = (
            empirical_probability(high_counter, high_count)
            / max_high_p
        )

        consecutive_score = (
            empirical_probability(
                consecutive_counter,
                consecutive_count
            )
            / max_consecutive_p
        )

        structure_score = (
            sum_score * 0.30
            + range_score * 0.20
            + odd_score * 0.20
            + high_score * 0.20
            + consecutive_score * 0.10
        )

        # 최종 조합 앙상블
        final_score = (
            individual_score * 0.45
            + relation_score * 0.30
            + structure_score * 0.25
        )

        return final_score

    # ========================================================
    # J. Monte Carlo 후보 생성
    # ========================================================
    numbers = list(range(1, 46))

    # 너무 낮은 점수도 완전히 배제하지 않음
    weights = [
        0.05 + number_scores[n]
        for n in numbers
    ]

    candidates = {}
    attempts = 0
    max_attempts = candidate_count * 3

    while (
        len(candidates) < candidate_count
        and attempts < max_attempts
    ):
        attempts += 1

        available_numbers = numbers.copy()
        available_weights = weights.copy()
        selected = []

        while len(selected) < 6:
            chosen = rng.choices(
                available_numbers,
                weights=available_weights,
                k=1
            )[0]

            idx = available_numbers.index(chosen)

            selected.append(chosen)
            available_numbers.pop(idx)
            available_weights.pop(idx)

        selected = tuple(sorted(selected))

        if selected in candidates:
            continue

        candidates[selected] = candidate_score(selected)

    # 후보 점수 정렬
    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # ========================================================
    # K. 상위 조합 중 다양성 확보
    # 동일한 번호가 과도하게 반복되지 않도록 최대 4개 겹침 허용
    # ========================================================
    selected_sets = []

    for numbers_tuple, score in ranked:
        if len(selected_sets) >= set_count:
            break

        candidate_set = set(numbers_tuple)

        too_similar = False

        for existing_numbers, _ in selected_sets:
            overlap = len(
                candidate_set.intersection(existing_numbers)
            )

            if overlap >= 5:
                too_similar = True
                break

        if too_similar:
            continue

        selected_sets.append(
            (numbers_tuple, score)
        )

    # 혹시 10세트 미만이면 중복만 피해서 보완
    if len(selected_sets) < set_count:
        used = {
            tuple(numbers_tuple)
            for numbers_tuple, _ in selected_sets
        }

        for numbers_tuple, score in ranked:
            if len(selected_sets) >= set_count:
                break

            if numbers_tuple in used:
                continue

            used.add(numbers_tuple)
            selected_sets.append(
                (numbers_tuple, score)
            )

    # ========================================================
    # L. 결과 DataFrame
    # ========================================================
    rows = []

    for idx, (numbers_tuple, score) in enumerate(
        selected_sets,
        start=1
    ):
        rows.append({
            "세트": idx,
            "번호1": numbers_tuple[0],
            "번호2": numbers_tuple[1],
            "번호3": numbers_tuple[2],
            "번호4": numbers_tuple[3],
            "번호5": numbers_tuple[4],
            "번호6": numbers_tuple[5],
            "종합점수": round(score, 6)
        })

    return pd.DataFrame(rows)

def evaluate_prediction_history(
        history_df,
        filename=PREDICTION_FILE
):
    """
    기존 '당첨예상번호.csv'에 저장된 과거 예상번호 중
    실제 당첨결과가 발표된 회차를 찾아 자동 비교/평가합니다.

    평가 규칙:
    - 1등: 본번호 6개 일치
    - 2등: 본번호 5개 + 보너스 번호 일치
    - 3등: 본번호 5개 일치
    - 4등: 본번호 4개 일치
    - 5등: 본번호 3개 일치
    - 그 외: 낙첨

    비교 결과는 같은 '당첨예상번호.csv'에 누적 보존합니다.
    """

    if history_df is None or history_df.empty:
        print(
            "실제 당첨번호 데이터가 없어 예상번호 비교를 건너뜁니다."
        )
        return pd.DataFrame()

    if "회차" not in history_df.columns:
        print(
            "실제 당첨번호 데이터에 '회차' 컬럼이 없어 "
            "예상번호 비교를 건너뜁니다."
        )
        return pd.DataFrame()

    if not os.path.exists(filename):
        print("기존 예상번호 파일이 없어 비교할 과거 예측이 없습니다.")
        return pd.DataFrame()

    try:
        prediction_df = pd.read_csv(
            filename,
            encoding="utf-8-sig"
        )
    except Exception as e:
        print(f"[예상번호 CSV 로드 오류] {e}")
        return pd.DataFrame()

    if prediction_df.empty:
        return prediction_df

    # 이전 버전 파일은 예측회차가 없으므로 자동 비교가 불가능함
    if "예측회차" not in prediction_df.columns:
        print()
        print(
            "기존 당첨예상번호.csv에는 '예측회차' 컬럼이 없어 "
            "과거 예측과 실제 회차를 자동 매칭할 수 없습니다."
        )
        print(
            "이번 실행부터 생성되는 예상번호에는 예측회차를 기록하고 "
            "다음 회차부터 자동 비교합니다."
        )
        return prediction_df

    prediction_df["예측회차"] = pd.to_numeric(
        prediction_df["예측회차"],
        errors="coerce"
    )

    # 평가 결과 컬럼 준비
    result_columns = [
        "실제번호1", "실제번호2", "실제번호3",
        "실제번호4", "실제번호5", "실제번호6",
        "실제보너스",
        "본번호일치수",
        "보너스일치",
        "총일치수",
        "당첨등수",
        "평가상태"
    ]

    for col in result_columns:
        if col not in prediction_df.columns:
            prediction_df[col] = ""

    main_cols = [
        "번호1", "번호2", "번호3",
        "번호4", "번호5", "번호6"
    ]

    actual_lookup = {}

    for _, row in history_df.iterrows():
        draw = int(row["회차"])

        actual_lookup[draw] = {
            "main": {
                int(row[col])
                for col in main_cols
            },
            "ordered": [
                int(row[col])
                for col in main_cols
            ],
            "bonus": int(row["보너스"])
        }

    newly_evaluated_rounds = set()

    for idx, row in prediction_df.iterrows():

        if pd.isna(row["예측회차"]):
            continue

        prediction_round = int(row["예측회차"])

        if prediction_round not in actual_lookup:
            # 아직 실제 결과가 발표되지 않은 미래 회차
            if str(row.get("평가상태", "")).strip() == "":
                prediction_df.at[idx, "평가상태"] = "대기"
            continue

        # 이미 평가 완료된 행은 다시 계산하지 않아도 됨
        if str(row.get("평가상태", "")).strip() == "평가완료":
            continue

        actual = actual_lookup[prediction_round]

        predicted_numbers = {
            int(row[col])
            for col in main_cols
        }

        main_match_count = len(
            predicted_numbers.intersection(
                actual["main"]
            )
        )

        bonus_match = int(
            actual["bonus"] in predicted_numbers
        )

        total_match_count = (
            main_match_count + bonus_match
        )

        # 로또 6/45 등수 판정
        if main_match_count == 6:
            prize_rank = "1등"
        elif main_match_count == 5 and bonus_match == 1:
            prize_rank = "2등"
        elif main_match_count == 5:
            prize_rank = "3등"
        elif main_match_count == 4:
            prize_rank = "4등"
        elif main_match_count == 3:
            prize_rank = "5등"
        else:
            prize_rank = "낙첨"

        for n_idx, number in enumerate(
            actual["ordered"],
            start=1
        ):
            prediction_df.at[
                idx,
                f"실제번호{n_idx}"
            ] = number

        prediction_df.at[
            idx,
            "실제보너스"
        ] = actual["bonus"]

        prediction_df.at[
            idx,
            "본번호일치수"
        ] = main_match_count

        prediction_df.at[
            idx,
            "보너스일치"
        ] = bonus_match

        prediction_df.at[
            idx,
            "총일치수"
        ] = total_match_count

        prediction_df.at[
            idx,
            "당첨등수"
        ] = prize_rank

        prediction_df.at[
            idx,
            "평가상태"
        ] = "평가완료"

        newly_evaluated_rounds.add(
            prediction_round
        )

    # 회차/세트 순 정렬
    sort_cols = [
        col
        for col in ["예측회차", "세트"]
        if col in prediction_df.columns
    ]

    if sort_cols:
        prediction_df = (
            prediction_df
            .sort_values(sort_cols)
            .reset_index(drop=True)
        )

    prediction_df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    # 새로 평가된 회차 요약
    for prediction_round in sorted(
        newly_evaluated_rounds
    ):
        round_df = prediction_df[
            prediction_df["예측회차"]
            == prediction_round
        ].copy()

        if round_df.empty:
            continue

        round_df["본번호일치수_num"] = pd.to_numeric(
            round_df["본번호일치수"],
            errors="coerce"
        ).fillna(0)

        round_df["보너스일치_num"] = pd.to_numeric(
            round_df["보너스일치"],
            errors="coerce"
        ).fillna(0)

        best_match = int(
            round_df["본번호일치수_num"].max()
        )

        best_bonus = int(
            round_df.loc[
                round_df["본번호일치수_num"]
                == best_match,
                "보너스일치_num"
            ].max()
        )

        winning_sets = round_df[
            round_df["당첨등수"]
            .isin(["1등", "2등", "3등", "4등", "5등"])
        ]

        actual_row = actual_lookup[prediction_round]

        print()
        print("=" * 80)
        print(
            f"{prediction_round}회 예상번호 vs 실제 당첨번호 비교"
        )
        print("=" * 80)
        print(
            "실제 당첨번호 : "
            + ", ".join(
                str(n)
                for n in actual_row["ordered"]
            )
            + f" + 보너스 {actual_row['bonus']}"
        )
        print(
            f"10세트 중 최고 본번호 일치 : {best_match}개"
        )
        print(
            f"해당 최고 세트의 보너스 일치 최대값 : {best_bonus}"
        )
        print(
            f"5등 이상 당첨 세트 수 : {len(winning_sets)}개"
        )

    return prediction_df


def save_prediction_sets(
        df,
        filename=PREDICTION_FILE
):
    """
    현재 과거 당첨번호 CSV의 최종 회차 + 1을
    '예측회차'로 부여하여 예상번호 10세트를 만듭니다.

    예)
    현재 최종 실제 회차 = 1185
    -> 새 예상번호의 예측회차 = 1186

    동일한 예측회차가 이미 CSV에 존재하면 재생성하지 않습니다.
    이는 실제 결과가 발표되기 전에 예측값이 바뀌는 것을 방지하기 위함입니다.

    예상번호 파일은 회차별 누적 기록으로 사용합니다.
    """

    if df is None or df.empty:
        print(
            "과거 당첨번호 데이터가 없어 다음 회차 예상번호를 생성하지 않습니다."
        )
        return pd.DataFrame()

    if "회차" not in df.columns:
        print(
            "과거 당첨번호 데이터에 '회차' 컬럼이 없어 "
            "예상번호를 생성하지 않습니다."
        )
        return pd.DataFrame()

    latest_actual_round = int(
        df["회차"].max()
    )

    prediction_round = (
        latest_actual_round + 1
    )

    # 기존 예상번호 파일 로드
    if os.path.exists(filename):
        try:
            existing_prediction_df = pd.read_csv(
                filename,
                encoding="utf-8-sig"
            )
        except Exception:
            existing_prediction_df = pd.DataFrame()
    else:
        existing_prediction_df = pd.DataFrame()

    # 이전 버전 CSV에 예측회차가 없으면
    # 기존 데이터는 보존하되 새 형식과 혼합하지 않고 백업
    if (
        not existing_prediction_df.empty
        and "예측회차"
        not in existing_prediction_df.columns
    ):
        backup_filename = (
            "당첨예상번호_예측회차없음_백업.csv"
        )

        existing_prediction_df.to_csv(
            backup_filename,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            f"기존 예상번호 파일에 예측회차가 없어 "
            f"{backup_filename}으로 백업했습니다."
        )

        existing_prediction_df = pd.DataFrame()

    # 이미 다음 회차 예상번호가 있으면 그대로 유지
    if (
        not existing_prediction_df.empty
        and "예측회차"
        in existing_prediction_df.columns
    ):
        existing_prediction_df["예측회차"] = pd.to_numeric(
            existing_prediction_df["예측회차"],
            errors="coerce"
        )

        same_round = existing_prediction_df[
            existing_prediction_df["예측회차"]
            == prediction_round
        ].copy()

        if not same_round.empty:
            print()
            print("=" * 80)
            print(
                f"{prediction_round}회 예상번호는 이미 생성되어 있습니다."
            )
            print("기존 예상번호를 유지하며 재생성하지 않습니다.")
            print("=" * 80)

            display_cols = [
                col
                for col in [
                    "예측회차",
                    "세트",
                    "번호1", "번호2", "번호3",
                    "번호4", "번호5", "번호6",
                    "종합점수",
                    "평가상태"
                ]
                if col in same_round.columns
            ]

            print(
                same_round[
                    display_cols
                ].to_string(index=False)
            )

            return same_round

    # 새 다음 회차 10세트 생성
    prediction_df = generate_prediction_sets(
        df,
        set_count=10,
        candidate_count=30000
    )

    prediction_df.insert(
        0,
        "예측회차",
        prediction_round
    )

    # 추후 실제 결과 비교를 위한 컬럼
    prediction_df["실제번호1"] = ""
    prediction_df["실제번호2"] = ""
    prediction_df["실제번호3"] = ""
    prediction_df["실제번호4"] = ""
    prediction_df["실제번호5"] = ""
    prediction_df["실제번호6"] = ""
    prediction_df["실제보너스"] = ""
    prediction_df["본번호일치수"] = ""
    prediction_df["보너스일치"] = ""
    prediction_df["총일치수"] = ""
    prediction_df["당첨등수"] = ""
    prediction_df["평가상태"] = "대기"

    # 회차별 예상번호 누적
    if existing_prediction_df.empty:
        combined_prediction_df = (
            prediction_df.copy()
        )
    else:
        combined_prediction_df = pd.concat(
            [
                existing_prediction_df,
                prediction_df
            ],
            ignore_index=True
        )

    combined_prediction_df = (
        combined_prediction_df
        .sort_values(
            ["예측회차", "세트"]
        )
        .reset_index(drop=True)
    )

    combined_prediction_df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print(
        f"{prediction_round}회 당첨 예상번호 10세트"
    )
    print("=" * 80)

    display_cols = [
        "예측회차",
        "세트",
        "번호1", "번호2", "번호3",
        "번호4", "번호5", "번호6",
        "종합점수"
    ]

    print(
        prediction_df[
            display_cols
        ].to_string(index=False)
    )

    print()
    print(
        f"예상번호 누적 CSV 저장 완료 : {filename}"
    )
    print(
        f"현재 실제 최종 회차 {latest_actual_round}회 기준 "
        f"다음 회차인 {prediction_round}회로 예측번호를 부여했습니다."
    )
    print(
        "다음 실행에서 해당 회차 실제 당첨번호가 수집되면 "
        "기존 예상번호 10세트와 자동 비교합니다."
    )

    return prediction_df


# ============================================================
# 9. 분석 결과 화면 출력
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
# 10. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # 조회 회차와 CSV 마지막 회차 비교 후 필요한 신규 회차만 수집
    # --------------------------------------------------------

    df = collect_all_lotto(
        query_round=QUERY_ROUND,
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
    # 전 회차 분석용 파생변수 생성
    # --------------------------------------------------------

    df = create_features(df)

    # --------------------------------------------------------
    # STEP 3
    # 본번호 6개 + 보너스 번호의 전 회차 빈도 분석
    # --------------------------------------------------------

    frequency_df = (
        calculate_frequency(df)
    )

    # --------------------------------------------------------
    # STEP 4
    # 최근 100회 화면 참고 분석
    # --------------------------------------------------------

    recent_frequency_df = (
        calculate_recent_frequency(
            df,
            recent_count=100
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # 전 회차 기준 미출현 기간 분석
    # --------------------------------------------------------

    missing_df = (
        calculate_missing_draws(df)
    )


    # --------------------------------------------------------
    # STEP 6
    # 이전에 생성한 예상번호와 실제 당첨번호 자동 비교/평가
    # --------------------------------------------------------

    evaluated_prediction_df = evaluate_prediction_history(
        df,
        filename=PREDICTION_FILE
    )

    # --------------------------------------------------------
    # STEP 7
    # 현재 실제 최종 회차 + 1 회차의 예상번호 10세트 생성
    # --------------------------------------------------------

    prediction_df = save_prediction_sets(
        df,
        filename=PREDICTION_FILE
    )

    # --------------------------------------------------------
    # STEP 7
    # 분석 결과 자체는 별도 CSV로 저장하지 않음
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STEP 8
    # 분석 결과 화면 출력
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
    print("예상번호는 회차별로 당첨예상번호.csv에 누적 저장하고, 실제 결과 발표 후 자동 비교/평가합니다.")


# ============================================================
# 프로그램 실행
# ============================================================

if __name__ == "__main__":

    main()
