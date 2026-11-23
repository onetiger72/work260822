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
# - 기본 실행은 두 생성기를 비교합니다. --review-session으로 대화 중 검토를 이어갑니다.
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
from collections import Counter, defaultdict
from itertools import combinations


# ============================================================
# 1. 기본 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "과거로또 당첨번호.csv")
PREDICTION_FILE = os.path.join(BASE_DIR, "당첨예상번호.csv")
# 실제 조회할 로또 회차를 입력하세요. (0 이하이면 추가 조회 없이 기존 CSV 사용)
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
    # 1. 기존 CSV 로드 및 마지막 회차 확인
    # --------------------------------------------------------
    existing_df = load_history_csv()

    if existing_df.empty:
        csv_last_round = 0
        print(f"기존 CSV 없음 : {CSV_FILE}")
    else:
        csv_last_round = int(existing_df["회차"].max())
        print(f"CSV 마지막 회차 : {csv_last_round}회")

    # --------------------------------------------------------
    # 2. 조회 회차 유효성 확인
    # --------------------------------------------------------
    query_round = int(query_round)

    if query_round < 1:
        if csv_last_round > 0:
            print(
                "조회 회차가 지정되지 않아(0 이하) 신규 회차 조회를 건너뛰고 "
                "기존 CSV 데이터를 사용합니다."
            )
        else:
            print("조회 회차를 1 이상의 값으로 설정해야 합니다.")
        return existing_df

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
    한 게임(1줄 6개 번호) 안에 당첨번호가 최대한 뭉쳐서(응집) 들어가도록 설계된
    압축 클리크(Clique) & 마르코프 전이 기반 예상번호 10세트 생성 엔진입니다.

    [번호 단위 다중 신호 분석]
    1. 직전 1~3회차 -> 다음 회차 마르코프 전이 확률(Markov Transition) 및 보너스 전이
    2. 단기(half-life 10) / 중기(35) / 장기(120) 다중 지수감쇠 최근성
    3. 이월수(직전 당첨/보너스 재출현) 및 이웃수(±1), 징검다리(2회 전), 최근 핫 끝수 동조
    4. 경과회차별 실제 출현 확률(Empirical Hazard Rate) 및 번호별 고유 출현 주기 적합도

    [한 줄(1게임) 내 응집력 극대화 분석]
    5. 번호 쌍 동시출현 Lift + 시간가중 동시출현 + 직전 회차 공전이(Co-Transition) 시너지
    6. 3개 번호 동시출현(Triplet Co-occurrence) 결속력
    7. 번호가 여러 게임으로 찢어지지 않도록 상위 핵심 후보군(Core Pool) 전수 압축 조합과
       조건부 연쇄 클리크(Conditional Clique Growth)로 후보 생성
    8. AC값(>=7), 합계(92~178), 홀짝/고저/구간/끝수 표준 1등 구조 필터 적용
    9. 3-tuple / 4-tuple 커버링(Abbreviated Wheeling)을 결합해 핵심 번호 적중 시
       반드시 한 줄 안에 3~6개가 동시에 모이도록(다수 게임 중복 당첨 포함) 상위 10세트 선택
    """

    if df.empty:
        return pd.DataFrame()

    ordered = (
        df.sort_values("회차")
          .reset_index(drop=True)
          .copy()
    )
    total_draws = len(ordered)

    main_cols = [
        "번호1", "번호2", "번호3",
        "번호4", "번호5", "번호6"
    ]

    draws_main = [
        tuple(sorted(int(row[c]) for c in main_cols))
        for _, row in ordered.iterrows()
    ]
    draws_bonus = [
        int(row["보너스"])
        for _, row in ordered.iterrows()
    ]

    # ========================================================
    # A. 기본 빈도, 다중 시간감쇠, 동시출현(Pair/Triplet), 마르코프 전이 집계
    # ========================================================
    main_counts = Counter()
    bonus_counts = Counter()
    short_decay = Counter()   # half-life 10회
    mid_decay = Counter()     # half-life 35회
    long_decay = Counter()    # half-life 120회

    l_short = math.log(2.0) / 10.0
    l_mid = math.log(2.0) / 35.0
    l_long = math.log(2.0) / 120.0

    pair_counts = Counter()
    pair_decay = Counter()
    triplet_decay = Counter()
    main_bonus_decay = Counter()

    trans_lag1 = defaultdict(Counter)
    trans_lag2 = defaultdict(Counter)
    trans_lag3 = defaultdict(Counter)
    bonus_trans_lag1 = defaultdict(Counter)

    appear_indices = defaultdict(list)

    for idx in range(total_draws):
        age = total_draws - 1 - idx
        w_s = math.exp(-l_short * age)
        w_m = math.exp(-l_mid * age)
        w_l = math.exp(-l_long * age)

        nums = draws_main[idx]
        bns = draws_bonus[idx]

        for n in nums:
            appear_indices[n].append(idx)
            main_counts[n] += 1
            short_decay[n] += w_s
            mid_decay[n] += w_m
            long_decay[n] += w_l

        bonus_counts[bns] += 1

        p_w = 0.45 * w_s + 0.35 * w_m + 0.20 * w_l
        for a, b in combinations(nums, 2):
            pair_counts[(a, b)] += 1
            pair_decay[(a, b)] += p_w

        for a, b, c in combinations(nums, 3):
            triplet_decay[(a, b, c)] += p_w

        for n in nums:
            key = (min(n, bns), max(n, bns))
            main_bonus_decay[key] += w_m

        t_weight = 0.25 + 0.75 * w_l
        if idx >= 1:
            for p in draws_main[idx - 1]:
                for cur in nums:
                    trans_lag1[p][cur] += t_weight
            prev_b = draws_bonus[idx - 1]
            for cur in nums:
                bonus_trans_lag1[prev_b][cur] += t_weight
        if idx >= 2:
            for p in draws_main[idx - 2]:
                for cur in nums:
                    trans_lag2[p][cur] += t_weight
        if idx >= 3:
            for p in draws_main[idx - 3]:
                for cur in nums:
                    trans_lag3[p][cur] += t_weight

    # ========================================================
    # B. 경과회차별 실제 출현율(Empirical Hazard) 및 고유 주기 적합도
    # ========================================================
    gap_appear = Counter()
    gap_survive = Counter()
    num_gaps = {}

    for n in range(1, 46):
        idxs = appear_indices[n]
        gaps = []
        for i in range(1, len(idxs)):
            g = idxs[i] - idxs[i - 1] - 1
            gaps.append(g)
            g_cap = min(g, 35)
            gap_appear[g_cap] += 1
            for step in range(g_cap + 1):
                gap_survive[step] += 1
        num_gaps[n] = gaps

    cycle_score = {}
    cur_gaps = {}
    for n in range(1, 46):
        idxs = appear_indices[n]
        cur_g = (total_draws - 1 - idxs[-1]) if idxs else total_draws
        cur_gaps[n] = cur_g
        g_cap = min(cur_g, 35)
        emp_hazard = (gap_appear[g_cap] + 1.0) / (gap_survive[g_cap] + 7.5)
        recent_g = num_gaps[n][-12:]
        if recent_g:
            avg_g = sum(recent_g) / len(recent_g)
            dist_ratio = abs(cur_g - avg_g) / max(avg_g, 1.0)
            rhythm_fit = math.exp(-0.75 * (dist_ratio ** 2))
        else:
            rhythm_fit = 0.5
        cycle_score[n] = emp_hazard * 0.60 + rhythm_fit * 0.40

    # ========================================================
    # C. 직전 1~3회차 기반 마르코프 전이(Transition) 점수
    # ========================================================
    last1_nums = draws_main[-1]
    last1_bns = draws_bonus[-1]
    last2_nums = draws_main[-2] if total_draws >= 2 else ()
    last2_bns = draws_bonus[-2] if total_draws >= 2 else -1
    last3_nums = draws_main[-3] if total_draws >= 3 else ()

    transition_score = {}
    for n in range(1, 46):
        norm_f = max(main_counts[n], 1) ** 0.35
        s1 = sum(trans_lag1[p][n] for p in last1_nums) / norm_f
        sb = bonus_trans_lag1[last1_bns][n] / norm_f
        s2 = (
            sum(trans_lag2[p][n] for p in last2_nums) / norm_f
            if last2_nums else 0.0
        )
        s3 = (
            sum(trans_lag3[p][n] for p in last3_nums) / norm_f
            if last3_nums else 0.0
        )
        transition_score[n] = (
            s1 * 0.52 + sb * 0.14 + s2 * 0.22 + s3 * 0.12
        )

    # ========================================================
    # D. SQUAD 1 (세트 1~5): 교차 브릿지(Cross-Parent Bridge) & 공명 이월 클리크 압축
    # (직전·전전 회차 끝수 점프 + 이웃수(±1, ±2) 교차점 및 공명 이월수를 9개 고밀도 풀로 압축)
    # ========================================================
    p1_main = set(last1_nums)
    p1_bns = last1_bns
    p1_all = p1_main | {p1_bns}
    p2_main = set(last2_nums)
    p2_all = p2_main | ({last2_bns} if last2_bns > 0 else set())

    r10 = Counter(n for d in draws_main[-10:] for n in d)
    r30 = Counter(n for d in draws_main[-30:] for n in d)

    # 단기 극과열 부모(최근 10회 중 5회 이상 출현)는 단기 에너지를 소진했으므로 파생 브릿지 부모에서 제외
    active_p1_main = {p for p in p1_main if r10[p] < 5}
    active_p1_all = {p for p in p1_all if r10[p] < 5}
    p1_end_counts = Counter(p % 10 for p in p1_all)
    p12_end_decades = {
        end_d: len({(p - 1) // 10 for p in (active_p1_all | p2_all) if p % 10 == end_d})
        for end_d in range(10)
    }

    bridge_scores = {}
    for n in range(1, 46):
        g = cur_gaps[n]
        end_parents = {p for p in active_p1_all if (p % 10) == (n % 10) and p != n}
        nbr1_main = {p for p in active_p1_main if abs(n - p) == 1}
        nbr1_all = {p for p in active_p1_all if abs(n - p) == 1}
        nbr2_all = {p for p in active_p1_all if abs(n - p) == 2}
        in_p1 = (n in p1_all)
        in_p2_main = (n in p2_main and not in_p1)
        in_p2 = (n in p2_all and not in_p1)

        cross_decade_end = {p for p in end_parents if (p - 1) // 10 != (n - 1) // 10}
        local_nbrs = nbr1_all | nbr2_all
        dec = (n - 1) // 10
        dec_p1_cnt = sum(1 for p in p1_all if (p - 1) // 10 == dec)
        is_overcrowded_interior = (
            (not in_p1)
            and dec_p1_cnt >= 4
            and any(p < n for p in local_nbrs)
            and any(p > n for p in local_nbrs)
        )

        score = 0.0
        if in_p1:
            is_middle_of_triple = ((n - 1) in p1_main and (n + 1) in p1_main)
            end_saturated_pen = -0.75 if p1_end_counts[n % 10] >= 3 else 0.0
            middle_pen = -0.55 if is_middle_of_triple else 0.0
            resonance = (
                len(nbr1_all) * 0.45
                + len(cross_decade_end) * 0.40
                + (0.35 if n == p1_bns else 0.0)
                + end_saturated_pen
                + middle_pen
            )
            if 1 <= r30[n] <= 6 or n == p1_bns:
                score += 4.25 + resonance + 0.10 * min(r30[n], 6)
            else:
                score += 2.40 + resonance
        else:
            reb_boost = 0.38 if 8 <= g <= 25 else 0.0
            if cross_decade_end and nbr1_main:
                score += (2.6 if is_overcrowded_interior else 4.95 + reb_boost) + 0.35 * len(cross_decade_end)
            elif cross_decade_end and local_nbrs:
                score += (2.5 if is_overcrowded_interior else 4.68 + reb_boost) + 0.35 * len(cross_decade_end)
            elif nbr1_main and nbr2_all and not is_overcrowded_interior:
                score += 4.65 + reb_boost + 0.20 * len(local_nbrs)
            elif cross_decade_end and in_p2:
                p2_boost = 0.25 if in_p2_main else 0.0
                score += 4.40 + p2_boost + 0.35 * len(cross_decade_end)
            elif len(cross_decade_end) >= 2:
                active_dec_boost = 0.45 if dec_p1_cnt >= 2 else 0.0
                score += 4.35 + active_dec_boost + 0.35 * len(cross_decade_end)
            elif cross_decade_end and p12_end_decades[n % 10] >= 3 and not any((p - 1) // 10 == dec and p % 10 == n % 10 for p in (active_p1_all | p2_all)):
                score += 4.70 + (0.35 if 1 <= g <= 4 else 0.0)
            elif nbr1_main and in_p2 and not is_overcrowded_interior:
                score += 4.10
            elif nbr2_all and r10[n] >= 2 and 2 <= r30[n] <= 5 and not is_overcrowded_interior:
                fresh_nbr2 = 0.15 if any(r10[p] <= 2 for p in nbr2_all) else 0.0
                score += 4.25 + fresh_nbr2
            elif nbr1_all and 2 <= r30[n] <= 6:
                score += (2.2 if is_overcrowded_interior else 3.20 + reb_boost)
            elif cross_decade_end and (2 <= r30[n] <= 6 or 1 <= g <= 4):
                score += 3.00 + (0.25 if 1 <= g <= 3 else 0.0)
            elif in_p2 and 3 <= r30[n] <= 6:
                score += 2.60

            if r30[n] >= 7:
                score *= 0.45

        bridge_scores[n] = score + 0.07 * min(r30[n], 6)

    ranked_b = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
    top_bridge = []
    carry_cnt = 0
    carry_per_dec = Counter()
    dec_cnt = Counter()
    end_cnt = Counter()

    for n, sc in ranked_b:
        dec = (n - 1) // 10
        end_d = n % 10
        if n in p1_all:
            if carry_cnt >= 2 or carry_per_dec[dec] >= 1:
                continue
        if dec_cnt[dec] >= 4:
            continue
        if end_cnt[end_d] >= 3:
            continue
        top_bridge.append(n)
        if n in p1_all:
            carry_cnt += 1
            carry_per_dec[dec] += 1
        dec_cnt[dec] += 1
        end_cnt[end_d] += 1
        if len(top_bridge) >= 10:
            break

    bridge_pool_9 = top_bridge[:9]
    bridge_combs = []
    for comb in combinations(sorted(bridge_pool_9), 6):
        c_set = set(comb)
        if len(c_set & p1_all) > 2:
            continue
        s = sum(comb)
        if not (90 <= s <= 195):
            continue
        odds = sum(1 for x in comb if x % 2 == 1)
        if odds in (0, 6):
            continue
        decs = Counter((x - 1) // 10 for x in comb)
        if max(decs.values()) >= 4 or len(decs) < 3:
            continue

        node_s = sum(bridge_scores[x] + 0.35 * cycle_score[x] for x in comb)
        end_digits = Counter(x % 10 for x in comb)
        same_end_pairs = sum(1 for v in end_digits.values() if v == 2)
        end_syn = 1.15 if same_end_pairs >= 1 else 0.0
        if max(end_digits.values()) >= 3:
            end_syn -= 1.5

        consec = sum(1 for i in range(5) if comb[i + 1] - comb[i] == 1)
        consec_syn = 0.85 if consec == 1 else (-1.2 if consec >= 2 else 0.0)
        dec_syn = 0.35 if (len(decs) >= 4 and max(decs.values()) <= 2) else 0.0
        top7_hits = len(c_set & set(top_bridge[:7]))
        top7_bonus = 0.45 * top7_hits

        total_b_score = node_s + end_syn + consec_syn + dec_syn + top7_bonus
        bridge_combs.append((total_b_score, tuple(sorted(comb))))

    bridge_combs.sort(key=lambda x: x[0], reverse=True)
    squad1_count = min(5, set_count)
    squad1_selected = []
    covered_b3 = set()
    covered_b4 = set()

    top7_combs = [item for item in bridge_combs if len(set(item[1]) & set(top_bridge[:7])) >= 5]
    if top7_combs:
        sc0, c1 = top7_combs[0]
        squad1_selected.append((c1, sc0 / 50.0))
        for t3 in combinations(c1, 3):
            covered_b3.add(t3)
        for t4 in combinations(c1, 4):
            covered_b4.add(t4)

    while len(squad1_selected) < squad1_count and bridge_combs:
        best_c = None
        best_sc = 0.0
        best_val = -1e9
        use_t4_fusion = (len(squad1_selected) >= 3)
        for sc, comb in bridge_combs:
            if any(comb == ex for ex, _ in squad1_selected):
                continue
            max_ov = max((len(set(comb) & set(ex)) for ex, _ in squad1_selected), default=0)
            if max_ov >= 5:
                continue
            if use_t4_fusion:
                new_t4 = sum(1 for t4 in combinations(comb, 4) if t4 not in covered_b4)
                val = sc + 0.85 * new_t4
            else:
                new_t3 = sum(1 for t3 in combinations(comb, 3) if t3 not in covered_b3)
                val = sc + 0.55 * new_t3
            if val > best_val:
                best_val = val
                best_c = comb
                best_sc = sc
        if best_c is None:
            for sc, comb in bridge_combs:
                if not any(comb == ex for ex, _ in squad1_selected):
                    best_c = comb
                    best_sc = sc
                    break
        if best_c is None:
            break
        squad1_selected.append((best_c, best_sc / 50.0))
        for t3 in combinations(best_c, 3):
            covered_b3.add(t3)
        for t4 in combinations(best_c, 4):
            covered_b4.add(t4)

    # ========================================================
    # E. SQUAD 2 (세트 6~10): 이월수 / 이웃수 / 징검다리 / 마르코프 다중주기 클리크
    # ========================================================
    recent_endings = Counter(
        n % 10 for r in draws_main[-4:] for n in r
    )
    last1_set = set(last1_nums)
    last2_set = set(last2_nums)
    nbr_1 = {
        x for p in last1_nums for x in (p - 1, p + 1) if 1 <= x <= 45
    } - last1_set
    nbr_bns = {
        x for x in (last1_bns - 1, last1_bns + 1) if 1 <= x <= 45
    } - last1_set

    pattern_score = {}
    for n in range(1, 46):
        val = 0.0
        if n in last1_set:
            val += 0.55
        elif n == last1_bns:
            val += 0.45
        elif n in nbr_1:
            val += 0.48
        elif n in nbr_bns:
            val += 0.30
        if n in last2_set and n not in last1_set:
            val += 0.25
        val += min(recent_endings[n % 10] / 6.0, 0.35)
        pattern_score[n] = val

    # ========================================================
    # E. Z-score Sigmoid 정규화 및 번호별 트랙 점수 산출
    # ========================================================
    def smooth_norm(d):
        vals = [float(d.get(n, 0.0)) for n in range(1, 46)]
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(
            sum((v - mean_v) ** 2 for v in vals) / len(vals)
        )
        if std_v < 1e-9:
            return {n: 0.5 for n in range(1, 46)}
        return {
            n: 1.0 / (
                1.0 + math.exp(-((float(d.get(n, 0.0)) - mean_v) / std_v))
            )
            for n in range(1, 46)
        }

    pair_deg = Counter()
    for (a, b), v in pair_decay.items():
        pair_deg[a] += v
        pair_deg[b] += v

    n_main = smooth_norm(main_counts)
    n_short = smooth_norm(short_decay)
    n_mid = smooth_norm(mid_decay)
    n_long = smooth_norm(long_decay)
    n_trans = smooth_norm(transition_score)
    n_cycle = smooth_norm(cycle_score)
    n_pat = smooth_norm(pattern_score)
    n_deg = smooth_norm(pair_deg)

    score_total = {}
    score_trend = {}
    score_rebound = {}

    for n in range(1, 46):
        score_trend[n] = (
            n_trans[n] * 0.28
            + n_short[n] * 0.20
            + n_mid[n] * 0.20
            + n_deg[n] * 0.14
            + n_long[n] * 0.10
            + n_main[n] * 0.08
        )
        score_rebound[n] = (
            n_pat[n] * 0.30
            + n_cycle[n] * 0.26
            + n_trans[n] * 0.20
            + n_mid[n] * 0.14
            + n_long[n] * 0.10
        )
        score_total[n] = (
            0.58 * score_trend[n] + 0.42 * score_rebound[n]
        )

    # ========================================================
    # F. 한 줄 내 번호 결속력(Pair & Triplet Synergy) 행렬
    # ========================================================
    max_pair_d = max(pair_decay.values()) if pair_decay else 1.0
    max_pair_c = max(pair_counts.values()) if pair_counts else 1.0
    max_trip_d = max(triplet_decay.values()) if triplet_decay else 1.0
    max_mb_d = max(main_bonus_decay.values()) if main_bonus_decay else 1.0

    co_trans = Counter()
    for a, b in combinations(range(1, 46), 2):
        co_trans[(a, b)] = math.sqrt(
            max(transition_score[a] * transition_score[b], 0.0)
        )
    max_cot = max(co_trans.values()) if co_trans else 1.0

    pair_synergy = {}
    for a, b in combinations(range(1, 46), 2):
        key = (a, b)
        expected = max(
            (main_counts[a] * main_counts[b]) / max(total_draws, 1),
            1.0
        )
        lift = min((pair_counts[key] / expected) / 2.2, 1.0)
        p_d = pair_decay[key] / max_pair_d
        p_c = pair_counts[key] / max_pair_c
        cot = co_trans[key] / max_cot
        mb = main_bonus_decay[key] / max_mb_d
        pair_synergy[key] = (
            p_d * 0.38
            + cot * 0.25
            + lift * 0.20
            + p_c * 0.10
            + mb * 0.07
        )

    # ========================================================
    # G. 1등 조합 구조 필터 및 한 줄 응집 점수 함수
    # ========================================================
    hist_sums = [sum(nums) for nums in draws_main]
    sum_mean = sum(hist_sums) / len(hist_sums)
    sum_std = math.sqrt(
        sum((x - sum_mean) ** 2 for x in hist_sums) / len(hist_sums)
    )

    def calc_ac_value(nums):
        diffs = {abs(b - a) for a, b in combinations(nums, 2)}
        return len(diffs) - 5

    def is_valid_structure(comb):
        s = sum(comb)
        if not (92 <= s <= 178):
            return False
        odds = sum(n % 2 == 1 for n in comb)
        if odds in (0, 6):
            return False
        highs = sum(n >= 23 for n in comb)
        if highs in (0, 6):
            return False
        consec = sum(comb[i + 1] - comb[i] == 1 for i in range(5))
        if consec >= 3:
            return False
        if calc_ac_value(comb) < 7:
            return False
        decades = Counter((n - 1) // 10 for n in comb)
        if max(decades.values()) >= 4:
            return False
        endings = Counter(n % 10 for n in comb)
        if max(endings.values()) >= 3:
            return False
        return True

    def score_combination(comb):
        ind_s = sum(score_total[n] for n in comb) / 6.0
        p_vals = sorted(
            (pair_synergy[(a, b)] for a, b in combinations(comb, 2)),
            reverse=True
        )
        pair_s = (
            0.65 * (sum(p_vals[:6]) / 6.0)
            + 0.35 * (sum(p_vals) / 15.0)
        )

        t_vals = sorted(
            (
                triplet_decay.get((a, b, c), 0.0) / max_trip_d
                for a, b, c in combinations(comb, 3)
            ),
            reverse=True
        )
        trip_s = (
            0.70 * (sum(t_vals[:4]) / 4.0)
            + 0.30 * (sum(t_vals) / 20.0)
        )

        s = sum(comb)
        sum_z = (s - sum_mean) / max(sum_std, 1.0)
        sum_s = math.exp(-0.5 * (sum_z ** 2))
        odds = sum(n % 2 == 1 for n in comb)
        odd_s = 1.0 if odds in (2, 3, 4) else 0.55
        highs = sum(n >= 23 for n in comb)
        high_s = 1.0 if highs in (2, 3, 4) else 0.55
        ac_s = 1.0 if calc_ac_value(comb) >= 8 else 0.85
        has_carry_or_nbr = any(
            (n in last1_set or n in nbr_1) for n in comb
        )
        carry_s = 1.0 if has_carry_or_nbr else 0.75

        struct_s = (
            sum_s * 0.30
            + odd_s * 0.20
            + high_s * 0.20
            + ac_s * 0.15
            + carry_s * 0.15
        )

        return (
            ind_s * 0.42
            + pair_s * 0.28
            + trip_s * 0.12
            + struct_s * 0.18
        )

    def build_clique_from_seed(seed_nums, pool_nums):
        clique = list(seed_nums)
        while len(clique) < 6:
            best_n = None
            best_gain = -1.0
            for cand in pool_nums:
                if cand in clique:
                    continue
                syn = sum(
                    pair_synergy[(min(cand, m), max(cand, m))]
                    for m in clique
                ) / len(clique)
                trip_syn = 0.0
                if len(clique) >= 2:
                    t_list = [
                        triplet_decay.get(
                            tuple(sorted((cand, m1, m2))), 0.0
                        ) / max_trip_d
                        for m1, m2 in combinations(clique, 2)
                    ]
                    trip_syn = (
                        max(t_list) * 0.6
                        + (sum(t_list) / len(t_list)) * 0.4
                    )
                gain = (
                    score_total[cand] * 0.44
                    + syn * 0.40
                    + trip_syn * 0.16
                )
                if gain > best_gain:
                    best_gain = gain
                    best_n = cand
            if best_n is None:
                break
            clique.append(best_n)
        return tuple(sorted(clique))

    # ========================================================
    # H. 고밀도 클러스터 전수 압축 전개 + 조건부 연쇄 클리크 후보 생성
    # ========================================================
    trend_pool = [
        n for n, _ in sorted(
            score_trend.items(), key=lambda x: x[1], reverse=True
        )[:12]
    ]
    rebound_pool = [
        n for n, _ in sorted(
            score_rebound.items(), key=lambda x: x[1], reverse=True
        )[:12]
    ]
    total_pool = [
        n for n, _ in sorted(
            score_total.items(), key=lambda x: x[1], reverse=True
        )[:13]
    ]
    wide_pool = [
        n for n, _ in sorted(
            score_total.items(), key=lambda x: x[1], reverse=True
        )[:24]
    ]

    candidate_map = {}

    for pool in (total_pool, trend_pool, rebound_pool):
        for comb in combinations(pool, 6):
            comb = tuple(sorted(comb))
            if comb not in candidate_map and is_valid_structure(comb):
                candidate_map[comb] = score_combination(comb)

    top_pairs = sorted(
        [
            (
                (min(a, b), max(a, b)),
                pair_synergy[(min(a, b), max(a, b))]
                + 0.5 * (score_total[a] + score_total[b])
            )
            for a, b in combinations(wide_pool, 2)
        ],
        key=lambda x: x[1],
        reverse=True
    )[:80]

    for (a, b), _ in top_pairs:
        comb = build_clique_from_seed((a, b), wide_pool)
        if (
            len(comb) == 6
            and comb not in candidate_map
            and is_valid_structure(comb)
        ):
            candidate_map[comb] = score_combination(comb)

    ranked_all = sorted(
        candidate_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ========================================================
    # I. 3-tuple / 4-tuple 커버링(Wheeling) 기반 Squad 2 (세트 6~10) 압축 및 통합
    # ========================================================
    squad2_selected = []
    covered_3tuples = set()
    covered_4tuples = set()

    for _ in range(max(0, set_count - len(squad1_selected))):
        best_item = None
        best_obj = -1.0

        for comb, base_score in ranked_all[:500]:
            if (
                any(comb == ex for ex, _ in squad1_selected)
                or any(comb == ex for ex, _ in squad2_selected)
            ):
                continue

            if any(
                len(set(comb) & set(ex)) >= 5
                for ex, _ in squad2_selected
            ):
                continue

            new_3s = sum(
                1 for sub3 in combinations(comb, 3)
                if sub3 not in covered_3tuples
            )
            new_4s = sum(
                1 for sub4 in combinations(comb, 4)
                if sub4 not in covered_4tuples
            )

            obj = (
                base_score
                + 0.024 * (new_3s / 20.0)
                + 0.012 * (new_4s / 15.0)
            )
            if obj > best_obj:
                best_obj = obj
                best_item = (comb, base_score)

        if best_item is None:
            for comb, base_score in ranked_all:
                if (
                    not any(comb == ex for ex, _ in squad1_selected)
                    and not any(comb == ex for ex, _ in squad2_selected)
                ):
                    best_item = (comb, base_score)
                    break

        if best_item is not None:
            squad2_selected.append(best_item)
            for sub3 in combinations(best_item[0], 3):
                covered_3tuples.add(sub3)
            for sub4 in combinations(best_item[0], 4):
                covered_4tuples.add(sub4)

    selected_sets = squad1_selected + squad2_selected

    # ========================================================
    # J. 결과 DataFrame
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
    ).astype("Int64")

    # 평가 결과 컬럼 준비 (숫자형 컬럼은 Int64, 문자열 컬럼은 object로 타입 고정)
    numeric_result_columns = [
        "실제번호1", "실제번호2", "실제번호3",
        "실제번호4", "실제번호5", "실제번호6",
        "실제보너스",
        "본번호일치수",
        "보너스일치",
        "총일치수"
    ]
    string_result_columns = [
        "당첨등수",
        "평가상태"
    ]

    for col in numeric_result_columns:
        if col not in prediction_df.columns:
            prediction_df[col] = pd.Series(
                pd.NA,
                index=prediction_df.index,
                dtype="Int64"
            )
        else:
            prediction_df[col] = pd.to_numeric(
                prediction_df[col],
                errors="coerce"
            ).astype("Int64")

    for col in string_result_columns:
        if col not in prediction_df.columns:
            prediction_df[col] = pd.Series(
                "",
                index=prediction_df.index,
                dtype="object"
            )
        else:
            prediction_df[col] = (
                prediction_df[col]
                .where(prediction_df[col].notna(), "")
                .astype(str)
                .astype("object")
            )

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
        raise ValueError("기존 예상번호 파일에 예측회차가 없어 저장을 중단합니다.")

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
    for col in [
        "실제번호1", "실제번호2", "실제번호3",
        "실제번호4", "실제번호5", "실제번호6",
        "실제보너스",
        "본번호일치수",
        "보너스일치",
        "총일치수"
    ]:
        prediction_df[col] = pd.Series(
            pd.NA,
            index=prediction_df.index,
            dtype="Int64"
        )
    prediction_df["당첨등수"] = pd.Series(
        "",
        index=prediction_df.index,
        dtype="object"
    )
    prediction_df["평가상태"] = pd.Series(
        "대기",
        index=prediction_df.index,
        dtype="object"
    )

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
    """F5와 명령줄 모두 공식 회차 확인과 동일한 후보 검토를 거친다."""
    import runpy
    return runpy.run_path(os.path.join(BASE_DIR, "로또 당첨번호 예측.py"), run_name="__main__")


if __name__ == "__main__":

    main()
