"""CSV 입력 경계: 값이나 회차를 추측하지 않고, 오류는 호출자에게 전달한다.

파일 부재만 빈 이력으로 취급한다. 기존 파일의 파싱/검증 실패를 빈 표로
바꾸면 호출자가 원본을 신규 자료로 덮어쓸 수 있으므로 복구를 시도하지 않는다.
이 모듈은 읽기와 메모리 검증만 담당하며 중간 파일을 만들지 않는다.
"""
import csv
import io
import math
from pathlib import Path

import pandas as pd

from lotto_validation import NUMBERS, strict_int, validate_history

ACTUAL_COLUMNS = [f"실제번호{i}" for i in range(1, 7)]
PREDICTION_COLUMNS = (["예측회차", "세트"] + NUMBERS + ["종합점수"]
                      + ACTUAL_COLUMNS + ["본번호일치수", "보너스일치", "총일치수"])
HISTORY_COLUMNS = ["회차", "날짜"] + NUMBERS + ["보너스"]


class CsvDataError(ValueError):
    """원본 보존을 위해 수집·생성·저장을 중단해야 하는 CSV 오류."""


def read_csv_checked(filename, *, missing_ok=False):
    """UTF-8 CSV를 원래 문자열로 읽는다. 반환값 None은 파일 부재만 뜻한다.

    pandas의 헤더 중복 자동 변경과 소수/빈 값의 묵시적 변환을 피하기 위해
    CSV 구조를 먼저 확인한다. 필드 수 불일치도 행을 버리지 않고 거부한다.
    """
    path = Path(filename)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise CsvDataError(f"필수 CSV 파일이 없습니다: {path.name}") from None
    except OSError as error:
        raise CsvDataError(f"CSV 읽기 실패, 원본을 보존합니다: {path.name}") from error
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True))
        if not rows or not rows[0] or any(not name.strip() for name in rows[0]):
            raise ValueError("empty file or empty header")
        if len(set(rows[0])) != len(rows[0]):
            raise ValueError("duplicate column names")
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            raise ValueError("inconsistent CSV field count")
        # 문자열을 그대로 전달하므로 '1.9'가 검증 전에 1로 바뀌지 않는다.
        return pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    except (UnicodeError, csv.Error, ValueError) as error:
        raise CsvDataError(f"CSV 형식 오류, 원본을 보존합니다: {path.name}: {error}") from error


def load_history(filename, *, missing_ok=False):
    """전체 연속 회차·날짜·본번호·보너스 검증 후 숫자형으로 반환한다."""
    frame = read_csv_checked(filename, missing_ok=missing_ok)
    if frame is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        if not set(HISTORY_COLUMNS).issubset(frame.columns):
            raise ValueError("missing history columns")
        validate_history(frame)
        # 검증 통과 후에만 변환한다. 정렬은 허용하지만 삭제·중복 제거는 하지 않는다.
        for column in ["회차", "보너스"] + NUMBERS:
            frame[column] = frame[column].map(strict_int)
        return frame.sort_values("회차").reset_index(drop=True)
    except (ValueError, KeyError, TypeError) as error:
        raise CsvDataError(f"당첨 이력 검증 실패, 원본을 보존합니다: {error}") from error


def _missing(value):
    """평가 전의 빈 칸과 pandas의 결측값만 미평가 값으로 인정한다."""
    return (isinstance(value, str) and not value.strip()) or pd.isna(value)


def validate_prediction_history(frame):
    """예측의 신원을 확인하고 고정 CSV 스키마로 반환한다.

    회차·세트·본번호는 필수이며 임의 생성하지 않는다. 과거 기록의 숫자 순서도
    보존한다. 평가 열은 아직 발표 전이면 비어 있어도 되지만 값이 있으면 정수와
    범위를 확인한다. 적중 계산의 정확성은 이후 실제 결과와 재대조한다.
    """
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    required = ["예측회차", "세트"] + NUMBERS
    if not frame.columns.is_unique or not set(required).issubset(frame.columns):
        raise CsvDataError("예측회차·세트·번호1~6 열이 필요하며 중복 열은 허용하지 않습니다.")
    result = frame.copy()
    try:
        for column in required:
            result[column] = result[column].map(strict_int)
        if (result[["예측회차", "세트"]] < 1).any().any():
            raise ValueError("round and set must be positive")
        seen = set()
        for row in result.to_dict("records"):
            numbers = [row[column] for column in NUMBERS]
            if len(set(numbers)) != 6 or any(n < 1 or n > 45 for n in numbers):
                raise ValueError("invalid prediction numbers")
            ticket = (row["예측회차"], tuple(sorted(numbers)))
            if ticket in seen:
                raise ValueError("duplicate tickets within a round")
            seen.add(ticket)
        if result.duplicated(["예측회차", "세트"]).any():
            raise ValueError("duplicate prediction round/set")
        for column in PREDICTION_COLUMNS:
            if column not in result:
                result[column] = pd.NA
        for column in ACTUAL_COLUMNS + ["본번호일치수", "보너스일치", "총일치수"]:
            maximum = 45 if column in ACTUAL_COLUMNS else {"본번호일치수": 6, "보너스일치": 1, "총일치수": 7}[column]
            minimum = 1 if column in ACTUAL_COLUMNS else 0
            values = []
            for value in result[column]:
                if _missing(value):
                    values.append(pd.NA)
                else:
                    number = strict_int(value)
                    if not minimum <= number <= maximum:
                        raise ValueError(f"invalid {column}")
                    values.append(number)
            result[column] = pd.array(values, dtype="Int64")
        for actual in result[ACTUAL_COLUMNS].itertuples(index=False, name=None):
            known = [n for n in actual if not pd.isna(n)]
            if known and (len(known) != 6 or len(set(known)) != 6):
                raise ValueError("partial or duplicate actual numbers")
        scores = []
        for value in result["종합점수"]:
            if _missing(value):
                scores.append(pd.NA)
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("non-finite score")
                scores.append(number)
        # 예전 모델의 점수도 임의로 /100 하거나 새 모델 점수로 바꾸지 않는다.
        result["종합점수"] = pd.array(scores, dtype="Float64")
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise CsvDataError(f"예측 이력 검증 실패, 원본을 보존합니다: {error}") from error
    return result[PREDICTION_COLUMNS].sort_values(["예측회차", "세트"]).reset_index(drop=True)


def load_predictions(filename, *, missing_ok=False):
    """없는 파일과 손상된 파일을 구분하고, 검증된 예측만 반환한다."""
    frame = read_csv_checked(filename, missing_ok=missing_ok)
    if frame is None:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return validate_prediction_history(frame)
