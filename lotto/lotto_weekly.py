"""Verify the latest completed draw before generating a weekly batch."""
from datetime import datetime, timedelta, timezone
from numbers import Integral

KOREA = timezone(timedelta(hours=9))
FIRST_RESULT_CHECK = datetime(2002, 12, 7, 21, tzinfo=KOREA)


def _korean_time(now):
    return datetime.now(KOREA) if now is None else now.astimezone(KOREA)


def latest_completed_draw(now=None):
    """한국 시간 토요일 21시 기준 조회 대상 회차를 계산한다. 공개 성공 판정은 아니다."""
    return max(0, (_korean_time(now) - FIRST_RESULT_CHECK) // timedelta(weeks=1) + 1)


class PredictionWindowError(ValueError):
    """현재 시각으로 사전 확정할 수 없는 회차."""


def verify_prediction_window(target, now=None):
    """공식 결과 조회 기준과 별도로 해당 회차 토요일 20시부터 확정을 막는다."""
    current = _korean_time(now)
    expected = latest_completed_draw(current) + 1
    if isinstance(target, bool) or not isinstance(target, Integral) or target != expected:
        raise PredictionWindowError(f"현재 검토 대상은 {expected}회입니다. 요청한 {target}회는 확정할 수 없습니다.")
    deadline = FIRST_RESULT_CHECK.replace(hour=20) + timedelta(weeks=expected - 1)
    if current >= deadline:
        raise PredictionWindowError(f"{target}회 판매 마감({deadline:%Y-%m-%d %H:%M} 한국시간)이 지났습니다.")
    return expected


def verify_latest_result(history, fetch_draw, expected_round=None):
    """전체 저장 이력을 검증하고 마지막 회차 핵심 필드가 공식 응답과 같은지 확인한다."""
    from lotto_validation import validate_draw, validate_history

    latest = validate_history(history)[-1]
    if expected_round is not None and latest["회차"] != expected_round:
        raise ValueError(f"최신 당첨 결과가 필요합니다: {expected_round}회 (현재 {latest['회차']}회)")
    official = fetch_draw(latest["회차"])
    if official is None:
        raise ValueError("공식 당첨 결과 확인에 실패했습니다. 연결 복구 후 다시 실행하세요.")
    if validate_draw(official) != latest:
        raise ValueError("공식 당첨 결과와 저장된 지난주 결과가 다릅니다.")
    return latest
