"""Verify the latest completed draw before generating a weekly batch."""
from datetime import datetime, timedelta, timezone

from lotto_validation import validate_draw, validate_history


def latest_completed_draw(now=None):
    """한국 시간 토요일 21시 기준 조회 대상 회차를 계산한다. 공개 성공 판정은 아니다."""
    korea = timezone(timedelta(hours=9))
    now = datetime.now(korea) if now is None else now.astimezone(korea)
    # 추첨 뒤 공개 여유 시간을 둔 조회 기준이다. 실제 발표 여부는 공식 응답으로 확인한다.
    first = datetime(2002, 12, 7, 21, 0, tzinfo=korea)
    return max(0, int((now - first).total_seconds() // (7 * 86400)) + 1)


def verify_latest_result(history, fetch_draw, expected_round=None):
    """전체 저장 이력을 검증하고 마지막 회차 핵심 필드가 공식 응답과 같은지 확인한다."""
    latest = validate_history(history)[-1]
    if expected_round is not None and latest["회차"] != expected_round:
        raise ValueError(f"최신 당첨 결과가 필요합니다: {expected_round}회 (현재 {latest['회차']}회)")
    official = fetch_draw(latest["회차"])
    if official is None:
        raise ValueError("공식 당첨 결과 확인에 실패했습니다. 연결 복구 후 다시 실행하세요.")
    if validate_draw(official) != latest:
        raise ValueError("공식 당첨 결과와 저장된 지난주 결과가 다릅니다.")
    return latest
