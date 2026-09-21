"""Verify the latest completed draw before generating a weekly batch."""
from datetime import datetime, timedelta, timezone

from lotto_validation import validate_draw, validate_history


def latest_completed_draw(now=None):
    korea = timezone(timedelta(hours=9))
    now = datetime.now(korea) if now is None else now.astimezone(korea)
    # Allow publication time after the Saturday draw.
    first = datetime(2002, 12, 7, 21, 0, tzinfo=korea)
    return max(0, int((now - first).total_seconds() // (7 * 86400)) + 1)


def verify_latest_result(history, fetch_draw, expected_round=None):
    latest = validate_history(history)[-1]
    if expected_round is not None and latest["회차"] != expected_round:
        raise ValueError(f"최신 당첨 결과가 필요합니다: {expected_round}회 (현재 {latest['회차']}회)")
    official = fetch_draw(latest["회차"])
    if official is None:
        raise ValueError("공식 당첨 결과 확인에 실패했습니다. 연결 복구 후 다시 실행하세요.")
    if validate_draw(official) != latest:
        raise ValueError("공식 당첨 결과와 저장된 지난주 결과가 다릅니다.")
    return latest
