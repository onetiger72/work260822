"""서로 다른 6/45 조합 묶음에서 한 장 이상 본번호 k개를 맞힐 정확한 확률."""
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from lotto_validation import strict_int


TOTAL_OUTCOMES = comb(45, 6)


@lru_cache(maxsize=14)
def _subset_masks(size, count):
    """각 절반에서 count개를 선택하는 모든 방법을 한 번만 열거한다."""
    bits = [1 << i for i in range(size)]
    return np.fromiter((sum(group) for group in combinations(bits, count)),
                       dtype=np.uint32, count=comb(size, count))


def coverage(tickets):
    """공정 추첨에서 1~10장의 3·4·5·6개 이상 적중 결과 수와 확률을 반환한다.

    1~22와 23~45에서 각각 k개와 6-k개를 뽑는 일곱 경우가 전체 추첨을
    정확히 한 번씩 센다. 같은 추첨에서 여러 장이 적중해도 최고 적중수만 센다.
    random_probabilities는 같은 장수의 서로 다른 균등 무작위 조합에 대한 평균이다.
    """
    masks = []
    for ticket in tickets:
        if isinstance(ticket, (str, bytes)):
            raise ValueError("each ticket requires six numbers")
        numbers = [strict_int(value) for value in ticket]
        if len(numbers) != 6 or len(set(numbers)) != 6 or not all(1 <= n <= 45 for n in numbers):
            raise ValueError("each ticket requires six distinct numbers in 1..45")
        masks.append(sum(1 << (n - 1) for n in numbers))
    if not 1 <= len(masks) <= 10 or len(set(masks)) != len(masks):
        raise ValueError("coverage requires 1 to 10 distinct tickets")

    exact = np.zeros(7, dtype=np.int64)
    for lower_count in range(7):
        lower = _subset_masks(22, lower_count)
        upper = _subset_masks(23, 6 - lower_count)
        best = np.zeros((len(lower), len(upper)), dtype=np.uint8)
        for mask in masks:
            low_hits = np.bitwise_count(lower & (mask & ((1 << 22) - 1)))
            high_hits = np.bitwise_count(upper & (mask >> 22))
            np.maximum(best, low_hits[:, None] + high_hits, out=best)
        exact += np.bincount(best.ravel(), minlength=7)

    hit_counts = {str(k): int(exact[k:].sum()) for k in range(3, 7)}
    random_probabilities = {}
    # 고정된 정답에 대해 구매 가능한 전체 묶음에서 모든 장이 기준 미달인 묶음을 뺀다.
    batch_count = comb(TOTAL_OUTCOMES, len(masks))
    for k in range(3, 7):
        winning_tickets = sum(comb(6, hits) * comb(39, 6 - hits) for hits in range(k, 7))
        losing_batches = comb(TOTAL_OUTCOMES - winning_tickets, len(masks))
        random_probabilities[str(k)] = (batch_count - losing_batches) / batch_count
    return {"set_count": len(masks), "total_outcomes": TOTAL_OUTCOMES, "hit_counts": hit_counts,
            "hit_probabilities": {k: count / TOTAL_OUTCOMES for k, count in hit_counts.items()},
            "random_probabilities": random_probabilities}
