"""같은 장수의 두 생성 결과에서 5등 이상을 얻는 추첨 조합 수를 비교한다."""
from hashlib import sha256
from pathlib import Path

from lotto import generate_prediction_sets as generate_cluster
from lotto_coverage import coverage
from lotto_portfolio import NUMBER_COLUMNS, VERSION, portfolio_diagnostics


def describe_batch(batch, set_count):
    tickets = batch[NUMBER_COLUMNS].to_numpy()
    if len(batch) != set_count:
        raise ValueError(f"후보는 정확히 {set_count}세트여야 합니다.")
    return {
        "tickets": tickets.tolist(),
        "coverage": coverage(tickets),
        "overlap": portfolio_diagnostics(tickets),
    }


def generator_source_hashes():
    base = Path(__file__).resolve().parent
    sources = ("lotto.py", "로또 당첨번호 예측.py", "lotto_portfolio.py",
               "lotto_selection.py", "lotto_coverage.py")
    return {name: sha256((base / name).read_bytes()).hexdigest() for name in sources}


def select_weekly_portfolio(history, generate_screened, set_count=10):
    target = int(history["회차"].max()) + 1
    screened = generate_screened(history, target_draw=target, set_count=set_count)
    weekly = screened.attrs["weekly_review"]
    batches = {"screened": screened.rename(columns={"조합점수": "종합점수"})}
    if len(history) >= 2:
        batches["cluster"] = generate_cluster(history, set_count=set_count)
    comparisons = {name: describe_batch(batch, set_count) for name, batch in batches.items()}

    def priority(name):
        counts = comparisons[name]["coverage"]["hit_counts"]
        return counts["3"], counts["4"], name == "screened"

    selected = max(batches, key=priority)
    evidence = {
        "objective": "같은 장수에서 한 장 이상 본번호 3개 적중 이상. 동률이면 4개 이상, 다음은 screened.",
        "assumption": "모든 6/45 당첨 조합은 같은 확률이며 서로 다른 10장의 1등 확률은 같다.",
        "selected": selected,
        "compared": comparisons,
        "screened_version": VERSION,
        "screened_policy": weekly["selection_policy"],
        "selection_seed": weekly["selection_seed"],
        "selected_strength": weekly["selected_strength"],
        "effective_methods": weekly["effective_methods"],
        "method_screening": weekly["method_screening"],
        "review_rounds": weekly["review_rounds"],
        "mean_brier_losses": weekly["mean_brier_losses"],
        "source_sha256": generator_source_hashes(),
    }
    return batches[selected], evidence
