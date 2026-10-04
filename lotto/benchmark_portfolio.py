"""과거 비교의 호환 진입점. 동일한 검증 절차를 benchmark_audit에 위임한다.

모델·균등 분산·일반 무작위 기준을 같은 회차에서 비교한다. 별도의 비교 구현을
유지하면 기준이 달라질 수 있으므로 하나의 실행 경로를 사용한다. 파일은 저장하지 않는다.
"""
from benchmark_audit import main


if __name__ == "__main__":
    main()
