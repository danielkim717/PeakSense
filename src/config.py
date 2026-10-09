"""공통 설정: 경로, 상수, 모델 하이퍼파라미터 (보고서 제2장 [표3]과 동일)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_NAME = "okm_augumented_2021.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
RES = OUT / "results"
MODELS = ROOT / "models"

# ---- 데이터·평가
QCOLS = ["15분", "30분", "45분", "60분"]   # 한 시간을 네 15분 구간으로 나눈 전력값
ISSUE_HOUR = 11                            # 예측 시점: 예측일 전날 12시(전날 11시 관측까지 사용)
EVAL_START, EVAL_END = "2021-07-01", "2021-09-14"   # 테스트 구간(주 단위 재학습 11회)
STEP_DAYS = 7                              # 재학습 주기(일)
HIGH_THR = 187                             # 고전력(피크) 기준: 시간 최대수요 ≥ 187 (제1장)

# ---- PeakSense
RUN_THR = 70.0       # 가동 라벨: 시간 최대수요 > 70 이면 가동(1), 아니면 정지·대기(0)
ROUTE_THR = 0.5      # 가동 확률 ≥ 0.5 → 가동 상태 회귀기, 아니면 정지 상태 회귀기
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=15,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, n_jobs=4, verbose=-1)
HUBER = dict(objective="huber", alpha=15)
RF = dict(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=0)

# ---- 15분 예측
HORIZONS = [1, 2, 4, 8]                    # 15분, 30분, 1시간, 2시간 앞
CAL_END = "2021-08-01"                     # 당일 보정 계수 추정은 7월, 15분 예측 평가는 8/1~9/14
Q15_STEP_DAYS = 14


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 한글 글꼴 (macOS: AppleGothic)
    plt.rcParams["axes.unicode_minus"] = False
    for d in (FIG, RES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    return plt
