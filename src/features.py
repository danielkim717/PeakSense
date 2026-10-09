"""입력 특성 생성 (보고서 제2장 [표1])

최종 입력 = 생산계획(+생산 흐름 파생) + 달력 + 기상. 모두 예측 시점(전날 12시)에 알 수 있는 값이다.
과거 전력 특성은 입력 비교(제2장 [표5])와 GRU 비교에만 쓰이며, 최종 모델에는 들어가지 않는다.
공장인원 = 생산량 ÷ (네 15분 전력의 합) 이므로 정답 전력이 들어 있어 입력에서 제외한다.
"""
import numpy as np
import pandas as pd

from .config import ISSUE_HOUR

PLAN = ["prod", "prod_log", "run", "prod_prev1", "prod_prev2", "prod_next1", "prod_next2", "startup", "shutdown",
        "run_streak", "idle_streak", "hours_to_stop", "prod_day_total", "run_day_hours", "prod_roll3"]
CALENDAR = ["hour", "dow", "sunday", "saturday", "인건비"]
WEATHER = ["기온", "풍속", "습도", "강수량", "cdd", "hdd"]
FINAL = CALENDAR + PLAN + WEATHER                    # 최종 모델 입력 26개
MORNING = ["mean_12h", "last_obs"]                    # 전날 오전 전력(입력 비교용)
HIST_ALL = MORNING + ["max_168h", "lag48", "lag168", "lag24_am", "same_state_lag", "run_level_7d", "idle_level_7d"]
GUIDE = ["hour", "prod", "기온", "풍속", "습도", "강수량", "dow", "인건비"]  # 가이드북 방식 랜덤포레스트 입력


def plan_features(p: pd.Series) -> pd.DataFrame:
    """생산계획(시간별 계획 생산량)만으로 계산하는 생산 흐름 특성"""
    X = pd.DataFrame(index=p.index)
    run = (p > 0).astype(int)
    X["prod"] = p
    X["prod_log"] = np.log1p(p)
    X["run"] = run
    for k in (1, 2):
        X[f"prod_prev{k}"] = p.shift(k).fillna(0)
        X[f"prod_next{k}"] = p.shift(-k).fillna(0)
    X["startup"] = ((run == 1) & (run.shift(1, fill_value=0) == 0)).astype(int)    # 생산 시작(0→양수)
    X["shutdown"] = ((run == 0) & (run.shift(1, fill_value=0) == 1)).astype(int)   # 생산 종료(양수→0)
    grp = (run != run.shift()).cumsum()
    streak = run.groupby(grp).cumcount() + 1
    X["run_streak"] = np.where(run == 1, streak, 0).clip(max=72)                   # 연속 생산시간
    X["idle_streak"] = np.where(run == 0, streak, 0).clip(max=72)                  # 연속 비생산시간
    X["hours_to_stop"] = np.where(run == 1, run.groupby(grp).transform("size") - streak, 0).clip(max=72)
    day = p.index.normalize()
    X["prod_day_total"] = p.groupby(day).transform("sum")
    X["run_day_hours"] = run.groupby(day).transform("sum")
    X["prod_roll3"] = p.rolling(3, center=True, min_periods=1).sum()
    return X


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.index
    y = df["peak"].where(df["peak"] > 0).astype(float)
    X = pd.DataFrame(index=ts)
    X["hour"] = ts.hour
    X["dow"] = ts.dayofweek
    X["sunday"] = (ts.dayofweek == 6).astype(int)
    X["saturday"] = (ts.dayofweek == 5).astype(int)
    X["인건비"] = df["인건비"]                         # 주간 1.0 / 야간·휴일 1.5
    X = X.join(plan_features(df["생산량"]))
    for c in ["기온", "풍속", "습도", "강수량"]:
        X[c] = df[c]
    X["cdd"] = (df["기온"] - 24).clip(lower=0)        # 냉방 필요 온도
    X["hdd"] = (5 - df["기온"]).clip(lower=0)         # 난방 필요 온도

    # ---- 과거 전력(입력 비교·GRU 전용): 예측 시점(전날 11시)까지 관측된 값만
    issue = ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    run_hist = (df["생산량"] > 0).astype(float)
    snap = pd.DataFrame(index=ts)
    snap["last_obs"] = y
    snap["mean_12h"] = y.rolling(12, min_periods=6).mean()
    snap["max_168h"] = y.rolling(168, min_periods=48).max()
    snap["run_level_7d"] = y.where(run_hist == 1).rolling(168, min_periods=6).mean()
    snap["idle_level_7d"] = y.where(run_hist == 0).rolling(168, min_periods=6).median()
    S = snap.ffill().reindex(issue)
    S.index = ts
    X = X.join(S)
    X["lag48"] = y.shift(48)
    X["lag168"] = y.shift(168)
    X["lag24_am"] = np.where(ts.hour <= ISSUE_HOUR, y.shift(24), np.nan)
    same = pd.Series(np.nan, index=ts)
    for k in range(2, 8):
        same = same.fillna(y.shift(24 * k).where(run_hist.shift(24 * k) == X["run"]))
    X["same_state_lag"] = same
    return X


def final_features(X: pd.DataFrame) -> list[str]:
    """최종 입력 26개를 build_features의 열 순서대로 반환(재현성: LightGBM 결과가 열 순서에 영향받음)"""
    keep = set(FINAL)
    return [c for c in X.columns if c in keep]
