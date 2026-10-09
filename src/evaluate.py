"""시간 순서를 지키는 모델 평가

역할: 과거 자료로 학습하고 이후 기간을 평가하는 주별 분할과 MAE·RMSE·일 최대 MAE를 계산합니다.
입력과 호출: X, y, 유효 행 마스크, 비교할 모델 목록. weekly_folds(), walk_forward(), score().
출력과 범위: 시각별 예측 DataFrame과 지표 dict를 반환합니다. 현재 주별 학습 경계는 index < issue이며 ISSUE_HOUR=11입니다.
상세: src/README.md 및 docs/FILE_GUIDE.md
"""
import numpy as np
import pandas as pd

from .config import EVAL_END, EVAL_START, ISSUE_HOUR, STEP_DAYS


def score(y, p) -> dict:
    """MAE, RMSE, MAPE(실측 0 제외), 상위 10% 시간 MAE, 일 최대 MAE"""
    y, p = pd.Series(y, dtype=float), pd.Series(np.asarray(p, float), index=y.index)
    nz = y > 0
    top = y >= y.quantile(0.9)
    dm = pd.DataFrame({"y": y, "p": p}).resample("D").max().dropna()
    return {"MAE": float((y - p).abs().mean()), "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "MAPE(%)": float(((y[nz] - p[nz]).abs() / y[nz]).mean() * 100),
            "상위10% MAE": float((y[top] - p[top]).abs().mean()), "일최대 MAE": float((dm["y"] - dm["p"]).abs().mean())}


def weekly_folds(start=EVAL_START, end=EVAL_END, step=STEP_DAYS):
    """(예측 구간 시작, 끝, 학습 마감 시각)을 차례로 돌려준다.
    학습 경계 issue = 예측 구간 첫날의 전날 ISSUE_HOUR시.
    index < issue 조건이므로 기본값 11에서는 전날 10시 행까지 포함한다."""
    t, E = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    while t <= E:
        t_end = min(t + pd.Timedelta(days=step) - pd.Timedelta(hours=1), E)
        yield t, t_end, t - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
        t = t_end + pd.Timedelta(hours=1)


def walk_forward(X, y, valid, models: dict, start=EVAL_START, end=EVAL_END, step=STEP_DAYS):
    """models: 이름 → (모델 생성 함수, 입력 특성 목록). 모든 모델을 같은 주·같은 학습 범위로 학습/예측한다.
    PeakSense류는 가동 확률(이름_p)도 함께 저장한다."""
    out = []
    for t, t_end, issue in weekly_folds(start, end, step):
        tr = valid & (X.index < issue)
        te = valid & (X.index >= t) & (X.index <= t_end)
        part = pd.DataFrame({"y": y[te]}, index=X.index[te])
        for name, (make, cols) in models.items():
            m = make().fit(X.loc[tr, cols], y[tr])
            if hasattr(m, "predict_parts"):
                p, pr = m.predict_parts(X.loc[te, cols])
                part[name], part[name + "_p"] = pr, p
            else:
                part[name] = m.predict(X.loc[te, cols])
        out.append(part)
    return pd.concat(out)
