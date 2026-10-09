"""15분 단위 예측 (보고서 제2장 [표9], [그림20])

(1) 전날 12시에 다음 날 96개 15분 구간 예측 — PeakSense를 15분 값으로 학습(시간 특성 + 구간 번호)
(2) 당일 갱신: 예측 = 전날 예측 + β × 최근 w구간 평균 오차, β·w는 7월 자료로만 정함(평가는 8/1~9/14)
"""
import numpy as np
import pandas as pd

from .config import CAL_END, EVAL_END, EVAL_START, HORIZONS, ISSUE_HOUR, Q15_STEP_DAYS, QCOLS
from .models import PeakSense


def to_15min(df):
    v = df[QCOLS].to_numpy(dtype=float).ravel()
    idx = pd.DatetimeIndex(np.repeat(df.index.values, 4)) + pd.to_timedelta(np.tile([0, 15, 30, 45], len(df)), unit="min")
    return pd.Series(v, index=idx)


def expand_hourly(X, cols):
    E = X[cols].loc[X.index.repeat(4)].copy()
    E.index = E.index + pd.to_timedelta(np.tile([0, 15, 30, 45], len(X)), unit="min")
    E["quarter"] = np.tile([0, 1, 2, 3], len(X))
    return E


def _s(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    nz = y > 0
    return {"MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAPE(%)": float(np.mean(np.abs(y[nz] - p[nz]) / y[nz]) * 100)}


def run(df, X, feats):
    v = to_15min(df)
    ok = v.notna().to_numpy()
    E = expand_hourly(X, feats)
    E["lag_week"] = v.shift(96 * 7)
    cols = feats + ["quarter"]
    preds = []
    t, end = pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END) + pd.Timedelta(hours=23, minutes=45)
    while t <= end:
        t_end = min(t + pd.Timedelta(days=Q15_STEP_DAYS) - pd.Timedelta(minutes=15), end)
        issue = t - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
        tr, te = ok & (v.index < issue), ok & (v.index >= t) & (v.index <= t_end)
        m = PeakSense().fit(E.loc[tr, cols], v[tr])
        preds.append(pd.Series(m.predict(E.loc[te, cols]), index=v.index[te]))
        t = t_end + pd.Timedelta(minutes=15)
    day = pd.concat(preds)

    full = pd.Series(np.nan, index=v.index)
    full[day.index] = day
    resid = v - full
    cal_end = pd.Timestamp(CAL_END)
    roll, calib = {}, []
    for h in HORIZONS:
        best = None
        for w in (1, 2, 4):
            x = resid.rolling(w, min_periods=1).mean().shift(h)
            d = pd.DataFrame({"y": v, "p": full, "x": x}).dropna()
            c = d[d.index < cal_end]
            b = float((c.x * (c.y - c.p)).sum() / (c.x ** 2).sum())
            mae = float(np.abs(c.y - c.p - b * c.x).mean())
            if best is None or mae < best[0]:
                best = (mae, w, b)
        _, w, b = best
        x = resid.rolling(w, min_periods=1).mean().shift(h)
        roll[h] = (full + b * x).where(x.notna(), full)
        calib.append({"h": h, "w": w, "beta": round(b, 3)})

    idx = day.index[day.index >= cal_end]
    y = v[idx]
    lw = E.loc[idx, "lag_week"]
    m = lw.notna()
    lab = {1: "15분 앞", 2: "30분 앞", 4: "1시간 앞", 8: "2시간 앞"}
    rows = [{"모델": "전일 15분 예측", "시계": "다음 날 96구간", **_s(y, day[idx])},
            {"모델": "기준: 지난주 같은 구간", "시계": "다음 날 96구간", **_s(y[m], lw[m])}]
    for h in HORIZONS:
        pers = v.shift(h)[idx]
        k = pers.notna()
        rows.append({"모델": "당일 갱신", "시계": lab[h], **_s(y, roll[h][idx])})
        rows.append({"모델": "기준: 직전 15분 값 유지", "시계": lab[h], **_s(y[k], pers[k])})
    return pd.DataFrame(rows), calib
