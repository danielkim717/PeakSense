"""[8단계, 확장] 오류분석 결과를 반영한 모델 개선과 예측구간 재보정 → 보고서 제5장
  (1) PeakSense+: 제3장 오류분석(미탐지 85%가 13~19시, 45%가 28℃ 이상, 고전력 과소예측 −12.7)을 반영
      - 기온×시간 결합 특성 4개: 주간(08~19시) 냉방 필요 온도, 주간 28℃ 이상 여부, 가동×냉방 필요 온도, 생산량×냉방 필요 온도
      - 피크 가중 학습: 가동 상태 회귀기 학습에서 시간 최대수요 170 이상 시간에 가중치 3
      - 선정 절차: 6월(테스트 이전) walk-forward로 후보 6개를 비교해 고른 뒤, 7~9월 테스트에서 한 번만 확인
  (2) 적응형 예측구간: 기존(주 단위로 이전 주 잔차 분위수) → 매일 잔차 갱신 + 적응형 보정(ACI, Gibbs & Candes 2021)
      - 하루 적중률이 목표(90%)보다 낮으면 다음 예측의 구간을 넓히고, 높으면 좁힌다. 결과는 이틀 뒤 예측부터 반영(발행 시점 준수)
  결과: outputs/results/ext_model.json, figures/fig_ext_model.png   (약 10분)
"""
import json
import warnings

import numpy as np
import pandas as pd

from src.config import FIG, HIGH_THR, HUBER, RES, setup_plot
from src.evaluate import score, walk_forward, weekly_folds
from src.features import build_features, final_features
from src.models import PeakSense, lgb_reg
from src.preprocess import load_data

warnings.filterwarnings("ignore")
plt = setup_plot()
INTER = ["cdd_day", "hot_day", "cdd_run", "cdd_prod"]


def add_interactions(X):
    day = ((X["hour"] >= 8) & (X["hour"] <= 19)).astype(int)
    X["cdd_day"] = X["cdd"] * day
    X["hot_day"] = ((X["기온"] >= 28) & (day == 1)).astype(int)
    X["cdd_run"] = X["cdd"] * X["run"]
    X["cdd_prod"] = X["cdd"] * X["prod_log"]
    return X


class PeakSensePlus(PeakSense):
    """PeakSense + 피크 가중 학습(가동 상태 회귀기에서 시간 최대수요 ≥ w_thr 시간에 가중치 w)"""

    def __init__(self, w=3.0, w_thr=170, alpha=HUBER["alpha"]):
        super().__init__()
        self.w, self.w_thr = w, w_thr
        self.run = lgb_reg(objective="huber", alpha=alpha)

    def fit(self, X, y):
        s = self._label(X, y)
        self.clf.fit(X, s)
        self.run.fit(X[s == 1], y[s == 1], sample_weight=np.where(y[s == 1] >= self.w_thr, self.w, 1.0))
        self.idle.fit(X[s == 0], y[s == 0])
        return self


def peak_stats(Q, k):
    act, e = Q.y >= HIGH_THR, Q[k] - Q.y
    s = score(Q.y, Q[k])
    return {"MAE": s["MAE"], "일최대 MAE": s["일최대 MAE"], "고전력 MAE": float(e[act].abs().mean()),
            "고전력 평균 오차": float(e[act].mean()), "예측 187 이상 적중": int(((Q[k] >= HIGH_THR) & act).sum()),
            "고전력 시간": int(act.sum())}


def adaptive_interval(Q, target=0.10, gamma=0.0, daily=True, start="2021-07-01"):
    """Q: y, pred, res, grp. 매일(또는 매주) 이전 잔차로 분위수 구간, gamma>0이면 ACI로 유의수준 조정"""
    days = Q.index.normalize().unique()
    a, pend, out = target, [], []
    for d in days[days >= start]:
        cut = d - pd.Timedelta(days=1) if daily else d - pd.Timedelta(days=(d.dayofweek - 3) % 7)
        hist = Q[Q.index < cut]
        part = Q[Q.index.normalize() == d]
        ok_day = []
        for g, pg in part.groupby("grp"):
            pool = hist.loc[hist.grp == g, "res"].to_numpy()
            if len(pool) < 30:
                pool = hist["res"].to_numpy()
            aa = float(np.clip(a, 0.005, 0.5))
            lo, hi = np.quantile(pool, [aa / 2, 1 - aa / 2])
            ok = (pg.y >= pg.pred + lo) & (pg.y <= pg.pred + hi)
            out.append(pd.DataFrame({"ok": ok, "w": hi - lo, "lo": pg.pred + lo, "hi": pg.pred + hi}, index=pg.index))
            ok_day += list(ok)
        pend.append(1 - np.mean(ok_day))
        if gamma and len(pend) >= 2:          # 오늘 결과는 내일 12시 발행되는 모레 예측부터 반영
            a = a + gamma * (target - pend[-2])
    return pd.concat(out).sort_index()


def posthoc(X, y, valid, make, cols):
    """6월부터 주 단위로 사후 예측을 쌓는다(예측구간·경보 계산용, 제2장과 같은 절차)"""
    rows = []
    for t, t_end, issue in weekly_folds("2021-06-03"):
        tr = valid & (X.index < issue)
        te = valid & (X.index >= t) & (X.index <= t_end)
        m = make().fit(X.loc[tr, cols], y[tr])
        p, pr = m.predict_parts(X.loc[te, cols])
        rows.append(pd.DataFrame({"y": y[te], "p": p, "pred": pr}, index=X.index[te]))
    Q = pd.concat(rows)
    Q["res"] = Q.y - Q.pred
    Q["grp"] = (Q.p >= .5).astype(int).astype(str) + ((Q.index.hour >= 8) & (Q.index.hour <= 17)).astype(int).astype(str)
    return Q


def alarm(Q):
    Q = Q.copy()
    Q["wk"] = Q.index.to_period("W-WED")
    pex = {}
    for (w, g), part in Q.groupby(["wk", "grp"]):
        pool = Q.loc[(Q.wk < w) & (Q.grp == g), "res"].to_numpy()
        if len(pool) < 30:
            pool = Q.loc[Q.wk < w, "res"].to_numpy()
        if len(pool) < 30:
            continue
        for ix, pv in zip(part.index, part.pred):
            pex[ix] = float(((pv + pool) >= HIGH_THR).mean())
    E = Q[Q.index >= "2021-07-01"].join(pd.Series(pex, name="pex")).dropna(subset=["pex"])
    act, al = E.y >= HIGH_THR, E.pex >= 0.3
    miss = E[act & ~al]
    return {"적중": int((al & act).sum()), "미탐지": int((~al & act).sum()), "오경보": int((al & ~act).sum()),
            "재현율": float((al & act).sum() / act.sum()), "정밀도": float((al & act).sum() / al.sum()),
            "미탐지 중 13~19시": int(((miss.index.hour >= 13) & (miss.index.hour <= 19)).sum())}


def main():
    R = {}
    df = load_data()
    X = add_interactions(build_features(df))
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X)
    FI = F + INTER

    # ---- (1) 후보 비교: 6월 검증으로 선정 → 7~9월 테스트에서 확인
    cands = {"PeakSense": (PeakSense, F), "결합 특성": (PeakSense, FI),
             "Huber 기준 30": (lambda: PeakSensePlus(1.0, 170, 30), F), "피크 가중": (PeakSensePlus, F),
             "피크 가중 + 결합 특성": (PeakSensePlus, FI),
             "피크 가중 + 결합 특성 + Huber 30": (lambda: PeakSensePlus(3.0, 170, 30), FI)}
    T = None
    for nm, (s, e) in {"validation_june": ("2021-06-03", "2021-06-30"), "test": ("2021-07-01", "2021-09-14")}.items():
        Q = walk_forward(X, y, valid, cands, start=s, end=e)
        R[nm] = {k: peak_stats(Q, k) for k in cands}
        print(nm, {k: round(v["MAE"], 3) for k, v in R[nm].items()}, flush=True)
        T = Q
    sel = min(R["validation_june"], key=lambda k: R["validation_june"][k]["MAE"])
    R["selected"] = sel
    w = T.index.to_period("W-WED")
    R["weekly_wins"] = int(sum((T[sel] - T.y).abs().groupby(w).mean() < (T["PeakSense"] - T.y).abs().groupby(w).mean()))
    R["n_weeks"] = int(w.nunique())

    # ---- 경보(고전력 확률 ≥ 0.3) 비교
    QA = {"PeakSense": posthoc(X, y, valid, PeakSense, F), sel: posthoc(X, y, valid, *cands[sel])}
    R["alarm"] = {k: alarm(Q) for k, Q in QA.items()}

    # ---- (2) 예측구간: 기존 PeakSense 예측값(제2장과 동일)에 대해 보정 방식만 비교
    Q = QA["PeakSense"]
    variants = {"기존(주 단위)": dict(daily=False), "매일 갱신": dict(), "매일 갱신 + 적응형 보정": dict(gamma=0.05)}
    R["interval"], IV = {}, {}
    for k, kw in variants.items():
        E = adaptive_interval(Q, **kw)
        IV[k] = E
        R["interval"][k] = {"적중률": float(E.ok.mean()), "평균 폭": float(E.w.mean()),
                            "월별 적중률": {str(p): float(v) for p, v in E.groupby(E.index.to_period("M")).ok.mean().items()}}
    (RES / "ext_model.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("selected", "weekly_wins", "alarm", "interval")}, ensure_ascii=False, indent=1, default=float))

    # ---- 그림: (가) 고전력 시간의 시각별 평균 오차, (나) 주별 예측구간 적중률
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    act = T.y >= HIGH_THR
    hrs = np.arange(8, 20)
    for k, c, off, lb in (("PeakSense", "#9aa5ab", -0.2, "⑤ PeakSense"), (sel, "#1d6b8c", 0.2, "PeakSense+")):
        e = (T[k] - T.y)[act]
        b = [e[e.index.hour == h].mean() if (e.index.hour == h).any() else np.nan for h in hrs]
        ax[0].bar(hrs + off, b, 0.4, color=c, label=lb)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(hrs); ax[0].set_xlabel("시각"); ax[0].set_ylabel("평균 오차(예측 - 실측)")
    ax[0].set_title("(가) 고전력 시간의 시각별 평균 오차", fontsize=11)
    for k, c in (("기존(주 단위)", "#9aa5ab"), ("매일 갱신 + 적응형 보정", "#1d6b8c")):
        E = IV[k]
        s = E.groupby(E.index.to_period("W-WED")).ok.mean() * 100
        ax[1].plot(range(len(s)), s.values, "-o", ms=4, color=c, label=f"{k} ({E.ok.mean() * 100:.1f}%)")
    ax[1].axhline(90, ls="--", color="#c0392b", lw=1)
    ax[1].set_ylim(50, 100)
    ax[1].set_xticks(range(len(s)), [p.start_time.strftime("%m/%d") for p in s.index], fontsize=8)
    ax[1].set_ylabel("주별 적중률(%)"); ax[1].set_title("(나) 90% 예측구간의 주별 적중률", fontsize=11)
    for a_ in ax:
        a_.legend(frameon=False, fontsize=8); a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig_ext_model.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
