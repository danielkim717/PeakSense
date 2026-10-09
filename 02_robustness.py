"""주별 안정성과 기준값 민감도 평가

목적: 주별·일별 성능 차이, 상태 기준 변경, 예측구간과 고전력 경보를 살펴봅니다.
입력: 원본 CSV와 01_model_comparison.py의 ch2_predictions.csv. 실행: python 02_robustness.py
출력: outputs/results/ch2_robustness.json 및 outputs/figures/fig18_weekly_mae.png.
해석: 잔차를 이용한 예측구간·경보에는 시간 경계 재검증이 필요합니다. docs/LIMITATIONS.md 참고.
상세: docs/FILE_GUIDE.md
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import EVAL_START, FIG, HIGH_THR, RES, setup_plot
from src.evaluate import score, walk_forward, weekly_folds
from src.features import build_features, final_features
from src.models import PeakSense
from src.preprocess import load_data

warnings.filterwarnings("ignore")
plt = setup_plot()


def main():
    R = {}
    P = pd.read_csv(RES / "ch2_predictions.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")

    # ---- 주별 안정성 + 통계 검정
    wk = P.index.to_period("W-WED")                 # 목요일 시작 주(테스트 첫날 7/1이 목요일)
    weekly = {k: P.groupby(wk).apply(lambda g, k=k: float((g[k] - g["y"]).abs().mean())).round(2).tolist()
              for k in ["lgbm", "peaksense"]}
    daily = {k: (P[k] - P["y"]).abs().groupby(P.index.normalize()).mean() for k in ["rf", "lgbm", "peaksense"]}
    R["weekly"] = {"labels": [str(p.start_time.date()) for p in wk.unique()], **weekly,
                   "wins_vs_lgbm": int(sum(a < b for a, b in zip(weekly["peaksense"], weekly["lgbm"])))}
    R["wilcoxon"] = {"vs_lgbm_p": float(wilcoxon(daily["peaksense"], daily["lgbm"], alternative="less").pvalue),
                     "vs_rf_p": float(wilcoxon(daily["peaksense"], daily["rf"], alternative="less").pvalue),
                     "days_better_vs_lgbm": int((daily["peaksense"] < daily["lgbm"]).sum()), "n_days": int(len(daily["lgbm"]))}

    lab = [l[5:].replace("-", "/") + "~" for l in R["weekly"]["labels"]]
    x = np.arange(len(lab))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.19, weekly["lgbm"], 0.36, color="#c6ced3", label="④ LightGBM 단일 모델")
    ax.bar(x + 0.19, weekly["peaksense"], 0.36, color="#1d6b8c", label="⑤ PeakSense")
    for i, v in enumerate(weekly["peaksense"]):
        ax.text(i + 0.19, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color="#1d6b8c")
    ax.set_xticks(x, lab, fontsize=9); ax.set_ylabel("주별 MAE"); ax.set_ylim(0, max(weekly["lgbm"]) * 1.15)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig18_weekly_mae.png", dpi=160); plt.close()
    print("주별", R["weekly"]["wins_vs_lgbm"], "/", len(lab), R["wilcoxon"], flush=True)

    df = load_data()
    X = build_features(df)
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X)

    # ---- 민감도: 가동 기준, 라벨 정의
    sens = {}
    for thr in (50, 70, 90):
        Q = walk_forward(X, y, valid, {"m": (lambda t=thr: PeakSense(run_thr=t), F)})
        sens[f"가동 기준 {thr}"] = score(Q["y"], Q["m"])
    Q = walk_forward(X, y, valid, {"m": (lambda: PeakSense(label="production"), F)})
    sens["라벨: 생산량 > 0"] = score(Q["y"], Q["m"])
    R["sensitivity"] = sens
    print({k: round(v["MAE"], 2) for k, v in sens.items()}, flush=True)

    # ---- 한 번만 학습(6/30 11시까지) → 테스트 전체
    tr = valid & (X.index < pd.Timestamp("2021-06-30 11:00"))
    te = valid & (X.index >= EVAL_START)
    m = PeakSense().fit(X.loc[tr, F], y[tr])
    R["train_once"] = score(y[te], pd.Series(m.predict(X.loc[te, F]), index=X.index[te]))

    # ---- 분기 기준·예측구간·경보·실행시간 (6월부터 사후 예측을 쌓아 이전 주 오차로 구간 생성)
    rows, fit_t, pred_t = [], [], []
    for t, t_end, issue in weekly_folds("2021-06-03"):
        tr = valid & (X.index < issue)
        te = valid & (X.index >= t) & (X.index <= t_end)
        t1 = time.time(); m = PeakSense().fit(X.loc[tr, F], y[tr]); fit_t.append(time.time() - t1)
        t2 = time.time(); p, mr, mi = m.parts(X.loc[te, F]); pred_t.append((time.time() - t2) / max(int(te.sum()), 1) * 1000)
        rows.append(pd.DataFrame({"y": y[te], "p": p, "mr": mr, "mi": mi}, index=X.index[te]))
    Q = pd.concat(rows)
    ev = Q.index >= EVAL_START
    R["route_thr"] = {str(th): score(Q.loc[ev, "y"], pd.Series(np.where(Q["p"] >= th, Q["mr"], Q["mi"]), index=Q.index)[ev])
                      for th in (0.3, 0.5, 0.7)}
    R["timing"] = {"fit_s_mean": float(np.mean(fit_t)), "pred_ms_per_hour": float(np.mean(pred_t))}

    Q["pred"] = np.where(Q["p"] >= 0.5, Q["mr"], Q["mi"])
    Q["res"] = Q["y"] - Q["pred"]
    Q["grp"] = (Q["p"] >= 0.5).astype(int).astype(str) + ((Q.index.hour >= 8) & (Q.index.hour <= 17)).astype(int).astype(str)
    Q["wk"] = Q.index.to_period("W-WED")
    lo, hi, pex = {}, {}, {}
    for (w, g), part in Q.groupby(["wk", "grp"]):
        pool = Q.loc[(Q["wk"] < w) & (Q["grp"] == g), "res"].to_numpy()
        if len(pool) < 30:
            pool = Q.loc[Q["wk"] < w, "res"].to_numpy()
        if len(pool) < 30:
            continue
        ql, qh = np.quantile(pool, [0.05, 0.95])
        for ix, pv in zip(part.index, part["pred"]):
            lo[ix], hi[ix], pex[ix] = pv + ql, pv + qh, float(((pv + pool) >= HIGH_THR).mean())
    E = Q[ev].join(pd.DataFrame({"lo": lo, "hi": hi, "pex": pex})).dropna(subset=["lo"])
    act, alarm = E["y"] >= HIGH_THR, E["pex"] >= 0.3
    R["interval"] = {"coverage90": float(((E.y >= E.lo) & (E.y <= E.hi)).mean()), "mean_width": float((E.hi - E.lo).mean())}
    R["alarm"] = {"고전력 시간": int(act.sum()), "적중": int((alarm & act).sum()), "미탐지": int((~alarm & act).sum()),
                  "오경보": int((alarm & ~act).sum()), "재현율": float((alarm & act).sum() / act.sum()),
                  "정밀도": float((alarm & act).sum() / alarm.sum())}
    (RES / "ch2_robustness.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("한 번 학습", round(R["train_once"]["MAE"], 2), "| 분기", {k: round(v["MAE"], 2) for k, v in R["route_thr"].items()})
    print("구간", R["interval"], "| 경보", R["alarm"], "| 시간", R["timing"])


if __name__ == "__main__":
    main()
