"""모델 비교와 26개 입력 선정

목적: 단순 기준·랜덤포레스트·단일 LightGBM·PeakSense를 비교하고, 입력과 모델 구조를 선택합니다.
입력: 원본 CSV. 실행: python 01_model_comparison.py
출력: outputs/results/ch2_main.json, ch2_predictions.csv 및 outputs/figures/fig16·17·19·20 그림.
해석: 시간 최대수요 평가와 15분 앞 예측은 평가 기간·정답 단위가 다릅니다. docs/RESULTS.md 참고.
상세: docs/FILE_GUIDE.md
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

from src import forecast15
from src.config import FIG, HIGH_THR, RES, setup_plot
from src.evaluate import score, walk_forward
from src.features import GUIDE, HIST_ALL, MORNING, PLAN, WEATHER, build_features, final_features
from src.models import GuideRF, PeakSense, SingleLGBM
from src.preprocess import load_data

warnings.filterwarnings("ignore")
plt = setup_plot()
T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)  # noqa: E731
NAMES = {"naive_last": "① 직전 전력 유지", "naive_week": "② 지난주 같은 시간", "rf": "③ 랜덤포레스트(가이드북)",
         "lgbm": "④ LightGBM 단일", "peaksense": "⑤ PeakSense"}


def main():
    df = load_data()
    X = build_features(df)
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X)
    log(f"유효 {int(valid.sum())}시간, 최종 입력 {len(F)}개")

    # ---- 모델 비교 (같은 테스트·같은 재학습 일정·같은 예측 시점)
    P = walk_forward(X, y, valid, {"rf": (GuideRF, GUIDE), "lgbm": (SingleLGBM, F), "peaksense": (PeakSense, F)})
    P["naive_last"] = X.loc[P.index, "last_obs"]
    P["naive_week"] = X.loc[P.index, "lag168"].fillna(X.loc[P.index, "last_obs"])
    comp = {NAMES[k]: score(P["y"], P[k]) for k in NAMES}
    hi = P["y"] >= HIGH_THR
    high = {NAMES[k]: float((P.loc[hi, k] - P.loc[hi, "y"]).abs().mean()) for k in NAMES}
    log("모델 비교")

    # ---- 입력 선정 (PeakSense 구조 고정)
    A = walk_forward(X, y, valid, {"morning": (PeakSense, F + MORNING), "hist_all": (PeakSense, F + HIST_ALL),
                                   "no_plan": (PeakSense, [f for f in F if f not in PLAN]),
                                   "no_wx": (PeakSense, [f for f in F if f not in WEATHER])})
    abl = {"생산계획+달력+기상(최종)": comp[NAMES["peaksense"]], "최종+전날 오전 전력": score(A["y"], A["morning"]),
           "최종+과거 전력 후보 전부": score(A["y"], A["hist_all"]), "최종−생산계획": score(A["y"], A["no_plan"]),
           "최종−기상": score(A["y"], A["no_wx"])}
    log("입력 선정")

    # ---- 구조 비교: 확률 가중 결합
    class Soft(PeakSense):
        def predict_parts(self, Xa):
            return self.parts(Xa)[0], self.predict_soft(Xa)
    S = walk_forward(X, y, valid, {"soft": (Soft, F)})
    struct = {"확률 가중 결합": score(S["y"], S["soft"]), "가동 판별 후 분할(최종)": comp[NAMES["peaksense"]]}

    # ---- 가동상태 판별 성능
    act = (P["y"] > 70).astype(int)
    pr = (P["peaksense_p"] >= 0.5).astype(int)
    plan_run = X.loc[P.index, "run"].astype(int)
    mism = plan_run != act
    clf = {"정확도": float((pr == act).mean()), "생산기록 기준 정확도": float((plan_run == act).mean()),
           "불일치 시간": int(mism.sum()), "불일치 교정 비율": float((pr[mism] == act[mism]).mean()),
           "TP": int(((pr == 1) & (act == 1)).sum()), "FP": int(((pr == 1) & (act == 0)).sum()),
           "FN": int(((pr == 0) & (act == 1)).sum()), "TN": int(((pr == 0) & (act == 0)).sum())}

    # ---- 15분 예측
    M15, calib = forecast15.run(df, X, F)
    log("15분 예측")

    R = {"n_valid_hours": int(valid.sum()), "n_test_hours": int(len(P)), "n_test_days": int(P.index.normalize().nunique()),
         "n_high": int(hi.sum()), "features": F, "comparison": comp, "high_power_mae": high, "ablation": abl,
         "structure": struct, "classifier": clf, "q15": M15.to_dict(orient="records"), "q15_calib": calib}
    (RES / "ch2_main.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    P.to_csv(RES / "ch2_predictions.csv", encoding="utf-8-sig")
    figures(P, comp, M15)
    for k, v in comp.items():
        print(f"{k:20s} MAE {v['MAE']:6.2f}  일최대 {v['일최대 MAE']:6.2f}")
    print({k: round(v["MAE"], 2) for k, v in abl.items()})
    print(M15.round(2).to_string(index=False))
    log("완료")


def figures(P, comp, M15):
    ACC, GREY, INK = "#1d6b8c", "#b8c2c9", "#13202a"
    # 그림16: 구조도
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 112); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y0, w, h, title, lines, fc="#eef1f3", ec="#9aa6ae", tc=INK):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y0 + h - 3.2, title, ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
        for i, l in enumerate(lines):
            ax.text(x + w / 2, y0 + h - 7.4 - i * 3.6, l, ha="center", va="center", fontsize=8.8, color="#4d5c67")

    def arrow(x1, y1, x2, y2, col="#4d5c67"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=col))
    box(1, 13, 20, 22, "입력 정보", ["생산계획(전날 확정)", "시각·요일·주야 구분", "기상(예보)", "※ 과거 전력·공장인원 제외"])
    box(26, 15, 22, 18, "① 가동상태 판별기", ["LightGBM 분류", "실제 고부하 가동일 확률 p", "(시간 최대수요 > 70)"], "#dcebf2", ACC, ACC)
    box(60, 27, 22, 15, "② 가동 상태 회귀기", ["LightGBM · Huber 손실", "가동 시간으로만 학습"], "#dcebf2", ACC, ACC)
    box(60, 4, 22, 15, "② 정지 상태 회귀기", ["LightGBM · Huber 손실", "정지·대기 시간으로만 학습"], "#dcebf2", ACC, ACC)
    box(87, 11, 24, 24, "출력", ["시간 최대수요 예측", "15분 구간 예측", "(당일 최근 실측으로 보정)", "90% 예측구간", "고전력(≥187) 확률"])
    arrow(21.6, 24, 25.6, 24); arrow(48.6, 27, 59.6, 34, ACC); arrow(48.6, 21, 59.6, 12, ACC)
    ax.text(53.2, 34, "p ≥ 0.5", ha="center", fontsize=9.5, color=ACC, fontweight="bold")
    ax.text(53.2, 12.4, "p < 0.5", ha="center", fontsize=9.5, color=ACC, fontweight="bold")
    arrow(82.6, 34, 86.6, 27); arrow(82.6, 11, 86.6, 19)
    plt.tight_layout(); plt.savefig(FIG / "fig16_architecture.png", dpi=170); plt.close()

    # 그림17: 모델 비교
    labs = ["① 직전 전력\n유지", "② 지난주\n같은 시간", "③ 랜덤\n포레스트", "④ LightGBM\n단일", "⑤ PeakSense"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, met, ttl in [(axes[0], "MAE", "시간 최대수요 MAE"), (axes[1], "일최대 MAE", "일 최대수요 MAE")]:
        vals = [comp[NAMES[k]][met] for k in NAMES]
        b = ax.bar(labs, vals, color=[GREY] * 4 + [ACC], width=0.62)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.015, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_title(ttl, fontsize=12); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9); ax.set_ylim(0, max(vals) * 1.15)
    plt.tight_layout(); plt.savefig(FIG / "fig17_model_comparison.png", dpi=160); plt.close()

    # 그림19: 예측 사례
    w = P.loc["2021-07-19":"2021-07-25"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(w.index, w["y"], color=INK, lw=1.4, label="실측")
    ax.plot(w.index, w["lgbm"], color="#e0a060", lw=1.1, label="④ LightGBM 단일 모델")
    ax.plot(w.index, w["peaksense"], color=ACC, lw=1.7, label="⑤ PeakSense")
    ax.axhline(HIGH_THR, color="#d9730d", ls=":", lw=1)
    ax.text(w.index[-1], 190, "고전력 기준 187", ha="right", fontsize=9, color="#d9730d")
    ax.set_ylim(0, 240); ax.set_ylabel("시간 최대수요")
    ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig19_week_example.png", dpi=160); plt.close()

    # 그림20: 15분 예측
    order = ["15분 앞", "30분 앞", "1시간 앞", "2시간 앞"]
    roll = [M15[(M15["시계"] == h) & (M15["모델"] == "당일 갱신")]["MAE"].iloc[0] for h in order]
    pers = [M15[(M15["시계"] == h) & (M15["모델"] == "기준: 직전 15분 값 유지")]["MAE"].iloc[0] for h in order]
    day = M15[M15["모델"] == "전일 15분 예측"]["MAE"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(order, pers, "o-", color="#9aa6ae", label="직전 15분 값 유지")
    ax.plot(order, roll, "o-", color=ACC, lw=2, label="PeakSense 15분 예측(최근 관측 보정)")
    ax.axhline(day, color=INK, ls=":", lw=1); ax.text(3, day + 0.3, f"전날 96구간 예측 {day:.2f}", ha="right", fontsize=9)
    for i, v in enumerate(roll):
        ax.text(i, v - 1.2, f"{v:.2f}", ha="center", fontsize=9, color=ACC)
    ax.set_ylabel("15분 전력 MAE"); ax.set_ylim(0, max(pers) * 1.15); ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig20_15min_horizons.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
