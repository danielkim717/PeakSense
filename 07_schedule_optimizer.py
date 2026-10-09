"""생산목표를 유지하는 일정 후보 탐색

목적: 시간별 생산량을 재배분하고 모델이 예상하는 피크를 비교하는 국소 탐색 실험입니다.
입력: 원본 CSV. 예: python 07_schedule_optimizer.py --day 2021-09-08 --shift 2
출력: outputs/results/ext_schedule.json, ext_schedule_days.csv 및 outputs/figures/fig_ext_schedule.png.
해석: 전역 최적해나 현장 저감 실적이 아닙니다. 목적함수 정규화와 누적 이동 제약은 docs/LIMITATIONS.md 참고.
상세: docs/FILE_GUIDE.md
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import FIG, HIGH_THR, RES, setup_plot
from src.evaluate import weekly_folds
from src.features import build_features, final_features
from src.models import PeakSense, SingleLGBM
from src.preprocess import load_data
from src.scheduler import DaySchedule

warnings.filterwarnings("ignore")
plt = setup_plot()
RISK = 180          # 예측 일 최대가 이 값 이상인 날을 '위험일'로 보고 별도 집계(예측의 고전력 과소 경향 반영)


def run_day(m, X, prod, F, day, cap, shift):
    idx = X.index[X.index.normalize() == day]
    plan0 = prod.loc[idx].to_numpy(float)
    S = DaySchedule(m, X, prod, F, day, cap)
    p0 = S.predict_many([plan0])[0]
    plan1, p1 = S.optimize(plan0, max_shift=shift)
    return idx, plan0, p0, plan1, p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day"); ap.add_argument("--shift", type=int, default=2)
    a = ap.parse_args()
    df = load_data(); X = build_features(df); y = df["peak"].astype(float)
    valid = y.notna().to_numpy(); F = final_features(X); prod = df["생산량"].fillna(0)

    rows, keep = [], {}
    for t, t_end, issue in weekly_folds():
        days = [d for d in pd.date_range(t, t_end.normalize()) if a.day is None or d == pd.Timestamp(a.day)]
        if not days:
            continue
        tr = valid & (X.index < issue)
        cap = float(prod[tr & (prod > 0).to_numpy()].quantile(0.95))     # 시간당 생산량 상한: 과거 기록의 95% 값
        m = PeakSense().fit(X.loc[tr, F], y[tr])
        chk = SingleLGBM().fit(X.loc[tr, F], y[tr])      # 교차 확인용 다른 모델: 최적화에 쓰지 않은 모델로도 낮아지는지
        for day in days:
            idx = X.index[X.index.normalize() == day]
            if len(idx) != 24 or not valid[X.index.get_indexer(idx)].all() or prod.loc[idx].sum() == 0:
                continue
            for sh in ([a.shift] if a.day else (1, 2)):
                idx, plan0, p0, plan1, p1 = run_day(m, X, prod, F, day, cap, sh)
                c0, c1 = DaySchedule(chk, X, prod, F, day, cap).predict_many([plan0, plan1])
                rows.append({"date": day.date().isoformat(), "이동 범위": sh, "예측 일최대(기존)": p0.max(),
                             "예측 일최대(조정)": p1.max(), "실측 일최대": float(y.loc[idx].max()),
                             "예측 180 이상 시간(기존)": int((p0 >= RISK).sum()), "예측 180 이상 시간(조정)": int((p1 >= RISK).sum()),
                             "이동 생산량 비율": float(np.abs(plan1 - plan0).sum() / 2 / plan0.sum()),
                             "교차확인 일최대(기존)": c0.max(), "교차확인 일최대(조정)": c1.max()})
                keep[(day, sh)] = (idx, plan0, p0, plan1, p1, y.loc[idx].to_numpy())
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    if a.day:
        print(D.to_string()); idx, plan0, p0, plan1, p1, act = keep[(pd.Timestamp(a.day), a.shift)]
        print(pd.DataFrame({"기존 계획": plan0, "조정 계획": plan1.round(0), "예측(기존)": p0.round(1), "예측(조정)": p1.round(1)}, index=idx.hour))
        return
    D.to_csv(RES / "ext_schedule_days.csv", index=False, encoding="utf-8-sig")
    R = {}
    for sh, g in D.groupby("이동 범위"):
        risk = g[g["예측 일최대(기존)"] >= RISK]
        R[f"앞뒤 {sh}시간"] = {
            "생산일": int(len(g)), "위험일": int(len(risk)),
            "예측 일최대 평균(기존→조정)": [float(g["예측 일최대(기존)"].mean()), float(g["예측 일최대(조정)"].mean())],
            "위험일 예측 일최대 평균(기존→조정)": [float(risk["예측 일최대(기존)"].mean()), float(risk["예측 일최대(조정)"].mean())],
            "위험일 저감률(%) 평균": float(((risk["예측 일최대(기존)"] - risk["예측 일최대(조정)"]) / risk["예측 일최대(기존)"]).mean() * 100),
            "위험일 저감 최대(%)": float(((risk["예측 일최대(기존)"] - risk["예측 일최대(조정)"]) / risk["예측 일최대(기존)"]).max() * 100),
            "예측 180 이상 시간(기존→조정)": [int(g["예측 180 이상 시간(기존)"].sum()), int(g["예측 180 이상 시간(조정)"].sum())],
            "예측 187 이상 날(기존→조정)": [int((g["예측 일최대(기존)"] >= HIGH_THR).sum()), int((g["예측 일최대(조정)"] >= HIGH_THR).sum())],
            "이동 생산량 비율 평균(%)": float(g["이동 생산량 비율"].mean() * 100),
            "교차확인(LightGBM 단일) 위험일 저감률(%) 평균": float(((risk["교차확인 일최대(기존)"] - risk["교차확인 일최대(조정)"]) / risk["교차확인 일최대(기존)"]).mean() * 100),
            "교차확인 저감된 위험일 비율(%)": float((risk["교차확인 일최대(조정)"] < risk["교차확인 일최대(기존)"]).mean() * 100)}
    (RES / "ext_schedule.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=1, default=float))

    # 그림: 위험일 중 PeakSense와 교차 확인 모델이 함께 가장 크게 낮아진 날(앞뒤 2시간)
    g = D[(D["이동 범위"] == 2) & (D["예측 일최대(기존)"] >= RISK)]
    both = np.minimum(g["예측 일최대(기존)"] - g["예측 일최대(조정)"], g["교차확인 일최대(기존)"] - g["교차확인 일최대(조정)"])
    best = g.loc[both.idxmax(), "date"]
    idx, plan0, p0, plan1, p1, act = keep[(pd.Timestamp(best), 2)]
    h = np.arange(24)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].bar(h - 0.2, plan0, 0.4, color="#c6ced3", label="기존 생산계획")
    ax[0].bar(h + 0.2, plan1, 0.4, color="#1d6b8c", label="조정 생산계획")
    ax[0].set_title(f"(가) 시간별 생산량 배분 ({best})", fontsize=11); ax[0].set_xlabel("시각"); ax[0].set_ylabel("생산량")
    ax[1].plot(h, p0, "-o", ms=3, color="#9aa5ab", label="예측(기존 계획)")
    ax[1].plot(h, p1, "-o", ms=3, color="#1d6b8c", label="예측(조정 계획)")
    ax[1].axhline(HIGH_THR, ls="--", color="#c0392b", lw=1); ax[1].text(0, HIGH_THR + 1, "고전력 기준 187", color="#c0392b", fontsize=8)
    ax[1].set_title(f"(나) 예측 시간 최대수요: 일 최대 {p0.max():.1f} → {p1.max():.1f}", fontsize=11); ax[1].set_xlabel("시각")
    for a_ in ax:
        a_.legend(frameon=False, fontsize=8); a_.spines[["top", "right"]].set_visible(False); a_.set_xticks(range(0, 24, 2))
    plt.tight_layout(); plt.savefig(FIG / "fig_ext_schedule.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
