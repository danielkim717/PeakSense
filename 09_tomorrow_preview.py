"""다음 날 피크와 생산계획을 한 화면으로 표시

목적: 15분 예측, 위험시간, 점검 우선순위, 생산일정 후보를 한 장의 그림으로 묶습니다.
입력: 원본 CSV, 대상 날짜, 필요 시 미래 계획. 예: python 09_tomorrow_preview.py --day 2021-09-15 --plan plan_template.csv
출력: outputs/preview_<날짜>.png 및 .csv.
해석: 06의 예측 함수와 src/scheduler.py를 재사용합니다. 실제 설비를 제어하는 앱이나 웹 서비스는 아닙니다.
상세: docs/FILE_GUIDE.md
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd

from src.config import HIGH_THR, ISSUE_HOUR, OUT, setup_plot
from src.features import build_features, final_features
from src.forecast15 import expand_hourly, to_15min
from src.scheduler import DaySchedule

warnings.filterwarnings("ignore")
plt = setup_plot()
P6 = importlib.import_module("06_predict")      # 지정 구간 예측 함수 재사용
RANK_COLOR = {1: "#c0392b", 2: "#e67e22", 3: "#9aa5ab", 0: "#e3e7ea"}


def priority(hour, temp, plan, pex, hot_day):
    """제4장 점검 우선순위: 1순위 08~11시·25℃ 이상·생산계획 있음 / 2순위 고전력 확률 30% 이상 또는 28℃ 이상 예보일 13~19시
    / 3순위 그 외 생산계획 있는 시간 / 0 생산계획 없음"""
    if 8 <= hour <= 11 and temp >= 25 and plan > 0:
        return 1
    if pex >= 0.3 or (hot_day and 13 <= hour <= 19):
        return 2
    return 3 if plan > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--shift", type=int, default=2)
    a = ap.parse_args()
    day = pd.Timestamp(a.day).normalize()

    df = P6.load_with_plan(a.plan)
    last_obs = df["peak"].last_valid_index()
    issue = min(day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR + 1), last_obs + pd.Timedelta(hours=1))
    X = build_features(df)
    F = final_features(X)
    y = df["peak"]
    clean = y.notna().to_numpy()

    gh = P6.group_key(X["run"], X.index.hour)
    mh, rh, grh = P6.fit_with_residuals(X, y, clean, issue, F, gh)
    th = X.index[X.index.normalize() == day]
    ph = mh.predict(X.loc[th, F])
    loh, hih = P6.interval(ph, gh[th].to_numpy(), rh.to_numpy(), grh.to_numpy())
    pex = P6.exceed_prob(ph, gh[th].to_numpy(), rh.to_numpy(), grh.to_numpy(), HIGH_THR)

    v = to_15min(df)
    E = expand_hourly(X, F)
    g15 = P6.group_key(E["run"], E.index.hour)
    m15, r15, gr15 = P6.fit_with_residuals(E, v, v.notna().to_numpy(), issue, F + ["quarter"], g15)
    t15 = E.index[E.index.normalize() == day]
    p15 = m15.predict(E.loc[t15, F + ["quarter"]])
    lo15, hi15 = P6.interval(p15, g15[t15].to_numpy(), r15.to_numpy(), gr15.to_numpy())

    temp = df.loc[th, "기온"].to_numpy()
    plan0 = df.loc[th, "생산량"].fillna(0).to_numpy(float)
    hot_day = temp.max() >= 28
    rank = np.array([priority(h, t, p, e, hot_day) for h, t, p, e in zip(th.hour, temp, plan0, pex)])

    prod = df["생산량"].fillna(0)
    tr = clean & (X.index < issue)
    cap = float(prod[tr & (prod > 0).to_numpy()].quantile(0.95))
    S = DaySchedule(mh, X, prod, F, day, cap)
    plan1, ph1 = S.optimize(plan0, max_shift=a.shift) if plan0.sum() > 0 else (plan0, ph)

    out = pd.DataFrame({"시각": th.hour, "생산계획": plan0, "권고 생산량": plan1.round(0), "기온 예보": temp,
                        "시간 최대수요 예측": ph.round(1), "90% 상단": hih.round(1), "고전력 확률": pex.round(2),
                        "점검 우선순위": rank, "권고 반영 예측": ph1.round(1)})
    if y.loc[th].notna().any():
        out["실측"] = y.loc[th].to_numpy()
    tag = f"{day:%Y%m%d}"
    out.to_csv(OUT / f"preview_{tag}.csv", index=False, encoding="utf-8-sig")

    # ---- 그림
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1.2, 1.8], hspace=0.55, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    hrs15 = t15.hour + t15.minute / 60
    ax0.fill_between(hrs15, lo15, hi15, step="post", color="#1d6b8c", alpha=0.15, label="90% 예측구간")
    ax0.step(hrs15, p15, where="post", color="#1d6b8c", lw=1.6, label="15분 예측")
    if "실측" in out:
        ax0.step(hrs15, v.reindex(t15).to_numpy(), where="post", color="#13202a", lw=0.9, alpha=0.7, label="15분 실측(사후 확인용)")
    ax0.axhline(HIGH_THR, ls="--", color="#c0392b", lw=1)
    ax0.text(0.1, HIGH_THR + 2, "고전력 기준 187", color="#c0392b", fontsize=8)
    for h, r in zip(th.hour, rank):
        if r in (1, 2):
            ax0.axvspan(h, h + 1, color=RANK_COLOR[r], alpha=0.08, lw=0)
    ax0.set_xlim(0, 24); ax0.set_xticks(range(0, 25, 2))
    ax0.set_title(f"(가) {day:%Y-%m-%d}({'월화수목금토일'[day.dayofweek]}) 15분 예측 · 예측 발행 {issue - pd.Timedelta(hours=1):%m-%d %H시} 관측까지 사용"
                  f" · 최고기온 예보 {temp.max():.1f}℃", fontsize=11, loc="left")
    ax0.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax0.set_ylabel("전력")

    ax1 = fig.add_subplot(gs[1, :])
    ax1.bar(th.hour + 0.5, pex * 100, 0.85, color=[RANK_COLOR[r] for r in rank])
    ax1.axhline(30, ls=":", color="#555", lw=1)
    ax1.set_xlim(0, 24); ax1.set_xticks(range(0, 25, 2)); ax1.set_ylim(0, 100)
    ax1.set_ylabel("고전력 확률(%)")
    ax1.set_title("(나) 시간별 고전력 확률과 점검 우선순위(제4장 기준)", fontsize=11, loc="left")
    for i, (lab, c) in enumerate((("1순위", 1), ("2순위", 2), ("3순위", 3))):
        ax1.text(0.79 + i * 0.07, 1.08, "■ " + lab, color=RANK_COLOR[c], transform=ax1.transAxes, fontsize=9)

    ax2 = fig.add_subplot(gs[2, :2])
    h = np.arange(24)
    ax2.bar(h - 0.2, plan0, 0.4, color="#c6ced3", label="기존 생산계획")
    ax2.bar(h + 0.2, plan1, 0.4, color="#1d6b8c", label="권고 생산계획")
    ax2b = ax2.twinx()
    ax2b.plot(h, ph, color="#9aa5ab", lw=1.2, label="예측(기존)")
    ax2b.plot(h, ph1, color="#c0392b", lw=1.4, label="예측(권고)")
    ax2b.set_ylabel("시간 최대수요 예측")
    ax2.set_xticks(range(0, 24, 2)); ax2.set_ylabel("생산량")
    ax2.set_title(f"(다) 생산일정 권고(하루 생산량 유지, 앞뒤 {a.shift}시간 안 조정): 예측 일 최대 {ph.max():.1f} → {ph1.max():.1f}",
                  fontsize=11, loc="left")
    l1, n1 = ax2.get_legend_handles_labels(); l2, n2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, n1 + n2, frameon=False, fontsize=8, ncol=2, loc="upper left")

    ax3 = fig.add_subplot(gs[2, 2]); ax3.axis("off")
    r1 = [f"{x}시" for x in th.hour[rank == 1]]
    r2 = [f"{x}시" for x in th.hour[rank == 2]]
    moved = np.abs(plan1 - plan0).sum() / 2 / max(plan0.sum(), 1) * 100
    lines = ["(라) 점검 목록",
             f"· 1순위 {len(r1)}시간: " + (", ".join(r1[:6]) if r1 else "없음"),
             f"· 2순위 {len(r2)}시간: " + (", ".join(r2[:6]) + (" 등" if len(r2) > 6 else "") if r2 else "없음"),
             f"· 예측 최대 {ph.max():.1f} ({th.hour[int(ph.argmax())]}시)",
             f"  90% 상단 {hih[int(ph.argmax())]:.1f}",
             f"· 권고: 생산량 {moved:.1f}% 이동",
             "· 08시 주간조 시작 시 순차 가동",
             "· 당일 15분 실측이 예측구간 상단을",
             "  넘으면 15분 보정 예측으로 갱신"]
    for i, s in enumerate(lines):
        ax3.text(0, 1 - i * 0.115, s, fontsize=10 if i else 11, fontweight="bold" if i == 0 else None, va="top",
                 transform=ax3.transAxes)
    png = OUT / f"preview_{tag}.png"
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(out.to_string(index=False))
    print("저장:", png)


if __name__ == "__main__":
    main()
