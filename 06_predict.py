"""사용자가 지정한 미래 구간의 전력 예측

목적: 시작·종료 시각과 계획을 받아 시간 최대수요, 15분 전력, 잔차 기반 예측구간을 계산합니다.
입력: 원본 CSV와 데이터 이후 구간의 계획 CSV. 예: python 06_predict.py --start "2021-09-15 00:00" --end "2021-09-15 23:45" --plan plan_template.csv
출력: outputs/forecast_<시작>_<끝>.csv 및 .png.
해석: 05의 저장 모델을 읽지 않고 예측 시점 이전 자료로 자체 학습합니다. 입력 양식은 docs/PLAN_INPUT.md 참고. 전력 합을 kWh로 단정하지 않습니다.
상세: docs/FILE_GUIDE.md
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from src.config import HIGH_THR, ISSUE_HOUR, OUT, QCOLS, setup_plot
from src.features import build_features, final_features
from src.forecast15 import expand_hourly, to_15min
from src.models import PeakSense
from src.preprocess import load_data

warnings.filterwarnings("ignore")
plt = setup_plot()
final_fit = lambda X, y: PeakSense().fit(X, y)  # noqa: E731
PLAN_COLS = ["생산량", "기온", "풍속", "습도", "강수량", "인건비"]
CAL_DAYS = 28  # 예측구간 보정용 잔차를 모으는 기간


def load_with_plan(plan_path: str | None):
    df = load_data()
    if plan_path:
        P = pd.read_csv(plan_path, encoding="utf-8-sig")
        P["ts"] = pd.to_datetime(P["일시"])
        P = P.set_index("ts").sort_index()
        missing = [c for c in PLAN_COLS if c not in P]
        if missing:
            raise SystemExit(f"계획 파일에 컬럼이 없습니다: {missing}")
        P = P[~P.index.isin(df.index)]
        fut = pd.DataFrame(index=P.index)
        for c in PLAN_COLS:
            fut[c] = P[c].astype(float)
        for c in QCOLS + ["peak"]:
            fut[c] = np.nan
        df = pd.concat([df, fut]).sort_index()
    return df


def group_key(run, hours):
    return run.astype(int).astype(str) + "_" + ((hours >= 8) & (hours <= 17)).astype(int).astype(str)


def fit_with_residuals(Xf, y, clean, issue, cols, grp):
    """보정 잔차(최근 CAL_DAYS일, 그 이전 데이터로 학습한 모델의 사후 오차) + 최종 모델"""
    cal_start = issue - pd.Timedelta(days=CAL_DAYS)
    tr_a = clean & (Xf.index < cal_start)
    cal = clean & (Xf.index >= cal_start) & (Xf.index < issue)
    ma = final_fit(Xf.loc[tr_a, cols], y[tr_a])
    res = pd.Series(y[cal].to_numpy() - ma.predict(Xf.loc[cal, cols]), index=Xf.index[cal])
    final = final_fit(Xf.loc[clean & (Xf.index < issue), cols], y[clean & (Xf.index < issue)])
    return final, res, grp[cal]


def interval(pred, g_pred, res, g_res, qs=(0.05, 0.95)):
    lo, hi = np.empty(len(pred)), np.empty(len(pred))
    for k in np.unique(g_pred):
        pool = res[g_res == k]
        if len(pool) < 30:
            pool = res
        q = np.quantile(pool, qs)
        m = g_pred == k
        lo[m], hi[m] = pred[m] + q[0], pred[m] + q[1]
    return lo, hi


def exceed_prob(pred, g_pred, res, g_res, thr):
    out = np.empty(len(pred))
    for i, (p, k) in enumerate(zip(pred, g_pred)):
        pool = res[g_res == k]
        if len(pool) < 30:
            pool = res
        out[i] = float(((p + pool) >= thr).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description="지정 구간 전력사용량 예측")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--plan", help="데이터 이후 구간을 예측할 때 필요한 생산계획·기상 CSV")
    a = ap.parse_args()
    start = pd.Timestamp(a.start).floor("15min")
    end = pd.Timestamp(a.end).floor("15min")
    if end < start:
        raise SystemExit("--end 가 --start 보다 빠릅니다.")

    df = load_with_plan(a.plan)
    last_obs = df["peak"].last_valid_index()
    if end.floor("h") > df.index.max():
        raise SystemExit(f"{df.index.max():%Y-%m-%d %H:%M} 이후 구간은 --plan 파일로 생산계획·기상을 넣어야 합니다.")

    # 발행시점: 시작일 전날 11시 관측까지 (데이터 이후라면 마지막 관측까지 전부 사용)
    issue = min(start.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR + 1),
                last_obs + pd.Timedelta(hours=1))
    X = build_features(df)          # 최종 입력은 생산계획·달력·기상뿐이라 발행시점 이후 실측을 쓰지 않음
    FEATS = final_features(X)
    yh = df["peak"]
    ch = yh.notna().to_numpy()      # 제1장에서 삭제한 시간 제외
    thr = HIGH_THR

    # ---- 시간 최대수요 모델
    gh = group_key(X["run"], X.index.hour)
    mh, rh, grh = fit_with_residuals(X, yh, ch, issue, FEATS, gh)

    # ---- 15분 모델
    v = to_15min(df)
    E = expand_hourly(X, FEATS)
    c15 = v.notna().to_numpy()
    g15 = group_key(E["run"], E.index.hour)
    cols15 = FEATS + ["quarter"]
    m15, r15, gr15 = fit_with_residuals(E, v, c15, issue, cols15, g15)

    # ---- 예측
    t15 = E.index[(E.index >= start) & (E.index <= end)]
    th = X.index[(X.index >= start.floor("h")) & (X.index <= end.floor("h"))]
    p15 = m15.predict(E.loc[t15, cols15])
    lo15, hi15 = interval(p15, g15[t15].to_numpy(), r15.to_numpy(), gr15.to_numpy())
    ph = mh.predict(X.loc[th, FEATS])
    loh, hih = interval(ph, gh[th].to_numpy(), rh.to_numpy(), grh.to_numpy())
    pex = exceed_prob(ph, gh[th].to_numpy(), rh.to_numpy(), grh.to_numpy(), thr)

    H = pd.DataFrame({"시간최대_예측": ph, "시간최대_하한90": loh, "시간최대_상한90": hih, "피크초과확률": pex}, index=th)
    out = pd.DataFrame({"15분_예측": p15, "15분_하한90": lo15, "15분_상한90": hi15}, index=t15)
    out = out.join(H.reindex(out.index.floor("h")).set_axis(out.index))
    out["생산계획"] = df["생산량"].reindex(out.index.floor("h")).to_numpy()
    out["기온"] = df["기온"].reindex(out.index.floor("h")).to_numpy()
    actual = v.reindex(t15)
    has_actual = actual.notna().sum() > 0
    if has_actual:
        out["15분_실측"] = actual
        out["시간최대_실측"] = yh.reindex(out.index.floor("h")).to_numpy()
    out.index.name = "일시"

    tag = f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
    csv = OUT / f"forecast_{tag}.csv"
    out.round(3).to_csv(csv, encoding="utf-8-sig")

    # ---- 요약
    kwh = float(p15.sum() * 0.25)
    imax = int(np.argmax(ph))
    risk = H[H["피크초과확률"] >= 0.3]
    print("=" * 64)
    print(f"예측 구간 : {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}  ({len(t15)}개 15분 구간)")
    print(f"학습 데이터: {issue - pd.Timedelta(hours=1):%Y-%m-%d %H:%M} 까지 관측 · 제1장 전처리 기준(삭제된 65시간 제외)"
          + (" · 미래 구간(계획 파일 사용)" if a.plan and start > last_obs else ""))
    print(f"예상 사용량: {kwh:,.0f} kWh  (15분 수요 × 0.25h 합, 단위 kW 가정)")
    print(f"예상 최대수요: {ph[imax]:.1f} ({th[imax]:%m-%d %H시}), 90% 구간 {loh[imax]:.1f} ~ {hih[imax]:.1f}")
    print(f"피크 기준 {thr:.0f} 초과확률 30% 이상: {len(risk)}시간"
          + (" → " + ", ".join(f"{t:%m-%d %H시}({p:.0%})" for t, p in risk['피크초과확률'].head(12).items()) if len(risk) else ""))
    if has_actual:
        m = actual.notna().to_numpy()
        a15, mh_ = actual.to_numpy()[m], p15[m]
        yhh = yh.reindex(th)
        mm = yhh.notna().to_numpy()
        cov = np.mean((a15 >= lo15[m]) & (a15 <= hi15[m]))
        print("-" * 64)
        nz = a15 > 0
        print(f"[실측 비교] 15분  MAE {np.mean(np.abs(a15 - mh_)):.2f} · MAPE {np.mean(np.abs(a15[nz] - mh_[nz]) / a15[nz]) * 100:.1f}% · 90%구간 적중 {cov:.0%}")
        print(f"[실측 비교] 시간최대 MAE {np.mean(np.abs(yhh[mm] - ph[mm])):.2f} · 실제 최대 {yhh.max():.0f} ({yhh.idxmax():%m-%d %H시})")
        print(f"[실측 비교] 사용량 실측 {a15.sum() * 0.25:,.0f} kWh vs 예측 {mh_.sum() * 0.25:,.0f} kWh "
              f"(오차 {(mh_.sum() / a15.sum() - 1) * 100:+.1f}%)")
    print(f"저장: {csv}")

    # ---- 그래프
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].fill_between(t15, lo15, hi15, color="#1d6b8c", alpha=0.15, step="post", label="90% 예측구간")
    ax[0].step(t15, p15, where="post", color="#1d6b8c", lw=1.4, label="15분 예측")
    if has_actual:
        ax[0].step(t15, actual, where="post", color="#13202a", lw=1, label="15분 실측")
    ax[0].axhline(thr, color="#d9730d", ls="--", lw=1, label=f"피크 기준 {thr:.0f}")
    ax[0].set_ylabel("15분 수요전력")
    ax[0].legend(ncol=4, loc="upper left", fontsize=9)
    ax[0].set_title(f"지정 구간 전력 예측 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}")
    ax[1].bar(th, pex * 100, width=1 / 24, align="edge",
              color=["#d9730d" if p >= 0.3 else "#9aa6ae" for p in pex])
    ax[1].set_ylabel("피크 초과확률 %")
    ax[1].set_ylim(0, 100)
    plt.tight_layout()
    png = csv.with_suffix(".png")
    plt.savefig(png, dpi=120)
    print(f"저장: {png}")


if __name__ == "__main__":
    main()
