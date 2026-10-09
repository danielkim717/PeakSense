"""[4단계] 피크 발생 요건과 오류분석 → 보고서 제3장
  (1) 데이터 기반: 실제 고전력(시간 최대수요 ≥ 187) 시간의 조건별 발생률 — 표11, 그림21, 그림22
  (2) 모델 기반: 최종 PeakSense의 요인별 영향(SHAP) + 테스트 구간 경보의 적중·미탐지·오경보 — 그림23, 그림24, 표12
  (3) 비교 — 표13은 (1)(2)의 수치로 작성
  결과: outputs/results/ch3.json, outputs/figures/fig21~24   (약 1분)
"""
import json
import warnings
import numpy as np
import pandas as pd

from src.config import FIG, HIGH_THR, RES, setup_plot
from src.evaluate import weekly_folds
from src.features import build_features, final_features
from src.models import PeakSense
from src.preprocess import load_data

warnings.filterwarnings("ignore")
plt = setup_plot()
THR = HIGH_THR
ACC, GREY, PEAK, INK = "#1d6b8c", "#c6ced3", "#d9730d", "#13202a"

NAME = {"hour": "시각", "prod": "생산량", "prod_log": "생산량(로그)", "prod_roll3": "앞뒤 3시간 생산량 합", "기온": "기온",
        "cdd": "냉방 필요 온도", "hdd": "난방 필요 온도", "습도": "습도", "풍속": "풍속", "강수량": "강수량", "dow": "요일",
        "인건비": "주·야간 구분", "run_streak": "연속 생산시간", "idle_streak": "연속 비생산시간", "hours_to_stop": "생산 종료까지 남은 시간",
        "prod_day_total": "하루 총생산량", "run_day_hours": "하루 생산 시간 수", "prod_next1": "다음 1시간 생산량",
        "prod_next2": "다음 2시간 생산량", "prod_prev1": "직전 1시간 생산량", "prod_prev2": "직전 2시간 생산량",
        "startup": "생산 시작", "shutdown": "생산 종료", "run": "생산 여부", "sunday": "일요일", "saturday": "토요일"}
GROUP = {"생산계획": ["prod", "prod_log", "prod_roll3", "run_streak", "idle_streak", "hours_to_stop", "prod_day_total",
                     "run_day_hours", "prod_next1", "prod_next2", "prod_prev1", "prod_prev2", "startup", "shutdown", "run"],
         "시간·요일": ["hour", "dow", "인건비", "sunday", "saturday"], "기상": ["기온", "cdd", "hdd", "습도", "풍속", "강수량"]}


def hour_group(h):
    if h < 7: return "심야 00~06시"
    if h == 7: return "출근 전 07시"
    if h == 8: return "주간조 시작 08시"
    if h <= 11: return "오전 09~11시"
    if h == 12: return "점심 12시"
    if h <= 16: return "오후 13~16시"
    if h <= 19: return "저녁 17~19시"
    return "야간 20~23시"


def op_state(X):
    return pd.Series(np.select(
        [X["startup"] == 1, (X["run"] == 1) & (X["run_streak"] <= 3), (X["run"] == 1) & (X["hours_to_stop"] == 0),
         X["run"] == 1, (X["run"] == 0) & (X["idle_streak"] <= 2)],
        ["생산 시작 시간", "생산 시작 후 1~3시간", "생산 종료 직전", "연속 생산 중", "생산 종료 후 2시간 이내"], "비생산"), index=X.index)


def conditions(X, df):
    C = pd.DataFrame(index=X.index)
    C["시간대"] = [hour_group(h) for h in X.index.hour]
    C["요일"] = np.select([X["sunday"] == 1, X["saturday"] == 1], ["일요일", "토요일"], "평일")
    C["기온"] = pd.cut(df["기온"], [-30, 10, 20, 25, 28, 50], labels=["10℃ 미만", "10~20℃", "20~25℃", "25~28℃", "28℃ 이상"]).astype(str)
    pos = df.loc[(df["생산량"] > 0) & df["peak"].notna(), "생산량"]
    C["생산량"] = pd.cut(df["생산량"], [-1, 0, pos.quantile(1 / 3), pos.quantile(2 / 3), np.inf],
                       labels=["0", "낮음", "중간", "높음"]).astype(str)
    C["생산 상태"] = op_state(X)
    C["월"] = X.index.month.astype(str) + "월"
    return C


def rate_table(C, pk, col, order=None):
    g = pd.DataFrame({"c": C[col], "pk": pk})
    t = g.groupby("c").agg(n=("pk", "size"), k=("pk", "sum"))
    t["rate"] = t["k"] / t["n"] * 100
    t["lift"] = t["rate"] / (pk.mean() * 100)
    t["share"] = t["k"] / pk.sum() * 100
    if order:
        t = t.reindex([o for o in order if o in t.index])
    return t


def main():
    df = load_data()
    X = build_features(df)
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X)
    C = conditions(X, df)
    v = valid
    pk = (y[v] >= THR)
    Cv = C[v]
    R = {"n_hours": int(v.sum()), "n_peak": int(pk.sum()), "base_rate": float(pk.mean() * 100)}

    # ======================= (1) 데이터 기반
    orders = {"시간대": ["심야 00~06시", "출근 전 07시", "주간조 시작 08시", "오전 09~11시", "점심 12시", "오후 13~16시", "저녁 17~19시", "야간 20~23시"],
              "요일": ["평일", "토요일", "일요일"], "기온": ["10℃ 미만", "10~20℃", "20~25℃", "25~28℃", "28℃ 이상"],
              "생산량": ["0", "낮음", "중간", "높음"],
              "생산 상태": ["비생산", "생산 시작 시간", "생산 시작 후 1~3시간", "연속 생산 중", "생산 종료 직전", "생산 종료 후 2시간 이내"],
              "월": [f"{m}월" for m in range(1, 10)]}
    R["data"] = {k: rate_table(Cv, pk, k, o).round(2).reset_index().to_dict(orient="records") for k, o in orders.items()}

    # 08시 전력 상승: 07시와 08시 모두 생산 중인 평일
    run = X["run"] == 1
    jump = y.diff()
    cont = v & run & (X["prod_prev1"] > 0) & (X["sunday"] == 0) & (X["saturday"] == 0)
    R["jump08"] = {"08시": float(jump[cont & (X.index.hour == 8)].mean()),
                   "기타": float(jump[cont & ~X.index.hour.isin([8, 13])].mean()), "n08": int((cont & (X.index.hour == 8)).sum())}
    # 고전력 시간 안에서 최댓값이 나온 15분 구간
    qpos = df.loc[v, "peak_q"]
    R["quarter"] = {"피크 시간": (qpos[pk].value_counts(normalize=True).sort_index() * 100).round(1).tolist(),
                    "그 외 시간": (qpos[~pk].value_counts(normalize=True).sort_index() * 100).round(1).tolist()}
    # 시간대 × 기온 (생산 중인 시간)
    H = pd.DataFrame({"h": X.index.hour[v], "t": Cv["기온"], "pk": pk.astype(int), "run": X.loc[v, "run"]})
    H = H[H["run"] == 1]
    heat = H.pivot_table(index="h", columns="t", values="pk", aggfunc="mean") * 100
    cnt = H.pivot_table(index="h", columns="t", values="pk", aggfunc="size")
    heat = heat.where(cnt >= 8).reindex(columns=orders["기온"])
    # 08시 × 28℃ 이상 등 조합
    combo = H.assign(h8=H["h"].between(8, 11), hot=H["t"].isin(["25~28℃", "28℃ 이상"]))
    R["combo"] = {f"{'08~11시' if a else '그 외 시간'} · {'25℃ 이상' if b else '25℃ 미만'}": float(g["pk"].mean() * 100)
                  for (a, b), g in combo.groupby(["h8", "hot"])}

    # 그림 1: 시각별 고전력 발생률
    hr = pd.Series(pk.values, index=X.index[v].hour).groupby(level=0).mean() * 100
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(hr.index, hr.values, color=[ACC if h == 8 else GREY for h in hr.index], width=0.7)
    ax.axhline(R["base_rate"], color=INK, ls=":", lw=1)
    ax.text(23.4, R["base_rate"] + 0.6, f"전체 평균 {R['base_rate']:.1f}%", ha="right", fontsize=9)
    for h_, val in hr.items():
        if val >= 5:
            ax.text(h_, val + 0.6, f"{val:.0f}", ha="center", fontsize=8.5, color=ACC if h_ == 8 else INK)
    ax.set_xticks(range(24)); ax.set_xlabel("시각"); ax.set_ylabel("고전력 발생률(%)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig21_peak_rate_by_hour.png", dpi=160); plt.close()

    # 그림 2: 시간대 × 기온 히트맵 (생산 중, 06~20시)
    hh = heat.loc[6:20]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(hh.values, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(hh.values))
    ax.set_yticks(range(len(hh.index)), [f"{h:02d}시" for h in hh.index], fontsize=8.5)
    ax.set_xticks(range(len(hh.columns)), hh.columns, fontsize=9)
    for i in range(hh.shape[0]):
        for j in range(hh.shape[1]):
            val = hh.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color="white" if val > np.nanmax(hh.values) * 0.55 else INK)
    plt.colorbar(im, ax=ax, label="고전력 발생률(%)", fraction=0.04)
    plt.tight_layout(); plt.savefig(FIG / "fig22_hour_temp_heatmap.png", dpi=160); plt.close()

    # ======================= (2) 모델 기반 — 학습된 영향(SHAP, 최종 모델)
    final = PeakSense().fit(X.loc[v, F], y[v])
    Xv = X.loc[v, F]
    p_run, pred = final.predict_parts(Xv)
    contrib = final.contrib(Xv)  # 라우팅된 회귀기의 기여(전력 단위)
    S = pd.DataFrame(contrib, columns=F, index=Xv.index)
    run_rows = p_run >= 0.5
    peak_rows = (y[v] >= THR).to_numpy()
    imp_all = S[run_rows].abs().mean().sort_values(ascending=False)
    imp_peak = S[peak_rows].abs().mean().sort_values(ascending=False)
    R["shap_peak_top"] = [{"feature": NAME.get(k, k), "value": float(val)} for k, val in imp_peak.head(10).items()]
    R["shap_group_peak"] = {gname: float(S.loc[peak_rows, [c for c in cols if c in F]].abs().sum(axis=1).mean()) for gname, cols in GROUP.items()}
    R["shap_group_run"] = {gname: float(S.loc[run_rows, [c for c in cols if c in F]].abs().sum(axis=1).mean()) for gname, cols in GROUP.items()}
    # 고전력 시간에서 각 요인이 예측을 얼마나 '올렸는가'(부호 포함 평균)
    R["shap_signed_peak"] = {NAME.get(k, k): float(S.loc[peak_rows, k].mean()) for k in imp_peak.head(8).index}
    # 의존 관계: 가동 시간에서 기온·시각의 기여
    dep = pd.DataFrame({"기온": Xv["기온"], "s_t": S["기온"] + S["cdd"] + S["hdd"], "h": Xv.index.hour, "s_h": S["hour"],
                        "prod": Xv["prod"], "s_p": S[["prod", "prod_log", "prod_roll3"]].sum(axis=1)})[run_rows]
    tb = pd.cut(dep["기온"], [-30, 10, 20, 25, 28, 50], labels=orders["기온"])
    R["dep_temp"] = dep.groupby(tb, observed=True)["s_t"].mean().round(2).to_dict()
    R["dep_hour"] = dep.groupby("h")["s_h"].mean().round(2).to_dict()
    R["dep_prod"] = dep.groupby(pd.cut(dep["prod"], [-1, 0, 193, 917, 1e9], labels=["0", "낮음", "중간", "높음"]), observed=True)["s_p"].mean().round(2).to_dict()

    # 그림 3: 고전력 시간의 요인별 영향 크기(상위 10)
    top = imp_peak.head(10)[::-1]
    grp_of = {c: g for g, cols in GROUP.items() for c in cols}
    colmap = {"생산계획": ACC, "시간·요일": PEAK, "기상": "#5a9e6f"}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh([NAME.get(k, k) for k in top.index], top.values, color=[colmap[grp_of.get(k, "생산계획")] for k in top.index])
    for i, val in enumerate(top.values):
        ax.text(val + 0.15, i, f"{val:.1f}", va="center", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in colmap.items()], frameon=False, fontsize=9, loc="lower right")
    ax.set_xlabel("고전력 시간에서의 평균 영향 크기 |SHAP| (전력 단위)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig23_shap_peak_hours.png", dpi=160); plt.close()

    # 그림 4: 기온·시각이 예측을 올리는 정도(가동 시간)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    t = pd.Series(R["dep_temp"])
    axes[0].bar(range(len(t)), t.values, color=[ACC if val > 0 else GREY for val in t.values])
    axes[0].set_xticks(range(len(t)), t.index, fontsize=9); axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_title("기온 관련 요인의 기여", fontsize=11); axes[0].set_ylabel("예측 전력 변화")
    hdep = pd.Series(R["dep_hour"]).loc[6:20]
    axes[1].bar(hdep.index, hdep.values, color=[PEAK if h == 8 else (ACC if val > 0 else GREY) for h, val in hdep.items()])
    axes[1].set_xticks(hdep.index, [f"{h}" for h in hdep.index], fontsize=8.5); axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_title("시각의 기여 (06~20시)", fontsize=11); axes[1].set_xlabel("시각")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "fig24_shap_temp_hour.png", dpi=160); plt.close()

    # ======================= (2) 모델 기반 — 테스트 구간 사후 예측의 경보 성공·실패
    rows = []
    for t, te_end, issue in weekly_folds("2021-06-03"):
        tr = valid & (X.index < issue)
        te = valid & (X.index >= t) & (X.index <= te_end)
        m = PeakSense().fit(X.loc[tr, F], y[tr])
        pp, pr = m.predict_parts(X.loc[te, F])
        rows.append(pd.DataFrame({"y": y[te], "pred": pr, "p": pp}, index=X.index[te]))
    Q = pd.concat(rows)
    Q["res"] = Q["y"] - Q["pred"]
    Q["grp"] = (Q["p"] >= 0.5).astype(int).astype(str) + ((Q.index.hour >= 8) & (Q.index.hour <= 17)).astype(int).astype(str)
    Q["wk"] = Q.index.to_period("W-WED")
    pex = pd.Series(np.nan, index=Q.index)
    for (w, g), part in Q.groupby(["wk", "grp"]):
        pool = Q.loc[(Q["wk"] < w) & (Q["grp"] == g), "res"].to_numpy()
        if len(pool) < 30:
            pool = Q.loc[Q["wk"] < w, "res"].to_numpy()
        if len(pool) < 30:
            continue
        pex[part.index] = [float(((pv + pool) >= THR).mean()) for pv in part["pred"]]
    E2 = Q.assign(pex=pex)
    E2 = E2[(E2.index >= "2021-07-01") & E2["pex"].notna()].join(C)
    E2["actual"] = E2["y"] >= THR
    E2["alarm"] = E2["pex"] >= 0.3
    E2["kind"] = np.select([E2.actual & E2.alarm, E2.actual & ~E2.alarm, ~E2.actual & E2.alarm], ["적중", "미탐지", "오경보"], "정상")
    R["alarm_counts"] = E2["kind"].value_counts().to_dict()

    def prof(mask, col):
        return (E2.loc[mask, col].value_counts(normalize=True) * 100).round(1).to_dict()
    R["miss_profile"] = {col: prof(E2.kind == "미탐지", col) for col in ["시간대", "생산 상태", "기온", "요일"]}
    R["hit_profile"] = {col: prof(E2.kind == "적중", col) for col in ["시간대", "생산 상태", "기온"]}
    R["fa_profile"] = {col: prof(E2.kind == "오경보", col) for col in ["시간대", "생산 상태", "기온", "요일"]}
    pkE = E2[E2.actual]
    R["peak_bias"] = {"전체 고전력 시간 평균 오차(예측−실측)": float((pkE["pred"] - pkE["y"]).mean()),
                      "실측 200 이상": float((pkE.loc[pkE.y >= 200, "pred"] - pkE.loc[pkE.y >= 200, "y"]).mean()),
                      "n200": int((pkE.y >= 200).sum()),
                      "미탐지 평균 실측": float(E2.loc[E2.kind == "미탐지", "y"].mean()),
                      "미탐지 평균 예측": float(E2.loc[E2.kind == "미탐지", "pred"].mean()),
                      "오경보 평균 실측": float(E2.loc[E2.kind == "오경보", "y"].mean())}
    # 조건별 수치 예측 오차(테스트)
    E2["ae"] = (E2["y"] - E2["pred"]).abs()
    R["err_by"] = {col: E2.groupby(col)["ae"].mean().round(2).to_dict() for col in ["생산 상태", "시간대", "기온"]}
    R["err_overall"] = float(E2["ae"].mean())
    # 모델이 '예측값 ≥ 187'로 본 시간의 조건 분포 vs 실제 고전력 시간의 조건 분포(테스트 구간)
    R["pred_vs_actual_hours"] = {"예측 고전력": prof(E2.pred >= THR, "시간대"), "실제 고전력": prof(E2.actual, "시간대")}
    R["n_test_alarm_eval"] = int(len(E2))

    (RES / "ch3.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["n_peak", "base_rate", "jump08", "quarter", "combo", "shap_group_peak",
                                        "alarm_counts", "peak_bias", "err_overall"]}, ensure_ascii=False, indent=1, default=float))
    for k in ["시간대", "생산 상태", "기온", "생산량", "요일"]:
        print(k, [(r["c"], round(r["rate"], 1), round(r["lift"], 2), round(r["share"], 0)) for r in R["data"][k]])
    print("SHAP top", [(d["feature"], round(d["value"], 1)) for d in R["shap_peak_top"]])
    print("signed", {k: round(v, 1) for k, v in R["shap_signed_peak"].items()})
    print("dep_temp", R["dep_temp"], "dep_prod", R["dep_prod"])
    print("dep_hour", {h: R["dep_hour"][h] for h in range(6, 21)})
    print("미탐지", R["miss_profile"]); print("적중", R["hit_profile"]); print("오경보", R["fa_profile"])
    print("err_by", R["err_by"])
    print("pred_vs_actual", R["pred_vs_actual_hours"])


if __name__ == "__main__":
    main()
