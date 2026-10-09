"""[3단계, 선택] 딥러닝 GRU를 같은 조건으로 평가 → 보고서 제2장 표4의 ⑥ 행
  PyTorch 필요(pip install torch). CPU 기준 약 30분.  결과: outputs/results/ch2_gru.json
"""
import json
import time
import warnings

import pandas as pd

from src.config import RES, setup_plot
from src.evaluate import score, walk_forward
from src.features import MORNING, build_features, final_features
from src.models import GRUModel
from src.preprocess import load_data

warnings.filterwarnings("ignore")
setup_plot()


def main():
    t0 = time.time()
    df = load_data()
    X = build_features(df)
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X) + MORNING
    G = walk_forward(X, y, valid, {"gru": (GRUModel, F)})
    ok = G["gru"].notna()                       # 일부 시간이 삭제된 날(8/28·29)은 하루 단위 입력이 안 되어 제외
    P = pd.read_csv(RES / "ch2_predictions.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")
    J = P.loc[G.index[ok]]
    R = {"gru": score(G.loc[ok, "y"], G.loc[ok, "gru"]), "n": int(ok.sum()),
         "same_hours": {k: score(J["y"], J[k]) for k in ["rf", "lgbm", "peaksense"]},
         "high_mae": float((G.loc[ok & (G.y >= 187), "gru"] - G.loc[ok & (G.y >= 187), "y"]).abs().mean()),
         "elapsed_s": time.time() - t0}
    (RES / "ch2_gru.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print({k: round(v, 2) for k, v in R["gru"].items()}, "평가", R["n"], "시간")


if __name__ == "__main__":
    main()
