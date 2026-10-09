"""GRU와 기존 모델의 공통 표본 비교

목적: 하루 단위 GRU를 학습하고, 예측 가능한 같은 시간만 골라 기존 모델과 비교합니다.
입력: 원본 CSV, 01의 ch2_predictions.csv, 별도 PyTorch 설치. 실행: python 03_gru_optional.py
출력: outputs/results/ch2_gru.json. 제출 당시 공통 평가 표본은 1,728시간입니다.
해석: 기본 평가 1,759시간의 점수와 직접 섞지 않습니다. 선택 실행이며 run_all에는 포함되지 않습니다.
상세: docs/FILE_GUIDE.md
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
