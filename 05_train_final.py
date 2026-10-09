"""전체 유효 자료로 최종 모델 학습과 저장

목적: 유효 전력 기록 전체와 26개 입력으로 PeakSense를 학습하고 다시 불러올 수 있게 저장합니다.
입력: 원본 CSV. 실행: python 05_train_final.py. 01 결과가 있으면 모델 설명에 기존 평가 지표도 기록합니다.
출력: models/peaksense.joblib 및 models/model_card.json.
해석: 전체 자료로 학습한 모델이므로 과거 테스트 구간의 성능 평가용으로 사용하지 않습니다.
상세: docs/FILE_GUIDE.md
"""
import json
import warnings

from src.config import HUBER, LGB, MODELS, RES, ROUTE_THR, RUN_THR
from src.features import build_features, final_features
from src.models import PeakSense
from src.preprocess import load_data

warnings.filterwarnings("ignore")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X = build_features(df)
    y = df["peak"].astype(float)
    valid = y.notna().to_numpy()
    F = final_features(X)
    model = PeakSense().fit(X.loc[valid, F], y[valid])
    model.save(MODELS / "peaksense.joblib")
    card = {"name": "PeakSense", "target": "시간 최대수요(한 시간의 네 15분 전력값 중 최댓값)",
            "data": f"okm_augumented_2021.csv, 제1장 전처리 후 {int(valid.sum()):,}시간",
            "structure": ["가동상태 판별기: LightGBM 분류, 라벨 = 시간 최대수요 > 70",
                          "가동 상태 회귀기 / 정지 상태 회귀기: LightGBM, Huber 손실",
                          f"가동 확률 ≥ {ROUTE_THR} → 가동 상태 회귀기"],
            "features": F, "lightgbm": {k: v for k, v in LGB.items() if k not in ("n_jobs", "verbose")},
            "huber": HUBER, "run_thr": RUN_THR}
    main_json = RES / "ch2_main.json"
    if main_json.exists():
        R = json.loads(main_json.read_text(encoding="utf-8"))
        card["test_metrics"] = R["comparison"]
    (MODELS / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("저장:", MODELS / "peaksense.joblib", "| 입력", len(F), "개")


if __name__ == "__main__":
    main()
