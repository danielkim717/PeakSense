# 공통 코드 안내

번호가 붙은 실행 파일이 이 폴더의 함수를 불러옵니다. 이 폴더의 파일을 직접 실행하면 실험 결과가 생성되는 구조는 아닙니다.

| 파일 | 역할 | 주요 함수 또는 클래스 |
|---|---|---|
| [__init__.py](__init__.py) | src를 가져올 수 있는 Python 패키지로 표시합니다. | 없음. 다른 스크립트의 import src에서 사용합니다. |
| [config.py](config.py) | 공통 경로, 70·0.5·187 기준값, 평가 기간, LightGBM 설정과 그래프 글꼴을 모읍니다. | 각 실행 스크립트에서 상수를 가져옵니다. |
| [preprocess.py](preprocess.py) | 잘못된 시간과 공장인원 결측 행을 제외하고, 기상 결측을 보완하며 peak와 peak_q를 만듭니다. | data/의 원본 CSV를 우선 검색합니다. 대표 함수: load_data(). |
| [features.py](features.py) | 생산계획 15개, 달력 5개, 기상 6개와 별도의 과거 전력 비교용 특성을 계산합니다. | 전처리된 시간별 DataFrame. 대표 함수: plan_features(), build_features(), final_features(). |
| [models.py](models.py) | 상태 분류기와 두 회귀기를 결합한 PeakSense, 단일 LightGBM, 랜덤포레스트, 선택 GRU를 정의합니다. | 입력 X와 학습 정답 y. fit(), predict(), predict_parts()로 사용합니다. |
| [evaluate.py](evaluate.py) | 과거 자료로 학습하고 이후 기간을 평가하는 주별 분할과 MAE·RMSE·일 최대 MAE를 계산합니다. | X, y, 유효 행 마스크, 비교할 모델 목록. weekly_folds(), walk_forward(), score(). |
| [forecast15.py](forecast15.py) | 시간별 전력 4개를 15분 시계열로 펼치고 구간 번호를 추가해 예측합니다. 최근 관측 오차로 당일 예측을 갱신합니다. | 전처리 자료와 모델 입력. to_15min(), expand_hourly(), run(). |
| [scheduler.py](scheduler.py) | 계획을 바꿀 때 생산 흐름 입력을 다시 만들고 여러 후보를 예측해 더 낮은 목적함수의 후보를 찾습니다. | 학습 모델, 생산계획, 대상 날짜, 생산량 상한. DaySchedule.predict_many(), optimize(). |

## 자료가 흐르는 순서

`preprocess.load_data()` → `features.build_features()` → `features.final_features()` → `models.PeakSense.fit()` / `predict()` 순서로 원본 자료가 예측으로 이어집니다. 모델 비교는 `evaluate.walk_forward()`가 이 과정을 시간 순서에 맞게 반복합니다.

`forecast15.py`는 15분 해상도의 별도 예측·보정을, `scheduler.py`는 생산계획 후보의 재계산·비교를 담당합니다. 모든 공통 설정은 `config.py`에 있습니다.

[모델의 입력과 라벨](../docs/MODEL.md) · [전체 파일 안내](../docs/FILE_GUIDE.md) · [첫 화면](../README.md)
