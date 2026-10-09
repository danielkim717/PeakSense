# 전체 파일 안내

코드를 처음 읽는 팀원과 실행해 보려는 사용자를 위한 안내입니다. 실행 파일은 실험을 시작하는 입구이고, 공통 모듈은 그 안에서 호출되는 처리 과정입니다. 결과 JSON과 PNG는 이미 계산된 기록입니다.

## 실행 파일

| 파일 | 목적 | 입력과 실행 조건 | 대표 출력 |
|---|---|---|---|
| [01_model_comparison.py](../01_model_comparison.py) | 단순 기준·랜덤포레스트·단일 LightGBM·PeakSense를 비교하고, 입력과 모델 구조를 선택합니다. | 원본 CSV. 실행: python 01_model_comparison.py | outputs/results/ch2_main.json, ch2_predictions.csv 및 outputs/figures/fig16·17·19·20 그림. |
| [02_robustness.py](../02_robustness.py) | 주별·일별 성능 차이, 상태 기준 변경, 예측구간과 고전력 경보를 살펴봅니다. | 원본 CSV와 01_model_comparison.py의 ch2_predictions.csv. 실행: python 02_robustness.py | outputs/results/ch2_robustness.json 및 outputs/figures/fig18_weekly_mae.png. |
| [03_gru_optional.py](../03_gru_optional.py) | 하루 단위 GRU를 학습하고, 예측 가능한 같은 시간만 골라 기존 모델과 비교합니다. | 원본 CSV, 01의 ch2_predictions.csv, 별도 PyTorch 설치. 실행: python 03_gru_optional.py | outputs/results/ch2_gru.json. 제출 당시 공통 평가 표본은 1,728시간입니다. |
| [04_peak_factors.py](../04_peak_factors.py) | 시간대·생산·기온별 실제 피크 비율과 모델의 SHAP 기여값, 과소예측 조건을 분석합니다. | 원본 CSV. 실행: python 04_peak_factors.py. 필요한 모델은 이 파일에서 학습합니다. | outputs/results/ch3.json 및 outputs/figures/fig21~24 그림. |
| [05_train_final.py](../05_train_final.py) | 유효 전력 기록 전체와 26개 입력으로 PeakSense를 학습하고 다시 불러올 수 있게 저장합니다. | 원본 CSV. 실행: python 05_train_final.py. 01 결과가 있으면 모델 설명에 기존 평가 지표도 기록합니다. | models/peaksense.joblib 및 models/model_card.json. |
| [06_predict.py](../06_predict.py) | 시작·종료 시각과 계획을 받아 시간 최대수요, 15분 전력, 잔차 기반 예측구간을 계산합니다. | 원본 CSV와 데이터 이후 구간의 계획 CSV. 예: python 06_predict.py --start "2021-09-15 00:00" --end "2021-09-15 23:45" --plan plan_template.csv | outputs/forecast_<시작>_<끝>.csv 및 .png. |
| [07_schedule_optimizer.py](../07_schedule_optimizer.py) | 시간별 생산량을 재배분하고 모델이 예상하는 피크를 비교하는 국소 탐색 실험입니다. | 원본 CSV. 예: python 07_schedule_optimizer.py --day 2021-09-08 --shift 2 | outputs/results/ext_schedule.json, ext_schedule_days.csv 및 outputs/figures/fig_ext_schedule.png. |
| [08_model_extensions.py](../08_model_extensions.py) | 기온×시간·생산 결합 특성, 고전력 표본 가중치, 적응형 예측구간을 비교합니다. | 원본 CSV. 실행: python 08_model_extensions.py. 기본 비교를 이해하려면 01 결과를 먼저 확인합니다. | outputs/results/ext_model.json 및 outputs/figures/fig_ext_model.png. |
| [09_tomorrow_preview.py](../09_tomorrow_preview.py) | 15분 예측, 위험시간, 점검 우선순위, 생산일정 후보를 한 장의 그림으로 묶습니다. | 원본 CSV, 대상 날짜, 필요 시 미래 계획. 예: python 09_tomorrow_preview.py --day 2021-09-15 --plan plan_template.csv | outputs/preview_<날짜>.png 및 .csv. |

## 공통 모듈

| 파일 | 역할 | 반환값과 사용 범위 |
|---|---|---|
| [src/__init__.py](../src/__init__.py) | src를 가져올 수 있는 Python 패키지로 표시합니다. | 학습이나 파일 저장을 수행하지 않습니다. |
| [src/config.py](../src/config.py) | 공통 경로, 70·0.5·187 기준값, 평가 기간, LightGBM 설정과 그래프 글꼴을 모읍니다. | setup_plot()은 출력 폴더와 그래프 환경을 준비합니다. 기준값의 용도는 docs/MODEL.md 참고. |
| [src/preprocess.py](../src/preprocess.py) | 잘못된 시간과 공장인원 결측 행을 제외하고, 기상 결측을 보완하며 peak와 peak_q를 만듭니다. | 한 시간 간격 DataFrame을 반환합니다. 삭제한 65시간은 NaN으로 남으며 유효 전력은 6,103시간입니다. |
| [src/features.py](../src/features.py) | 생산계획 15개, 달력 5개, 기상 6개와 별도의 과거 전력 비교용 특성을 계산합니다. | 입력 DataFrame과 최종 26개 열 목록을 반환합니다. 미래 생산계획·기상예보를 미리 확보한다는 운영 가정이 필요합니다. |
| [src/models.py](../src/models.py) | 상태 분류기와 두 회귀기를 결합한 PeakSense, 단일 LightGBM, 랜덤포레스트, 선택 GRU를 정의합니다. | 시간별 전력 예측과 높은 부하 확률을 반환합니다. save()/load()는 모델 보관용이며 상태 라벨은 실제 설비 센서값이 아닙니다. |
| [src/evaluate.py](../src/evaluate.py) | 과거 자료로 학습하고 이후 기간을 평가하는 주별 분할과 MAE·RMSE·일 최대 MAE를 계산합니다. | 시각별 예측 DataFrame과 지표 dict를 반환합니다. 현재 주별 학습 경계는 index < issue이며 ISSUE_HOUR=11입니다. |
| [src/forecast15.py](../src/forecast15.py) | 시간별 전력 4개를 15분 시계열로 펼치고 구간 번호를 추가해 예측합니다. 최근 관측 오차로 당일 예측을 갱신합니다. | 예측 방식별 점수 표와 보정 계수를 반환합니다. 보정 설정은 7월 자료, 당일 갱신 평가는 8~9월 자료를 사용합니다. |
| [src/scheduler.py](../src/scheduler.py) | 계획을 바꿀 때 생산 흐름 입력을 다시 만들고 여러 후보를 예측해 더 낮은 목적함수의 후보를 찾습니다. | 조정 계획과 예상 전력을 반환합니다. 상위 3개 예측 전력과 인건비 항을 조합하며 알려진 제약·정규화 문제는 docs/LIMITATIONS.md에 기록했습니다. |

## 설정과 입력 양식

| 파일 | 역할 |
|---|---|
| [README.md](../README.md) | 저장소 첫 화면에 표시되는 프로젝트 소개·성과·시작 안내 |
| [requirements.txt](../requirements.txt) | pandas·LightGBM 등 설치 패키지 목록. GRU의 PyTorch는 선택 설치 |
| [plan_template.csv](../plan_template.csv) | 미래 생산계획·기상 예보 입력 예시. [7개 열 설명](PLAN_INPUT.md) |
| [run_all.bat](../run_all.bat) | Windows에서 기본·확장 실험을 순서대로 실행. 중간 실패 시 중단 |
| [run_all.sh](../run_all.sh) | macOS·Linux에서 같은 실험을 순서대로 실행. 한글 글꼴 설정 확인 필요 |
| [.gitignore](../.gitignore) | 원본 데이터·학습 모델·개인 환경·발표자료가 Git에 추가되지 않도록 제외 |
| [.gitattributes](../.gitattributes) | 운영체제별 줄바꿈과 이미지 같은 바이너리 파일의 처리 규칙 |
| [data/README.md](../data/README.md) | 공식 원본 CSV를 준비할 위치·형식·인코딩 안내 |
| [src/README.md](../src/README.md) | 공통 모듈의 호출 순서와 함수 안내 |
| [results/README.md](../results/README.md) | 기존 제출 결과와 새 실행 결과의 위치 구분 |

## 결과 파일

[results/competition/README.md](../results/competition/README.md)는 아래 JSON을 읽는 안내입니다. 원본 수치는 그대로 보존하며, 새 실험으로 덮어쓰지 않습니다.

| 파일 | 의미 | 생성하는 실행 파일 |
|---|---|---|
| [ch2_main.json](../results/competition/ch2_main.json) | comparison: 모델 비교, ablation: 입력 비교, features: 최종 입력 등. MAE 7.25의 근거 | [01_model_comparison.py](../01_model_comparison.py) |
| [ch2_robustness.json](../results/competition/ch2_robustness.json) | 주별 성능·통계 비교·라벨 민감도·예측구간·경보. 잔차 경계 재검증 대상 포함 | [02_robustness.py](../02_robustness.py) |
| [ch2_gru.json](../results/competition/ch2_gru.json) | GRU와 기존 모델을 같은 1,728시간에서 비교한 점수 | [03_gru_optional.py](../03_gru_optional.py) |
| [ch3.json](../results/competition/ch3.json) | 조건별 피크 발생률·SHAP·오류 집계. 관측 관계와 인과관계를 구분 | [04_peak_factors.py](../04_peak_factors.py) |
| [ext_schedule.json](../results/competition/ext_schedule.json) | 일정 후보 변경 전후의 모델 추정치. 현장 절감 실적이 아님 | [07_schedule_optimizer.py](../07_schedule_optimizer.py) |
| [ext_model.json](../results/competition/ext_model.json) | 추가 특성·피크 가중 학습·예측구간의 탐색 결과 | [08_model_extensions.py](../08_model_extensions.py) |

## 설명 문서

| 파일 | 설명 |
|---|---|
| [README.md](README.md) | 설명 문서의 목차와 읽는 순서 |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | 공모전의 문제 정의, 수행 과정, 최종 발표 내용과 코드의 범위 |
| [DATA.md](DATA.md) | 18개 원본 항목의 의미, 행 제외·결측 보완 근거, 6,103시간의 구성 |
| [MODEL.md](MODEL.md) | 26개 입력, 상태 라벨 70과 분석 기준 187의 차이, 모델·평가 구조 |
| [RESULTS.md](RESULTS.md) | 주 모델·입력·GRU·15분 갱신 성능과 비교 조건 |
| [ANALYSIS_AND_APPLICATION.md](ANALYSIS_AND_APPLICATION.md) | 피크 발생 조건, 오류분석, 현장 점검과 일정 조정 활용 |
| [REPRODUCING.md](REPRODUCING.md) | 설치, 데이터 준비, 실행 순서, 결과 저장 위치 |
| [PLAN_INPUT.md](PLAN_INPUT.md) | 미래 생산계획 CSV의 7개 열과 작성 예시 |
| [FILE_GUIDE.md](FILE_GUIDE.md) | 모든 코드·결과·설정·그림 파일의 역할과 연결 관계 |
| [LIMITATIONS.md](LIMITATIONS.md) | 입력 가용 시점, 잔차 경계, 일정 제약 등 재검증 과제 |
| [ATTRIBUTION.md](ATTRIBUTION.md) | 데이터·기반 기술 출처와 공개 범위 |
| [PUBLICATION_CHECKS.md](PUBLICATION_CHECKS.md) | 전처리·수치·코드·문서의 확인 내역 |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | 제공된 원본 코드 묶음의 파일별 SHA-256 기록 |

## 그래프

[assets/README.md](assets/README.md)는 그림의 의미와 생성 코드를 안내합니다.

| 파일 | 무엇을 보여 주나요? |
|---|---|
| [fig17_model_comparison.png](assets/fig17_model_comparison.png) | 같은 평가 구간에서 시간 최대수요와 일 최대수요의 MAE를 비교 |
| [fig18_weekly_mae.png](assets/fig18_weekly_mae.png) | 주별로 단일 LightGBM과 PeakSense의 MAE를 비교해 성능 변동 확인 |
| [fig20_15min_horizons.png](assets/fig20_15min_horizons.png) | 15분·30분·1시간·2시간 앞 예측에서 당일 갱신과 기준 모델을 비교 |

## 목적별 실행 흐름

- **기본 성능 비교:** 데이터 준비 → 01 → 02. GRU 비교가 필요할 때만 03을 추가합니다.
- **원인과 오류 확인:** 04에서 조건별 피크와 모델 기여를 살펴봅니다.
- **학습 모델 보관:** 05가 모델 파일을 만듭니다. 06·09는 이 모델을 읽지 않고 예측 시점에 맞춰 자체 학습합니다.
- **미래 구간 예측:** 데이터와 계획 CSV 준비 → 06. 한 장으로 살펴보려면 09를 실행합니다.
- **확장 실험:** 07은 일정 후보, 08은 추가 모델과 예측구간을 다룹니다. [검증 범위](LIMITATIONS.md)를 함께 확인합니다.

[첫 화면](../README.md) · [자세한 실행 명령](REPRODUCING.md)
