# 실행과 재현

## 환경 준비

Python 3.10 이상과 `requirements.txt`의 패키지를 사용합니다. 버전은 하한만 지정되어 있으므로 운영체제·패키지 버전에 따라 결과나 실행 시간이 달라질 수 있습니다. 제출 당시 환경을 완전히 고정한 lock 파일은 없습니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

원본 `okm_augumented_2021.csv`를 `data/`에 배치합니다. 이미 전처리한 CSV는 사용하지 않습니다. 기본 한글 그래프 글꼴은 `Malgun Gothic`입니다. macOS/Linux에서는 `src/config.py`의 글꼴 설정을 설치된 한글 글꼴로 바꿔야 할 수 있습니다.

## 단계별 실행

저장소 루트에서 실행합니다.

| 명령 | 수행 내용 | 선행 조건 |
|---|---|---|
| `python 01_model_comparison.py` | 기준 모델·입력·구조·15분 예측 비교 | 원본 데이터 |
| `python 02_robustness.py` | 주별 성능·민감도·예측구간·경보 분석 | 01 결과 |
| `python 03_gru_optional.py` | GRU 공통 표본 비교(선택) | 01 결과, PyTorch |
| `python 04_peak_factors.py` | 피크 조건·오류·SHAP 분석 | 01 결과 |
| `python 05_train_final.py` | 전체 유효 자료로 모델 학습·저장 | 원본 데이터 |
| `python 07_schedule_optimizer.py` | 일정 후보 탐색 실험 | 원본 데이터 |
| `python 08_model_extensions.py` | 개선모델·예측구간 후속 실험 | 01 결과 |
| `python 09_tomorrow_preview.py --day 2021-07-19` | 피크 미리보기 | 원본 데이터 |

GRU는 `python -m pip install torch`로 PyTorch를 추가 설치합니다. 일괄 실행은 Windows의 `run_all.bat` 또는 `bash run_all.sh`를 사용합니다. 일괄 실행에는 [검증 과제](LIMITATIONS.md)가 남은 확장 실험도 포함되며 GRU는 포함되지 않습니다.

## 지정 구간 예측

자료 안에 있는 기간을 예측하는 예시:

```bash
python 06_predict.py --start "2021-09-08 00:00" --end "2021-09-14 23:45"
```

자료 이후 기간은 계획 CSV를 제공합니다.

```bash
python 06_predict.py --start "2021-09-15 00:00" --end "2021-09-15 23:45" --plan plan_template.csv
```

필수 열은 `일시, 생산량, 기온, 풍속, 습도, 강수량, 인건비`이며 한 행은 한 시간입니다. 양식의 예시 값은 실제 계획·예보가 아닙니다. 예측 구간의 계획과 앞뒤 맥락을 가능한 한 완전하게 제공해야 합니다. 현재 구현은 기존 데이터와 겹치는 계획 행을 제거하므로 과거 생산계획을 덮어쓰는 용도로 `--plan`을 사용할 수 없습니다.

`06_predict.py`는 해당 예측 시점 기준으로 모델을 학습합니다. `05_train_final.py`에서 저장한 전체 자료 학습 모델로 과거 기간을 평가하면 미래 자료가 포함되므로 올바른 테스트가 아닙니다.

## 출력 위치와 버전 보존

- `outputs/results/`: 새 실행의 지표 JSON·예측 CSV
- `outputs/figures/`: 새 실행의 분석 그림
- `outputs/forecast_*.csv/png`: 지정 구간 예측
- `models/`: 학습된 모델과 모델 설명 JSON
- `results/competition/`: 원본 코드 묶음에 있던 제출 당시 집계 결과(새 실행으로 덮어쓰지 않음)

새로 생성한 자료는 `.gitignore`로 제외합니다. 제출 당시 원본 CSV, 시간별 실측·예측 CSV, 학습 모델, 내부 검토 메모는 저장소에 포함하지 않았습니다.

## 현재 확인한 범위

코드 출처는 사용자가 최종 버전으로 제공한 `PeakSense_code.zip`입니다. 원래의 9개 실행 스크립트, `src/`, 실행 설정을 보존했으며 알고리즘 수정이나 재학습은 수행하지 않았습니다. 공개 준비 과정에서 문서·폴더 구성과 `.gitignore`를 추가했습니다. 최종 발표자료의 통합 엔진은 현재 제공 코드에 대응 구현이 없어 재현 범위에 포함하지 않습니다.

Python 구문 검사, 원본 자료로 전처리·입력 생성 결과 확인, 저장 예측값에서 주요 MAE 재계산, 공개 대상 파일 점검을 수행합니다. 실제 확인 결과는 [`PUBLICATION_CHECKS.md`](PUBLICATION_CHECKS.md)에 기록합니다. 출처별 해시는 [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json)에 있습니다.
