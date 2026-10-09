# 제출 당시 실험 결과

여섯 JSON 파일은 제출 당시 코드 묶음의 집계 결과를 보존합니다. 파일을 읽는 것만으로 재학습되지는 않습니다. 새 실행 결과는 `outputs/results/`에 별도로 생성됩니다.

| 결과 파일 | 생성하는 코드 | 무엇을 담고 있나요? |
|---|---|---|
| [ch2_main.json](ch2_main.json) | [01_model_comparison.py](../../01_model_comparison.py) | comparison: 모델 비교, ablation: 입력 비교, features: 최종 입력 등. MAE 7.25의 근거 |
| [ch2_robustness.json](ch2_robustness.json) | [02_robustness.py](../../02_robustness.py) | 주별 성능·통계 비교·라벨 민감도·예측구간·경보. 잔차 경계 재검증 대상 포함 |
| [ch2_gru.json](ch2_gru.json) | [03_gru_optional.py](../../03_gru_optional.py) | GRU와 기존 모델을 같은 1,728시간에서 비교한 점수 |
| [ch3.json](ch3.json) | [04_peak_factors.py](../../04_peak_factors.py) | 조건별 피크 발생률·SHAP·오류 집계. 관측 관계와 인과관계를 구분 |
| [ext_schedule.json](ext_schedule.json) | [07_schedule_optimizer.py](../../07_schedule_optimizer.py) | 일정 후보 변경 전후의 모델 추정치. 현장 절감 실적이 아님 |
| [ext_model.json](ext_model.json) | [08_model_extensions.py](../../08_model_extensions.py) | 추가 특성·피크 가중 학습·예측구간의 탐색 결과 |

숫자의 뜻을 표와 그래프로 읽으려면 [실험 결과 설명](../../docs/RESULTS.md)을 보세요. 주 평가 1,759시간과 GRU 공통 평가 1,728시간은 표본이 다릅니다.

예측구간·경보·일정 저감률에는 [재검증 과제](../../docs/LIMITATIONS.md)가 있습니다. 시간별 실측·예측 CSV와 원본 데이터는 공개하지 않습니다. 이 폴더의 집계 JSON은 문서 보완 과정에서도 값을 바꾸지 않았습니다.

[전체 파일 안내](../../docs/FILE_GUIDE.md) · [첫 화면](../../README.md)
