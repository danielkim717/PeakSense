# 결과 그래프 안내

제출 당시 결과를 설명하는 그림입니다. 수치는 [실험 결과](../RESULTS.md), 생성 방법은 [실행 안내](../REPRODUCING.md)에서 확인할 수 있습니다. MAE는 원자료의 전력값 척도입니다.

| 그림 | 나타내는 내용 | 생성 코드 |
|---|---|---|
| [fig17_model_comparison.png](fig17_model_comparison.png) | 같은 평가 구간에서 시간 최대수요와 일 최대수요의 MAE를 비교 | [01_model_comparison.py](../../01_model_comparison.py) |
| [fig18_weekly_mae.png](fig18_weekly_mae.png) | 주별로 단일 LightGBM과 PeakSense의 MAE를 비교해 성능 변동 확인 | [02_robustness.py](../../02_robustness.py) |
| [fig20_15min_horizons.png](fig20_15min_horizons.png) | 15분·30분·1시간·2시간 앞 예측에서 당일 갱신과 기준 모델을 비교 | [01_model_comparison.py](../../01_model_comparison.py) |

[전체 파일 안내](../FILE_GUIDE.md)
