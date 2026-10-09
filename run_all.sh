#!/usr/bin/env bash
# 전처리부터 보고서 표·그림 생성까지 순서대로 실행 (GRU는 선택: python 03_gru_optional.py)
set -e
python 01_model_comparison.py
python 02_robustness.py
python 04_peak_factors.py
python 05_train_final.py
python 06_predict.py --start "2021-09-08 00:00" --end "2021-09-14 23:45"
python 07_schedule_optimizer.py
python 08_model_extensions.py
python 09_tomorrow_preview.py --day 2021-07-19
echo "완료: outputs/results, outputs/figures, models 를 확인하세요."
