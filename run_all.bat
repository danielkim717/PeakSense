@echo off
rem 입력: data의 원본 CSV. 출력: outputs와 models. 실행 안내: docs/REPRODUCING.md
rem 01,02,04,05,06,07,08,09 순서이며 선택 GRU(03)는 별도 실행합니다.
chcp 65001 > nul
rem 전처리부터 보고서 표·그림 생성까지 순서대로 실행 (GRU는 선택: 03_gru_optional.py)
python 01_model_comparison.py || exit /b 1
python 02_robustness.py || exit /b 1
python 04_peak_factors.py || exit /b 1
python 05_train_final.py || exit /b 1
python 06_predict.py --start "2021-09-08 00:00" --end "2021-09-14 23:45" || exit /b 1
python 07_schedule_optimizer.py || exit /b 1
python 08_model_extensions.py || exit /b 1
python 09_tomorrow_preview.py --day 2021-07-19 || exit /b 1
echo 완료: outputs\results, outputs\figures, models 를 확인하세요.
