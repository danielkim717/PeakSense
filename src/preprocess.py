"""데이터 불러오기와 전처리 (보고서 제1장의 전처리와 동일)"""
from pathlib import Path

import pandas as pd

from .config import DATA_NAME, QCOLS, ROOT


def find_data() -> Path:
    """데이터 파일 위치: data/ 폴더 → 이 폴더 → 상위 폴더 순으로 찾는다."""
    for d in (ROOT / "data", ROOT, ROOT.parent):
        if (d / DATA_NAME).exists():
            return d / DATA_NAME
    raise SystemExit(f"{DATA_NAME} 을 찾을 수 없습니다. KAMP에서 내려받아 {ROOT / 'data'} 에 넣어 주세요.")


def load_data(path: Path | None = None) -> pd.DataFrame:
    """제1장 전처리 후, 1시간 간격의 빈틈없는 시간축으로 반환한다.

    (1) 시간 값이 0~23 밖인 48행(7/13, 7/15) 삭제 — 값이 한 칸씩 밀려 원래 측정 시각을 알 수 없음
    (2) 공장인원 결측 17행(8/28 18시~8/29 10시) 삭제 — 생산량·전력이 모두 0이라 계산 불가
    (3) 결측 대체: 강수량 → 0, 풍속 → 연속 결측은 앞값, 단독 결측은 앞뒤 평균
    (4) 삭제된 시간은 빈 행(NaN)으로 남겨 앞뒤 기록을 연속 시간으로 잇지 않음
    추가 열: peak = 시간 최대수요(네 15분 값 중 최댓값, 예측 대상), peak_q = 최댓값이 나온 15분 구간(0~3)
    """
    path = path or find_data()
    for enc in ("utf-8-sig", "cp949"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    raw.columns = [c.strip() for c in raw.columns]
    df = raw.rename(columns={"공장직원": "공장인원"})

    df = df[df["시간"].between(0, 23)].copy()                       # (1)
    df = df[df["공장인원"].notna()].copy()                           # (2)
    df["ts"] = pd.to_datetime(df["날짜"].astype(str)) + pd.to_timedelta(df["시간"], unit="h")
    df = df.set_index("ts").sort_index()

    df["강수량"] = df["강수량"].fillna(0)                             # (3)
    w = df["풍속"]
    run_len = w.isna().groupby(w.notna().cumsum()).transform("sum")
    single = w.isna() & (run_len == 1)
    df["풍속"] = w.where(~single, (w.ffill() + w.bfill()) / 2).ffill()

    full = pd.date_range(df.index.min().normalize(), df.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    df = df.reindex(full)                                            # (4)
    df.index.name = "ts"
    df["peak"] = df[QCOLS].max(axis=1)
    df["peak_q"] = df[QCOLS].to_numpy().argmax(1)
    return df
