"""모델 정의: 제안 모델 PeakSense와 비교 모델 (보고서 제2장)"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import HUBER, LGB, RF, ROUTE_THR, RUN_THR


def lgb_reg(**kw):
    return lgb.LGBMRegressor(**{**LGB, **kw})


class SingleLGBM:
    """④ LightGBM 단일 모델: 모든 시간을 하나의 회귀기로 예측 (PeakSense와 같은 입력·설정)"""

    def fit(self, X, y):
        self.m = lgb_reg(**HUBER).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GuideRF:
    """③ 가이드북 방식 랜덤포레스트"""

    def fit(self, X, y):
        self.m = RandomForestRegressor(**RF).fit(X.fillna(-1), y)
        return self

    def predict(self, X):
        return self.m.predict(X.fillna(-1))


class PeakSense:
    """⑤ 제안 모델: 가동상태 판별기 + 상태별 전문 회귀기

    1) 데이터에 가동·정지 정보가 없으므로 '가동 라벨'을 새로 정의: 시간 최대수요 > run_thr → 1(가동)
    2) 판별기(LightGBM 분류)가 예측 시점에 알 수 있는 입력으로 가동 확률 p를 추정
    3) p ≥ route_thr 이면 가동 상태 회귀기, 아니면 정지 상태 회귀기(각각 자기 상태 시간으로만 학습, Huber 손실)
    """

    def __init__(self, run_thr=RUN_THR, route_thr=ROUTE_THR, label="power"):
        self.run_thr, self.route_thr, self.label = run_thr, route_thr, label
        self.clf = lgb.LGBMClassifier(**LGB)
        self.run = lgb_reg(**HUBER)
        self.idle = lgb_reg(**HUBER)

    def _label(self, X, y):
        if self.label == "production":          # 비교용: 생산량 > 0 으로 라벨을 붙인 경우(제2장 [표8])
            return (X["prod"] > 0).astype(int)
        return (y > self.run_thr).astype(int)

    def fit(self, X, y):
        s = self._label(X, y)
        self.clf.fit(X, s)
        self.run.fit(X[s == 1], y[s == 1])
        self.idle.fit(X[s == 0], y[s == 0])
        return self

    def parts(self, X):
        """가동 확률, 가동 회귀기 예측, 정지 회귀기 예측"""
        return self.clf.predict_proba(X)[:, 1], self.run.predict(X), self.idle.predict(X)

    def predict_parts(self, X):
        p, mr, mi = self.parts(X)
        return p, np.where(p >= self.route_thr, mr, mi)

    def predict(self, X):
        return self.predict_parts(X)[1]

    def predict_soft(self, X):
        """비교용: 두 회귀기 예측을 가동 확률로 섞는 방식(제2장 [표6])"""
        p, mr, mi = self.parts(X)
        return p * mr + (1 - p) * mi

    def contrib(self, X):
        """행마다 실제로 적용된 회귀기의 SHAP 기여(전력 단위), (행 수 × 특성 수) — 제3장"""
        p = self.clf.predict_proba(X)[:, 1]
        cr = self.run.predict(X, pred_contrib=True)[:, :-1]
        ci = self.idle.predict(X, pred_contrib=True)[:, :-1]
        return np.where((p >= self.route_thr)[:, None], cr, ci)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


# ---------------------------------------------------------------------------- ⑥ GRU (선택, PyTorch 필요)
SEQ_COLS = ["prod_log", "run", "startup", "shutdown", "run_streak", "idle_streak", "hours_to_stop",
            "prod_next1", "prod_prev1", "기온", "습도", "풍속", "강수량", "인건비", "cdd",
            "mean_12h", "last_obs", "sunday", "saturday", "run_day_hours", "prod_day_total"]


class GRUModel:
    """하루 24시간의 입력 흐름 → 24시간 출력 (양방향 GRU, 초기값 2개 평균)"""

    def __init__(self, epochs=120, seeds=(0, 1), hidden=48):
        import torch
        from torch import nn
        torch.set_num_threads(4)
        self.torch, self.nn = torch, nn
        self.epochs, self.seeds, self.hidden = epochs, seeds, hidden

    def _net(self, f):
        torch, nn, h = self.torch, self.nn, self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(f + 24, h)
                self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
                self.out = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))

            def forward(self, x):
                pos = torch.eye(24).unsqueeze(0).expand(x.shape[0], -1, -1)
                z, _ = self.gru(torch.relu(self.inp(torch.cat([x, pos], -1))))
                return self.out(z).squeeze(-1)
        return Net()

    def fit(self, X, y):
        torch = self.torch
        self.cols = [c for c in SEQ_COLS if c in X]
        A = X[self.cols].astype(float).fillna(0.0).groupby(X.index.normalize()).filter(lambda g: len(g) == 24)
        days = A.index.normalize().unique()
        arr = A.to_numpy().reshape(len(days), 24, len(self.cols))
        self.mu, self.sd = arr.reshape(-1, arr.shape[-1]).mean(0), arr.reshape(-1, arr.shape[-1]).std(0) + 1e-6
        Y = y.reindex(A.index).to_numpy().reshape(len(days), 24)
        xt = torch.tensor((arr - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(Y / 100.0, dtype=torch.float32)
        self.nets = []
        for s in self.seeds:
            torch.manual_seed(s)
            net = self._net(xt.shape[-1])
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
            loss_fn = self.nn.HuberLoss(delta=0.15)
            for _ in range(self.epochs):
                perm = torch.randperm(len(xt))
                for i in range(0, len(xt), 16):
                    b = perm[i:i + 16]
                    opt.zero_grad()
                    loss_fn(net(xt[b]), yt[b]).backward()
                    opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, X):
        torch = self.torch
        out = pd.Series(np.nan, index=X.index)
        A = X[self.cols].astype(float).fillna(0.0)
        for _, g in A.groupby(X.index.normalize()):
            if len(g) != 24:
                continue
            xt = torch.tensor(((g.to_numpy() - self.mu) / self.sd)[None], dtype=torch.float32)
            with torch.no_grad():
                out[g.index] = np.mean([n(xt).numpy()[0] for n in self.nets], axis=0) * 100
        return out.to_numpy()
