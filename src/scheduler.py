"""생산일정 후보 생성과 점수 계산

역할: 계획을 바꿀 때 생산 흐름 입력을 다시 만들고 여러 후보를 예측해 더 낮은 목적함수의 후보를 찾습니다.
입력과 호출: 학습 모델, 생산계획, 대상 날짜, 생산량 상한. DaySchedule.predict_many(), optimize().
출력과 범위: 조정 계획과 예상 전력을 반환합니다. 상위 3개 예측 전력과 인건비 항을 조합하며 알려진 제약·정규화 문제는 docs/LIMITATIONS.md에 기록했습니다.
상세: src/README.md 및 docs/FILE_GUIDE.md
"""
import numpy as np
import pandas as pd



def plan_features_batch(P, n_before):
    """plan_features와 같은 계산을 여러 계획(n×L, 시간순 연속 구간)에 한 번에 적용 → {특성: n×24}.
    n_before: 구간 시작부터 대상일 0시까지의 시간 수(72 이상이면 연속 시간 특성이 원래 계산과 같다)"""
    P = np.asarray(P, float)
    n, L = P.shape
    run = (P > 0).astype(int)
    pad = lambda A, k: np.concatenate([np.zeros((n, k)), A[:, :-k]], 1) if k > 0 else np.concatenate([A[:, -k:], np.zeros((n, -k))], 1)  # noqa: E731
    prev_run = np.concatenate([np.zeros((n, 1), int), run[:, :-1]], 1)
    streak = np.ones((n, L))
    for t in range(1, L):
        streak[:, t] = np.where(run[:, t] == run[:, t - 1], streak[:, t - 1] + 1, 1)
    size_left = np.ones((n, L))                     # 같은 상태가 이후로 이어지는 시간(자기 포함)
    for t in range(L - 2, -1, -1):
        size_left[:, t] = np.where(run[:, t] == run[:, t + 1], size_left[:, t + 1] + 1, 1)
    roll = P + pad(P, 1) + pad(P, -1)
    s = slice(n_before, n_before + 24)
    out = {"prod": P[:, s], "prod_log": np.log1p(P[:, s]), "run": run[:, s],
           "prod_prev1": pad(P, 1)[:, s], "prod_prev2": pad(P, 2)[:, s],
           "prod_next1": pad(P, -1)[:, s], "prod_next2": pad(P, -2)[:, s],
           "startup": ((run == 1) & (prev_run == 0)).astype(int)[:, s],
           "shutdown": ((run == 0) & (prev_run == 1)).astype(int)[:, s],
           "run_streak": np.where(run == 1, streak, 0).clip(max=72)[:, s],
           "idle_streak": np.where(run == 0, streak, 0).clip(max=72)[:, s],
           "hours_to_stop": np.where(run == 1, size_left - 1, 0).clip(max=72)[:, s],
           "prod_day_total": np.repeat(P[:, s].sum(1, keepdims=True), 24, 1),
           "run_day_hours": np.repeat(run[:, s].sum(1, keepdims=True), 24, 1),
           "prod_roll3": roll[:, s]}
    return out


class DaySchedule:
    def __init__(self, model, X, prod, feats, day, cap):
        self.model, self.X, self.prod, self.F, self.day, self.cap = model, X, prod, feats, day, cap
        self.idx = X.index[X.index.normalize() == day]
        self.labor = X.loc[self.idx, "인건비"].to_numpy(float)

    def predict_many(self, plans):
        """여러 계획(n×24)을 한 번에 예측 → (n×24). 앞뒤 3일의 실제 계획을 붙여 생산 흐름 특성을 다시 계산"""
        plans = np.asarray(plans, float)
        lo, hi = self.day - pd.Timedelta(days=3), self.day + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
        ctx = self.prod.reindex(pd.date_range(lo, hi, freq="h")).fillna(0).to_numpy(float)
        P = np.repeat(ctx[None], len(plans), 0)
        P[:, 72:96] = plans
        PF = plan_features_batch(P, 72)
        base = self.X.loc[self.idx, self.F]
        A = np.repeat(base.to_numpy(float)[None], len(plans), 0)
        for j, c in enumerate(self.F):
            if c in PF:
                A[:, :, j] = PF[c]
        A = pd.DataFrame(A.reshape(-1, len(self.F)), columns=self.F)
        return self.model.predict(A).reshape(len(plans), 24)

    def objective(self, pred, plans):
        top = np.sort(pred, axis=1)[:, ::-1]
        labor = (np.asarray(plans) * (self.labor - 1)).sum(1) / max(np.asarray(plans).sum(), 1)
        return top[:, 0] + 0.2 * top[:, 1] + 0.05 * top[:, 2] + 0.01 * labor   # 1순위 일 최대, 2·3번째 큰 값과 인건비는 보조

    def optimize(self, plan0, max_shift=2, step_frac=0.25, max_iter=60, allow=None):
        plan = np.asarray(plan0, float).copy()
        allow = (plan > 0) if allow is None else allow
        best_pred = self.predict_many([plan])[0]
        best = self.objective(best_pred[None], [plan])[0]
        for _ in range(max_iter):
            cands = []
            for i in np.flatnonzero(plan > 0):
                d = plan[i] * step_frac
                for j in range(max(0, i - max_shift), min(24, i + max_shift + 1)):
                    if j == i or not allow[j] or plan[j] + d > self.cap:
                        continue
                    c = plan.copy(); c[i] -= d; c[j] += d
                    cands.append(c)
            if not cands:
                break
            P = self.predict_many(cands)
            obj = self.objective(P, cands)
            k = int(np.argmin(obj))
            if obj[k] >= best - 1e-6:
                break
            plan, best, best_pred = cands[k], obj[k], P[k]
        return plan, best_pred
