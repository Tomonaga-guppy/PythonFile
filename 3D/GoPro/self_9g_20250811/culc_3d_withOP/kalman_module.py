import numpy as np
from scipy.optimize import minimize

def _local_trend_kf(y, a1, P1, varEta, varEps):
    """
    ローカルトレンドモデルのカルマンフィルタリングを実行する。
    MATLABのlocalTrendKF関数の移植版。
    """
    L = len(y)
    a_tt1 = np.zeros(L + 1)
    a_tt1[0] = a1
    P_tt1 = np.zeros(L + 1)
    P_tt1[0] = P1
    v_t = np.zeros(L)
    F_t = np.zeros(L)
    a_tt = np.zeros(L)
    P_tt = np.zeros(L)
    K_t = np.zeros(L)

    for t in range(L):
        v_t[t] = y[t] - a_tt1[t]
        F_t[t] = P_tt1[t] + varEps
        if np.abs(F_t[t]) < 1e-8: F_t[t] = 1e-8
        K_t[t] = P_tt1[t] / F_t[t]
        a_tt[t] = a_tt1[t] + K_t[t] * v_t[t]
        P_tt[t] = P_tt1[t] * (1 - K_t[t])
        a_tt1[t + 1] = a_tt[t]
        P_tt1[t + 1] = P_tt[t] + varEta

    return a_tt, P_tt, F_t, v_t, a_tt1, P_tt1

def _calc_log_diffuse_llhd(params, y):
    """
    ローカルトレンドモデルの散漫な対数尤度を計算する。
    MATLABのcalcLogDiffuseLlhd関数の移植版。
    """
    # ★★★ 修正点: パラメータをクリップしてオーバーフローを防ぐ ★★★
    psiEta = np.clip(params[0], -10, 10)
    psiEps = np.clip(params[1], -10, 10)

    varEta = np.exp(2 * psiEta)
    varEps = np.exp(2 * psiEps)

    if len(y) < 2:
        return np.inf

    a1 = y[0]
    P1 = varEps

    _, _, F_t, v_t, _, _ = _local_trend_kf(y, a1, P1, varEta, varEps)

    F_t[F_t <= 1e-8] = 1e-8

    # ★★★ 修正点: inf/nan が含まれている場合は大きな値を返して最適化を誘導 ★★★
    if np.any(np.isinf(F_t)) or np.any(np.isnan(F_t)) or np.any(np.isinf(v_t)) or np.any(np.isnan(v_t)):
        return np.inf

    tmp = np.sum(np.log(F_t[1:]) + v_t[1:]**2 / F_t[1:])
    logLd = -0.5 * len(y) * np.log(2 * np.pi) - 0.5 * tmp

    return -logLd if np.isfinite(logLd) else np.inf

def predict_next_point(history_sequence, initial_value=0.005):
    """
    時系列データの履歴から、二階差分カルマンフィルタを用いて次の点を予測する。
    """
    # ★★★ 修正点: 入力データの検証 ★★★
    if not np.all(np.isfinite(history_sequence)):
        return history_sequence[-1] if len(history_sequence) > 0 and np.isfinite(history_sequence[-1]) else 0.0

    if len(history_sequence) < 5:
        # 履歴が短い場合は線形外挿
        if len(history_sequence) > 1:
            return history_sequence[-1] + (history_sequence[-1] - history_sequence[-2])
        return history_sequence[-1] if len(history_sequence) > 0 else 0.0

    velocity = np.diff(history_sequence)

    if len(velocity) >= 3:
        window_size = 3
        b = np.ones(window_size) / window_size
        velocity_maf = np.convolve(velocity, b, mode='valid')
    else:
        velocity_maf = velocity

    if len(velocity_maf) < 2:
        return history_sequence[-1]

    # ★★★ 修正点: 最適化処理をtry-exceptで囲み、失敗した場合のフォールバック処理を追加 ★★★
    try:
        par = initial_value
        psiEta = np.log(np.sqrt(par))
        psiEps = np.log(np.sqrt(par))
        x0 = [psiEta, psiEps]

        res = minimize(_calc_log_diffuse_llhd, x0, args=(velocity_maf,), method='BFGS', options={'disp': False})

        if not res.success:
            raise ValueError("Optimization failed")

        varEtaOpt = np.exp(2 * np.clip(res.x[0], -10, 10))
        varEpsOpt = np.exp(2 * np.clip(res.x[1], -10, 10))

        varEps = varEtaOpt
        varEta = varEpsOpt

        a1 = velocity_maf[0]
        P1 = varEps
        _, _, _, _, a_tt1, _ = _local_trend_kf(velocity_maf, a1, P1, varEta, varEps)

        predicted_velocity = a_tt1[-1]

        # 予測速度が異常に大きい場合は制限
        if np.abs(predicted_velocity) > np.std(velocity) * 5 and len(velocity)>1:
            predicted_velocity = np.mean(velocity)

        next_point = history_sequence[-1] + predicted_velocity

    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        print("Optimization failed, using fallback")
        # 最適化が失敗した場合は、単純な線形外挿で代用
        predicted_velocity = velocity[-1] if len(velocity) > 0 else 0
        next_point = history_sequence[-1] + predicted_velocity

    return next_point if np.isfinite(next_point) else history_sequence[-1]
