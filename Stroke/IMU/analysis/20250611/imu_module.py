import pandas as pd
import io
from scipy.signal import butter, filtfilt
import pickle
import numpy as np
from scipy.signal import welch

def read_ags_dataframe(csv_file):
    """
    CSVファイルから 'ags' で始まる行だけを読み込んで DataFrame を返す。

    Args:
        csv_file: Path オブジェクトまたはファイルパス文字列。

    Returns:
        pandas.DataFrame: 'ags' で始まる行からなる DataFrame。
                         ヘッダーは自動で推測されず、列名は ['col0', 'col1', ...] となる。
    """
    # 'ags' で始まる行を抽出
    with open(csv_file, 'r') as f:
        ags_lines = [line for line in f if line.startswith('ags')]

    df = pd.read_csv(
        io.StringIO(''.join(ags_lines)),  # 抽出した行を文字列として渡す
        sep=',',
        header=None,
    )
    df.columns = ['ags', 'time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    df.drop(['ags', 'gyro_x', 'gyro_y', 'gyro_z'], axis=1, inplace=True)
    return df

def arrange_imu_data(imu_df, sync_time):
    #開始をsync_timeに合わせる
    if imu_df["time"].iloc[0] <= sync_time:
        imu_df = imu_df[imu_df["time"] >= sync_time]
        imu_df.reset_index(drop=True, inplace=True)
    else:
        imu_df_start = imu_df["time"].iloc[0]
        for time in range(sync_time, imu_df_start, 10):
            imu_df.loc[-1] = [time, 0, 0, 0]
            imu_df.index = imu_df.index + 1
            imu_df = imu_df.sort_index()
    return imu_df

def resampling(df, time_col='time', pre_Hz=100, post_Hz=60):
    pre_period_ms = 1000 / pre_Hz
    post_period_ms = 1000 / post_Hz
    df['bin'] = (df[time_col] / (post_period_ms)).astype(int)
    downsampled_df = df.groupby('bin').mean()
    downsampled_df[time_col] = downsampled_df.index * post_period_ms
    downsampled_df = downsampled_df.reset_index(drop=True)
    return downsampled_df

def butter_lowpass_fillter(imu_df, sampling_freq, order, cutoff_freq):
    for col in list(imu_df.columns):
        if col.startswith("acc_"):
            nyquist_freq = sampling_freq / 2
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, imu_df[col])
            imu_df.loc[:, col] = y
    return imu_df

def subtract_mean(imu_df):
    # 各列の平均値を引く
    for col in list(imu_df.columns):
        if col.startswith("acc"):
            # print((f"imu_df[{col}].mean():{imu_df[col].mean()}"))
            imu_df[col] = imu_df[col] - imu_df[col].mean()
    return imu_df

def loadPickle(filename):
    with open(filename, "rb") as f:
        file_data = pickle.load(f)
    return file_data

def save_as_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"{filename}を保存しました。")

def find_sync_frame(imu_df, col_name):
    fillter_coef = 3
    # print(f"imu_df:{imu_df}")
    frame_check_df = imu_df[[col_name]].copy()
    q1 = frame_check_df[col_name].quantile(0.25)  # 第1四分位数
    q3 = frame_check_df[col_name].quantile(0.75)  # 第3四分位数
    iqr = q3 - q1
    upper_bound = q3 + fillter_coef * iqr  # 外れ値の上限
    lower_bound = q1 - fillter_coef * iqr  # 外れ値の下限
    if col_name=="acc_x":
        filtered_df = frame_check_df[frame_check_df[col_name] > upper_bound].copy()
        filtered_df.sort_values(col_name, ascending=False, inplace=True)
    elif col_name=="acc_y":
        filtered_df = frame_check_df[frame_check_df[col_name] < lower_bound].copy()
        filtered_df.sort_values(col_name, ascending=True, inplace=True)
    # print(F"filtered_df:{filtered_df[col_name][:10]}")
    fillter_list = []
    skip_list = []
    for frame in filtered_df.index:
        if frame in skip_list:
            continue
        fillter_list.append(frame)
        [skip_list.append(i) for i in range(frame-10, frame+10)]  # 前後10フレーム分はスキップ
    imu_sync_frame = min(fillter_list)  #外れ値の中でフレーム数が最少の位置を同期フレームとする
    # print(f"imu_sync_frame_x:{imu_sync_frame}")
    return imu_sync_frame

def calGaitPhase(ic_frame_dict_ori, to_frame_dict_ori):
    phase_dict = {key : [] for key in ic_frame_dict_ori.keys()}
    for iPeople in range(len(ic_frame_dict_ori)):
        ic_frame_dict = ic_frame_dict_ori[f"{iPeople}"]
        to_frame_dict = to_frame_dict_ori[f"{iPeople}"]
        print(f"ic_frame_dict:{ic_frame_dict}")
        print(f"to_frame_dict:{to_frame_dict}")
        phase_frame_list = []
        for i in range(len(ic_frame_dict["IC_L"])):
            try:
                IC_l_side_frame = ic_frame_dict["IC_L"][i]
                TO_r_side_frame = [to_frame for to_frame in to_frame_dict["TO_R"] if to_frame > IC_l_side_frame][0]
                IC_r_side_frame = [ic_frame for ic_frame in ic_frame_dict["IC_R"] if ic_frame > TO_r_side_frame][0]
                To_l_side_frame = [to_frame for to_frame in to_frame_dict["TO_L"] if to_frame > IC_r_side_frame][0]
                Next_IC_l_side_frame = [ic_frame for ic_frame in ic_frame_dict["IC_L"] if ic_frame > To_l_side_frame][0]

                if Next_IC_l_side_frame > [ic_frame for ic_frame in ic_frame_dict["IC_L"] if ic_frame > IC_l_side_frame][0]:
                    continue
                # phase_frame_list.append([IC_l_side_frame, TO_r_side_frame, IC_r_side_frame, To_l_side_frame, Next_IC_l_side_frame])
                phase_frame_list.append([IC_l_side_frame, TO_r_side_frame, To_l_side_frame, Next_IC_l_side_frame])
            except:
                continue
        print(f"phase_frame_list:{phase_frame_list}")
        phase_dict[f"{iPeople}"] = phase_frame_list
    return phase_dict


def adjust_gait_event(target_frame_list, base_frame_list):
    # 基準となるフレームリストに合わせて、ターゲットのフレームリストを調整する。
    while target_frame_list[0] < base_frame_list[0]:
        target_frame_list = target_frame_list[1:]
    while target_frame_list[-1] > base_frame_list[-1]:
        target_frame_list = target_frame_list[:-1]

    for i in range(len(base_frame_list)-1):  # 基準フレームリストの間にターゲットフレームが入っているか確認
        # baseframelistの最後の要素が足りない場合、最後のフレームを基準に追加する
        if i == len(base_frame_list)-2 and len(target_frame_list) < len(base_frame_list) -1:
            while len(target_frame_list) < len(base_frame_list) -1:
                target_frame_list.append(target_frame_list[-1] + int(np.mean(np.diff(base_frame_list))))
            break
        # baseframelistの最後の一つ手前までの間にtargetframelistが入っていない場合、最後のフレームを基準に追加する
        if base_frame_list[i] < target_frame_list[i] < base_frame_list[i+1]:
            pass
        else:
            if i == 0:
                target_frame_list.insert(i, base_frame_list[i+1] - int(np.mean(np.diff(base_frame_list))))
            else:
                target_frame_list.insert(i, base_frame_list[i+1] + int(np.mean(np.diff(base_frame_list))))
        # print(f"target_frame_list:{target_frame_list}")
    return target_frame_list

def get_gait_event_block(IC_frame_l, IC_frame_r, TO_frame_l, TO_frame_r):
    # 両足のIC, TOフレームリストから、歩行イベントのブロックを取得する。
    gait_event_block = np.zeros((len(IC_frame_l)-1,5), dtype=int)
    for i in range(len(IC_frame_l)-1):
        gait_event_block[i] = [IC_frame_l[i], TO_frame_r[i], IC_frame_r[i], TO_frame_l[i], IC_frame_l[i+1]]
    # print(f"gait_event_block:{gait_event_block}")
    return gait_event_block

def get_event_percent_block(gait_event_frame_array):
    # print(f"event_frame_array:{gait_event_frame_array}")
    # イベントのフレームリストから、各イベントの割合を計算する。
    event_percent_block = np.zeros((len(gait_event_frame_array), 5), dtype=float)
    for i, event_frame in enumerate(gait_event_frame_array):
        event_percent_block[i] = (event_frame - event_frame[0]) / (event_frame[-1]- event_frame[0]) * 100
    # print(f"event_percent_block:{event_percent_block}")
    return event_percent_block

def frame2percent(acc_frame_series):
    ori_idx = acc_frame_series.index.to_numpy()
    normalized_ori_idx = (ori_idx - ori_idx[0]) / (ori_idx[-1] - ori_idx[0]) * 100  # 横軸を0~100に正規化
    acc_data = acc_frame_series.to_numpy()
    # 0~99で1刻みになるようリサンプリング
    new_start_idx = 0
    new_end_idx = 100
    num_points = 100
    new_idx = np.linspace(new_start_idx, new_end_idx, num_points)
    new_acc_data = np.interp(new_idx, normalized_ori_idx, acc_data)
    # print(f"new_acc_data:{new_acc_data}")
    return new_acc_data

def calc_harmonic_ratio(acc_data, axis="xorz"):
    # 加速度データから調和比を計算する
    # フーリエ変換をして周波数成分を取得
    fft_coeffs = np.fft.rfft(acc_data)
    # 振幅を計算
    amplitude = np.abs(fft_coeffs)

    # # 振幅スペクトルをプロットする
    # import matplotlib.pyplot as plt
    # plt.plot(amplitude[1:21], label='Amplitude Spectrum')
    # plt.title(f'Amplitude Spectrum for {axis} axis')
    # plt.xlabel('Frequency Bin')
    # plt.ylabel('Amplitude')
    # plt.xticks(np.arange(0, 21, 1))
    # plt.legend()
    # plt.grid()
    # plt.show()

    # 奇数次高調波および偶数次高調波の振幅を取得
    sum_odd = np.sum(amplitude[1:21:2])  # 奇数次高調波の振幅の合計
    sum_even = np.sum(amplitude[2:21:2])  # 偶数次高調波の振幅の合計
    # 調和比を計算
    if axis == "xorz": #前後方向および垂直方向の場合
        harmonic_ratio = sum_even / sum_odd
    elif axis == "y":  #左右方向の加速度データの場合
        harmonic_ratio = sum_odd / sum_even
    return harmonic_ratio

def calc_f95(acc_x, acc_y, acc_z, sampling_freq):
    """
    論文を参考にwelchの方法で95%周波数を計算する．
    窓の持続時間が10秒とサンプリング周波数100Hzからnpersegを計算
    Args:
        acc_data (numpy.ndarray): 加速度データ
        sampling_freq (int): サンプリング周波数
    """
    print(f"長さ acc_x:{len(acc_x)}, acc_y:{len(acc_y)}, acc_z:{len(acc_z)}")

    window_duration = 2  # 窓の持続時間（秒）論文だと10秒だが、長さが足りないため2秒に設定
    nperseg = int(window_duration * sampling_freq)  # 窓のサンプル数を計算 *100
    # 50%のオーバーラップとハミング窓を使用
    noverlap = nperseg // 2
    window = 'hamming'
    print(f"nperseg:{nperseg}, noverlap:{noverlap}, window:{window}")
    # パワースペクトル密度を計算
    frequencies, psd_x = welch(acc_x, fs=sampling_freq, window=window, nperseg=nperseg, noverlap=noverlap)
    _, psd_y = welch(acc_y, fs=sampling_freq, window=window, nperseg=nperseg, noverlap=noverlap)
    _, psd_z = welch(acc_z, fs=sampling_freq, window=window, nperseg=nperseg, noverlap=noverlap)
    # 各軸のパワースペクトル密度を合計
    psd = psd_x + psd_y + psd_z
    cumulative_psd = np.cumsum(psd)  # 累積パワースペクトル密度を計算
    total_power = cumulative_psd[-1]  # 全体のパワーを取得
    power_threshold = 0.95 * total_power  # 95%のパワーを計算
    try:
        f95_index = np.where(cumulative_psd >= power_threshold)[0][0]  # 95%のパワーを超える最初のインデックスを取得
    except IndexError:
        print("Warning: 95%のパワーを超える周波数が見つかりませんでした。")
    f95 = frequencies[f95_index]  # 95%周波数を取得
    return f95

def calc_ApEn(acc_data, m=1, r=0.2, tau=1):
    """
    近似エントロピー(ApEn)を計算
    Args:
        time_series (np.ndarray): 1次元の時系列データ。
        m (int): 埋め込み次元（パターンの長さ）。
        r (float): 半径（類似性を判断するしきい値）。
        tau (int): 時間遅延。デフォルトは1。

    Returns:
        float: 計算された近似エントロピーの値。
    """
    def _phi(m_local: int) -> float:
        """指定された埋め込み次元でΩ(r)を計算するヘルパー関数。"""
        n = len(acc_data)
        num_vectors = n - (m_local - 1) * tau
        # 埋め込みベクトルを作成
        embedding_vectors = np.array([acc_data[i : i + (m_local - 1) * tau + 1 : tau] for i in range(num_vectors)])
        # 各ベクトルに対する類似パターンの数をカウント
        # 論文の定義通り、距離はチェビシェフ距離（各要素の差の最大値）
        counts = np.array([np.sum(np.max(np.abs(embedding_vectors - vec), axis=1) <= r) for vec in embedding_vectors])
        # ゼロカウントを避けるため、ゼロより大きいものだけを対象にlogをとる
        # ゼロの場合はlog(1/N)となり非常に小さい値になるが、ここでは単純に無視する
        C_i_m_r = counts / num_vectors
        valid_indices = C_i_m_r > 0
        if not np.any(valid_indices):
            return 0.0 # 有効なC_i_m_rがない場合は0を返す
        # Ω(r)の計算
        return np.mean(np.log(C_i_m_r[valid_indices]))
    # ApEn = Ω^m(r) - Ω^(m+1)(r)
    return _phi(m) - _phi(m + 1)