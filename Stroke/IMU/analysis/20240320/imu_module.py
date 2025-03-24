import pandas as pd
import io
from scipy.signal import butter, filtfilt
import pickle

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
    # print(f"imu_df:{imu_df}")
    for col in list(imu_df.columns):
        if col.startswith("acc_"):
            nyquist_freq = sampling_freq / 2
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, imu_df[col])
            imu_df.loc[:, col] = y
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

import matplotlib.pyplot as plt
import numpy as np

def create_plot(x_data, y_data_list, plot_type='plot', labels=None, colors=None, linestyles=None, markers=None,
                title=None, xlabel=None, ylabel=None, grid=False, save_path=None, figsize=(8, 6), plt_show=False,
                plt_close=True, plt_save=True, legend_outside=False):
    """
    汎用的なグラフ作成関数

    Args:
        x_data (list-like): x軸データ
        y_data_list (list of list-like): y軸データ (複数のyデータ系列をリストで指定)
        plot_type (str, optional): グラフの種類 ('plot', 'scatter', 'bar'など). デフォルトは 'plot'
        labels (list of str, optional): 各データ系列のラベル (凡例用). デフォルトは None
        colors (list of str, optional): 各データ系列の色. デフォルトは None (Matplotlibのデフォルト色を使用)
        linestyles (list of str, optional): 折れ線グラフの線種. デフォルトは None (Matplotlibのデフォルト線種を使用)
        markers (list of str, optional): 散布図などのマーカーの種類. デフォルトは None (マーカーなし)
        title (str, optional): グラフタイトル. デフォルトは None
        xlabel (str, optional): x軸ラベル. デフォルトは None
        ylabel (str, optional): y軸ラベル. デフォルトは None
        grid (bool, optional): グリッド線の表示/非表示. デフォルトは False (非表示)
        save_path (str, optional): グラフの保存パス. デフォルトは None (保存しない)
        figsize (tuple, optional): 図のサイズ (width, height). デフォルトは (8, 6)
    """
    if legend_outside:
        fig, ax = plt.subplots(figsize=(figsize[0]+1, figsize[1]))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # データ系列ごとにプロット
    for i, y_data in enumerate(y_data_list):
        label = labels[i] if labels else f'Data {i+1}' # ラベルが指定されていなければ自動でラベル生成
        color = colors[i] if colors else None
        linestyle = linestyles[i] if linestyles and plot_type == 'plot' else None # 折れ線グラフ以外ではlinestyleは無視
        marker = markers[i] if markers else None

        if plot_type == 'plot':
            ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle, marker=marker)
        elif plot_type == 'scatter':
            ax.scatter(x_data, y_data, label=label, color=color, marker=marker)
        elif plot_type == 'bar':
            ax.bar(x_data, y_data, label=label, color=color) # bar plot はlinestyleとmarkerをサポートしない
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Supported types are 'plot', 'scatter', 'bar'")

    # グラフ要素の設定
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if labels and plot_type != 'bar': # bar plot はlegendを表示しない方が良い場合がある
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            ax.legend()

    if grid:
        ax.grid(True)

    ax.margins(x=0) # グラフの端とデータの間に余白を設

    fig.tight_layout()

    # グラフの保存
    if save_path and plt_save:
        plt.savefig(save_path)

    if plt_show:
        plt.show()

    if plt_close:
        plt.close()


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
