"""
中身はtest20241016のqualysis_refと同じ
"""

import module_mocap as moc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as mticker

tsv_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\psub_label2")
# tsv_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm")
# tsv_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm\test_20241016")
tsv_files = tsv_dir.glob("*0003*.tsv")
# tsv_files = tsv_dir.glob("*sub4_com*.tsv")
tsv_files = list(tsv_files)
# tpose_path = tsv_dir / "sub4_tpose_ref_pos.json"
tpose_path = tsv_dir / "sub1-0001_ref_pos.json"


def plot_interpolation_results(dfs, labels, marker_name, output_path):
    """
    指定されたマーカーの補間結果をプロットして保存する。
    補間ステップ2と3で新たに埋められた点を強調表示する。
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
    coords = ['X', 'Y', 'Z']

    original_df, step1_df, step2_df, step3_df, filt_df = dfs

    for i, coord in enumerate(coords):
        ax = axes[i]
        col_name = f'{marker_name} {coord}'

        if col_name not in original_df.columns:
            print(f"警告: マーカー '{marker_name}' の列が見つかりません。プロットをスキップします。")
            plt.close(fig)
            return

        # 元のデータ(欠損あり)を点でプロット
        ax.plot(original_df.index, original_df[col_name], 'o', color='gray', label=labels[0], markersize=3, alpha=0.6)

        # ステップ1の補間結果を基準線としてプロット
        newly_filled_step11 = step1_df[original_df[col_name].isna() & step1_df[col_name].notna()]
        if not newly_filled_step11.empty:
            ax.plot(newly_filled_step11.index, newly_filled_step11[col_name], 'o', color='cyan',
                    label=f'{labels[1]} (newly filled)', markersize=4)

        # ステップ2で「新たに」補間された点のみを抽出してプロット
        newly_filled_step2 = step2_df[step1_df[col_name].isna() & step2_df[col_name].notna()]
        if not newly_filled_step2.empty:
            ax.plot(newly_filled_step2.index, newly_filled_step2[col_name], 'o', color='orange',
                    label=f'{labels[2]} (newly filled)', markersize=5)

        # ステップ3で「新たに」補間された点のみを抽出してプロット
        newly_filled_step3 = step3_df[step2_df[col_name].isna() & step3_df[col_name].notna()]
        if not newly_filled_step3.empty:
            ax.plot(newly_filled_step3.index, newly_filled_step3[col_name], 'o', color='blue',
                    label=f'{labels[3]} (newly filled)', markersize=4)
            
        # 平滑化したデータを薄い線でプロット
        ax.plot(filt_df.index, filt_df[col_name], '-', color='black', label=labels[4] + ' (butterworth)', alpha=0.6)

        ax.set_title(f'{marker_name} - {coord} coordinate', fontsize=16) # タイトルを追加
        ax.set_ylabel('Position (mm)', fontsize=16) # Y軸ラベルを追加
        ax.legend(fontsize=12) # 凡例のフォントサイズを設定

        # 目盛りのフォントサイズを設定
        ax.tick_params(axis='both', labelsize=14)

        # 100フレームごとに主要な目盛りとグリッド線を設定
        ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
        ax.grid(which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray')

        # 20フレームごとに補助的な目盛りとグリッド線を設定
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(20))
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.8, color='lightgray', alpha=0.5)



    axes[-1].set_xlabel('Frame', fontsize=16) # X軸ラベルを追加
    fig.suptitle(f'Interpolation Step-by-Step Check for {marker_name}', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.savefig(output_path)
    plt.close(fig)
    print(f"プロット画像を保存しました: '{output_path.name}'")

def main():
    angle_dict = {}
    for itsv, tsv_file in enumerate(tsv_files):
        # 各TSVファイルに対する角度計算のための辞書を初期化
        angle_dict[tsv_file.stem] = {}

        print(f"Processing {itsv+1}/{len(tsv_files)}: {tsv_file.name}")
        full_df = moc.read_tsv(tsv_file)  #tsvファイルの読み込み
        target_df = full_df.copy()
        
        
        
        
        
        
        
        # print(f"target_df: {target_df}")
        nan_df = target_df.replace(0, np.nan)  #0をNaNに置き換え
        # 補間処理1 欠損が20フレーム以下の区間をスプライン補間
        interpolated_df = moc.interpolate_short_gaps(nan_df, max_gap_size=20, method='spline', order=3)
        # 補間処理2 参考点が3点以上ある骨盤の座標を補間
        target_markers_to_fill = ['RASI', 'LASI', 'RPSI', 'LPSI']  # 補間したいターゲット
        interpolated2_df = moc.interpolate_pelvis_rigid_body(interpolated_df, tpose_path, target_markers_to_fill)
        # 補間処理3 骨盤補間後に欠損が20フレーム以下の区間を再度スプライン補間
        interpolated3_df = moc.interpolate_short_gaps(interpolated2_df, max_gap_size=20, method='spline', order=3)

        # 追加の補間処理　補間したいターゲット以外はすべての範囲をスプライン補間
        # 補間対象外とする骨盤マーカーの列名を特定
        pelvis_columns = []
        for marker in target_markers_to_fill:
            pelvis_columns.extend([f'{marker} X', f'{marker} Y', f'{marker} Z'])
        
        # データフレームに実際に存在する列のみに絞り込む
        pelvis_columns = [col for col in pelvis_columns if col in interpolated3_df.columns]

        # 補間対象の列名を特定（骨盤マーカー以外のすべての列）
        other_columns_to_interpolate = [col for col in interpolated3_df.columns if col not in pelvis_columns]

        # 対象の列をループして、ギャップサイズを問わずスプライン補間を実行
        for col in other_columns_to_interpolate:
            # 3次スプライン補間には最低でも4点のデータが必要なため、エラーを回避する
            if interpolated3_df[col].notna().sum() > 3:
                interpolated3_df[col] = interpolated3_df[col].interpolate(method='spline', order=3, limit_direction='both')
            else:
                # データが足りない場合は警告を表示
                print(f"警告: 列 '{col}' の有効データが4点未満のため、全範囲スプライン補間をスキップします。")
        

        # 平滑化処理 バターワースフィルタ
        butter_df = moc.butterworth_filter_no_nan_gaps(interpolated3_df, cutoff=6, order=4, fs=120) #4次のバターワースローパスフィルタ



        
        frame_max = None
        # frame_max = 390
        if frame_max is not None:
            nan_df = nan_df.iloc[:frame_max, :]  #デバッグ用にフレームを限定
            interpolated_df = interpolated_df.iloc[:frame_max, :]
            interpolated2_df = interpolated2_df.iloc[:frame_max, :]
            interpolated3_df = interpolated3_df.iloc[:frame_max, :]
            butter_df = butter_df.iloc[:frame_max, :]
        else:
            nan_df = nan_df
            interpolated_df = interpolated_df
            interpolated2_df = interpolated2_df
            interpolated3_df = interpolated3_df
            butter_df = butter_df

        # print("\n--- 補間結果のプロットを開始 ---")
        # plot_dfs = [nan_df, interpolated_df, interpolated2_df, interpolated3_df, butter_df]
        # plot_labels = ['Original Data (with Gaps)', 'Step 1: Spline Interpolation', 'Step 2: Rigid Body Fitting', 'Step 3: Final Spline Interpolation', 'Filtered Data (Butterworth)']

        # # プロットしたいマーカーをリストで指定
        # markers_to_plot = ['RASI', 'LASI', 'RPSI', 'LPSI']

        # for marker in markers_to_plot:
        #     if f'{marker} X' in full_df.columns:
        #         output_filename = tsv_dir / f"{tsv_file.stem}_{marker}_interpolation_check.png"
        #         plot_interpolation_results(plot_dfs, plot_labels, marker, output_filename)
        #     else:
        #         print(f"マーカー '{marker}' はファイルに存在しないため、プロットをスキップします。")
        # print("--- 補間結果のプロットが完了 ---\n")



        # 角度計算に必要な全てのマーカー名をリストアップ
        required_markers = [
            'RASI', 'LASI', 'RPSI', 'LPSI', 'RKNE', 'LKNE', 'RKNE2', 'LKNE2',
            'RANK', 'LANK', 'RANK2', 'LANK2', 'RTOE', 'LTOE', 'RHEE', 'LHEE'
        ]

        # マーカー名から、対応するすべての列名(X, Y, Z)のリストを作成
        all_marker_columns = []
        for marker in required_markers:
            all_marker_columns.extend([f'{marker} X', f'{marker} Y', f'{marker} Z'])

        # 実際にデータフレームに存在する列のみを対象にする
        existing_columns = [col for col in all_marker_columns if col in interpolated3_df.columns]

        # 必要なマーカーが全て揃っているフレーム（行）のインデックスを抽出
        valid_frames_df = interpolated3_df.dropna(subset=existing_columns)
        valid_frames = valid_frames_df.index

        print(f"全マーカーが存在する有効なフレーム総数: {len(valid_frames)}")
        if len(valid_frames) > 0:
            # 有効フレームのシーケンスの差分を計算し、1より大きい場所（ギャップ）を見つける
            gaps = np.where(np.diff(valid_frames) > 1)[0] + 1
            # ギャップの位置で配列を分割し、連続した範囲のリストを作成する
            ranges = np.split(valid_frames, gaps)

            # print("検出された有効フレームの範囲:")
            # for r in ranges:
            #     if len(r) > 1:
            #         print(f"  - フレーム {r[0]} から {r[-1]} (計 {len(r)} フレーム)")
            #     else: #範囲が1フレームしかない場合
            #         print(f"  - フレーム {r[0]} (単一フレーム)")
        else:
            print("警告: 全てのマーカーが揃っているフレームが1つもありません。角度計算をスキップします。")
            continue

        # 各有効範囲の角度計算結果を保存するリスト
        all_angle_dfs = []
        pel2hee_list = []  #骨盤から左踵へのベクトルを保存するリスト

        for _range in ranges:
            print(f"range: {_range[0]} - {_range[-1]} (計 {len(_range)} フレーム)")
            try:
                # 現在の範囲 'r' のデータのみを抽出する
                rasi = interpolated3_df.loc[_range, ['RASI X', 'RASI Y', 'RASI Z']].to_numpy()
                lasi = interpolated3_df.loc[_range, ['LASI X', 'LASI Y', 'LASI Z']].to_numpy()
                rpsi = interpolated3_df.loc[_range, ['RPSI X', 'RPSI Y', 'RPSI Z']].to_numpy()
                lpsi = interpolated3_df.loc[_range, ['LPSI X', 'LPSI Y', 'LPSI Z']].to_numpy()
                rknee = interpolated3_df.loc[_range, ['RKNE X', 'RKNE Y', 'RKNE Z']].to_numpy()
                lknee = interpolated3_df.loc[_range, ['LKNE X', 'LKNE Y', 'LKNE Z']].to_numpy()
                rknee2 = interpolated3_df.loc[_range, ['RKNE2 X', 'RKNE2 Y', 'RKNE2 Z']].to_numpy()
                lknee2 = interpolated3_df.loc[_range, ['LKNE2 X', 'LKNE2 Y', 'LKNE2 Z']].to_numpy()
                rank = interpolated3_df.loc[_range, ['RANK X', 'RANK Y', 'RANK Z']].to_numpy()
                lank = interpolated3_df.loc[_range, ['LANK X', 'LANK Y', 'LANK Z']].to_numpy()
                rank2 = interpolated3_df.loc[_range, ['RANK2 X', 'RANK2 Y', 'RANK2 Z']].to_numpy()
                lank2 = interpolated3_df.loc[_range, ['LANK2 X', 'LANK2 Y', 'LANK2 Z']].to_numpy()
                rtoe = interpolated3_df.loc[_range, ['RTOE X', 'RTOE Y', 'RTOE Z']].to_numpy()
                ltoe = interpolated3_df.loc[_range, ['LTOE X', 'LTOE Y', 'LTOE Z']].to_numpy()
                rhee = interpolated3_df.loc[_range, ['RHEE X', 'RHEE Y', 'RHEE Z']].to_numpy()
                lhee = interpolated3_df.loc[_range, ['LHEE X', 'LHEE Y', 'LHEE Z']].to_numpy()
            except KeyError as e:
                print(f"エラー: 範囲 {_range[0]}-{_range[-1]} で必要なマーカーが不足: {e}")
                continue

            angle_list_range = []
            # 前フレームの角度を保存する変数
            prev_r_hip_angle, prev_l_hip_angle, prev_r_knee_angle, prev_l_knee_angle, prev_r_ankle_angle, prev_l_ankle_angle = None, None, None, None, None, None

            # ループの範囲を現在の有効範囲 '_range' に限定
            for frame_idx_in_range, original_frame_num in enumerate(_range):
                try:
                    # 座標系定義の計算 (インデックスは範囲内でのインデックス frame_idx_in_range を使用)
                    # print(f"Frame {frame}:")
                    d_asi = np.linalg.norm(rasi[frame_idx_in_range,:] - lasi[frame_idx_in_range,:])  #ASIS間距離
                    """
                    修正前
                    # d_leg_bef = (np.linalg.norm(rank[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]) + np.linalg.norm(lank[frame_idx_in_range, :] - lasi[frame_idx_in_range,:]) / 2)  #大腿長の平均
                    """
                    # 修正後
                    d_leg = (np.linalg.norm(rank[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]) + np.linalg.norm(lank[frame_idx_in_range, :] - lasi[frame_idx_in_range,:])) / 2  #大腿長の平均
                    
                    """
                    print(f"d_asi: {d_asi}, d_leg_bef: {d_leg_bef}, d_leg: {d_leg}")
                    d_leg_bef: 1316.5442363768009, d_leg: 879.4018044930382
                    """

                    r = 0.012 #使用したマーカー径

                    h = 1.70 #被験者の身長
                    # h = 1.76 #被験者(kasa)の身長

                    k = h/1.7
                    beta = 0.1 * np.pi #[rad]
                    theta = 0.496 #[rad]
                    c = 0.115 * d_leg - 0.0153
                    x_dis = 0.1288 * d_leg - 0.04856



                    """
                    以前まで 
                    """
                    # # skycom + davis
                    # x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    # x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    # y_rthigh = +(c * np.sin(theta) - d_asi/2)
                    # y_lthigh = -(c * np.sin(theta)- d_asi/2)
                    # z_rthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
                    # z_lthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
                    # rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
                    # lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

                    # # 仮の骨盤中心 ASISの中点
                    # hip_0 = (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2
                    # # 腰椎節原点
                    # lumbar = (0.47 * (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2 + 0.53 * (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

                    # #骨盤節座標系（原点はhip）
                    # e_y0_pelvis_0 = lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]
                    # e_z_pelvis_0 = (lumbar - hip_0)/np.linalg.norm(lumbar - hip_0)
                    # e_x_pelvis_0 = np.cross(e_y0_pelvis_0, e_z_pelvis_0)/np.linalg.norm(np.cross(e_y0_pelvis_0, e_z_pelvis_0))
                    # e_y_pelvis_0 = np.cross(e_z_pelvis_0, e_x_pelvis_0)




                    """
                    変更後
                    """
                    # skycom + davis
                    x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    y_rthigh = +(c * np.sin(theta) - d_asi/2)
                    y_lthigh = -(c * np.sin(theta)- d_asi/2)
                    z_rthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
                    z_lthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
                    rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
                    lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

                    # 骨盤原点1 ASISの中点
                    hip_0 = (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2
                    # 仙骨 PSISの中点
                    sacrum = (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2

                    #骨盤節座標系1（原点はhip_0）
                    e_y0_pelvis_0 = (lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:])/np.linalg.norm(lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:])
                    e_x_pelvis_0 = (hip_0 - sacrum)/np.linalg.norm(hip_0 - sacrum)
                    e_z_pelvis_0 = np.cross(e_x_pelvis_0, e_y0_pelvis_0)/np.linalg.norm(np.cross(e_x_pelvis_0, e_y0_pelvis_0))
                    e_y_pelvis_0 = np.cross(e_z_pelvis_0, e_x_pelvis_0)

                    #######################################
                    transformation_matrix = np.array([[e_x_pelvis_0[0], e_y_pelvis_0[0], e_z_pelvis_0[0], hip_0[0]],
                                                        [e_x_pelvis_0[1], e_y_pelvis_0[1], e_z_pelvis_0[1], hip_0[1]],
                                                        [e_x_pelvis_0[2], e_y_pelvis_0[2], e_z_pelvis_0[2], hip_0[2]],
                                                        [0,       0,       0,       1]])


                    #モーキャプの座標系に変換してもう一度計算
                    rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
                    lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
                    hip = (rthigh + lthigh) / 2

                    # 腰椎節原点
                    lumbar = (0.47 * (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2 + 0.53 * (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

                    e_y0_pelvis = (lthigh - rthigh)/np.linalg.norm(lthigh - rthigh)
                    e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
                    e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
                    e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
                    rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

                    # # 欠損している膝、足首の座標を算出(元のfukuyamaのやつはそもそも内側のマーカーつけてなかった)
                    # rknee2 = rknee.copy()
                    # rknee2[frame_num, :] = rknee[frame_num, :] + (2 * r + 0.1 * k) * e_y_pelvis
                    # lknee2 = lknee.copy()
                    # lknee2[frame_num, :] = lknee[frame_num, :] - (2 * r + 0.1 * k) * e_y_pelvis
                    # rank2 = rank.copy()
                    # rank2[frame_num, :] = rank[frame_num, :] + (2 * r + 0.06 * k) * e_y_pelvis
                    # lank2 = lank.copy()
                    # lank2[frame_num, :] = lank[frame_num, :] - (2 * r + 0.06 * k) * e_y_pelvis

                    #必要な原点の設定
                    rshank = (rknee[frame_idx_in_range, :] + rknee2[frame_idx_in_range, :]) / 2
                    lshank = (lknee[frame_idx_in_range, :] + lknee2[frame_idx_in_range, :]) / 2
                    rfoot = (rank[frame_idx_in_range,:] + rank2[frame_idx_in_range,:]) / 2
                    lfoot = (lank[frame_idx_in_range, :] + lank2[frame_idx_in_range,:]) / 2

                    #右大腿節座標系（原点はrthigh）
                    e_y0_rthigh = rknee2[frame_idx_in_range, :] - rknee[frame_idx_in_range, :]
                    e_z_rthigh = (rshank - rthigh)/np.linalg.norm(rshank - rthigh)
                    e_x_rthigh = np.cross(e_y0_rthigh, e_z_rthigh)/np.linalg.norm(np.cross(e_y0_rthigh, e_z_rthigh))
                    e_y_rthigh = np.cross(e_z_rthigh, e_x_rthigh)
                    rot_rthigh = np.array([e_x_rthigh, e_y_rthigh, e_z_rthigh]).T

                    #左大腿節座標系（原点はlthigh）
                    e_y0_lthigh = lknee[frame_idx_in_range, :] - lknee2[frame_idx_in_range, :]
                    e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
                    e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
                    e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
                    rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T

                    #右下腿節座標系（原点はrshank）
                    e_y0_rshank = rknee2[frame_idx_in_range, :] - rknee[frame_idx_in_range, :]
                    e_z_rshank = (rshank - rfoot)/np.linalg.norm(rshank - rfoot)
                    e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
                    e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
                    rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

                    #左下腿節座標系（原点はlshank）
                    e_y0_lshank = lknee[frame_idx_in_range, :] - lknee2[frame_idx_in_range, :]
                    e_z_lshank = (lshank - lfoot)/np.linalg.norm(lshank - lfoot)
                    e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
                    e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
                    rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

                    #右足節座標系 AIST参照（原点はrfoot）
                    e_z_rfoot = (rtoe[frame_idx_in_range,:] - rhee[frame_idx_in_range,:]) / np.linalg.norm(rtoe[frame_idx_in_range,:] - rhee[frame_idx_in_range,:])
                    e_y0_rfoot = rank[frame_idx_in_range,:] - rank2[frame_idx_in_range,:]
                    e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
                    e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
                    rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

                    #左足節座標系 AIST参照（原点はlfoot）
                    e_z_lfoot = (ltoe[frame_idx_in_range,:] - lhee[frame_idx_in_range,:]) / np.linalg.norm(ltoe[frame_idx_in_range,:] - lhee[frame_idx_in_range,:])
                    e_y0_lfoot = lank2[frame_idx_in_range,:] - lank[frame_idx_in_range,:]
                    e_x_lfoot = np.cross(e_z_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_y0_lfoot))
                    e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
                    rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

                    # 相対回転行列の計算
                    r_hip_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_pelvis)  #骨盤節に合わせるための大腿節の回転行列
                    l_hip_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_pelvis)
                    r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)  #大腿節に合わせるための下腿節の回転行列
                    l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
                    r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)  #足節に合わせるための下腿節の回転行列
                    l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)

                    r_hip_angle_rot = R.from_matrix(r_hip_realative_rotation)
                    l_hip_angle_rot = R.from_matrix(l_hip_realative_rotation)
                    r_knee_angle_rot = R.from_matrix(r_knee_realative_rotation)
                    l_knee_angle_rot = R.from_matrix(l_knee_realative_rotation)
                    r_ankle_angle_rot = R.from_matrix(r_ankle_realative_rotation)
                    l_ankle_angle_rot = R.from_matrix(l_ankle_realative_rotation)

                    # 回転行列から回転角を計算 XYZ大文字だと内因性，xyz小文字だと外因性
                    # 屈曲-伸展
                    r_hip_angle = r_hip_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_hip_angle = l_hip_angle_rot.as_euler('YZX', degrees=True)[0]
                    r_knee_angle = r_knee_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_knee_angle = l_knee_angle_rot.as_euler('YZX', degrees=True)[0]
                    r_ankle_angle = r_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_ankle_angle = l_ankle_angle_rot.as_euler('YZX', degrees=True)[0]

                    # 内旋外旋
                    r_hip_angle_inex = r_hip_angle_rot.as_euler('YZX', degrees=True)[1]
                    l_hip_angle_inex = l_hip_angle_rot.as_euler('YZX', degrees=True)[1]
                    r_knee_angle_inex = r_knee_angle_rot.as_euler('YZX', degrees=True)[1]
                    l_knee_angle_inex = l_knee_angle_rot.as_euler('YZX', degrees=True)[1]
                    r_ankle_angle_inex = r_ankle_angle_rot.as_euler('YZX', degrees=True)[1]
                    l_ankle_angle_inex = l_ankle_angle_rot.as_euler('YZX', degrees=True)[1]

                    # 内転外転
                    r_hip_angle_adab = r_hip_angle_rot.as_euler('YZX', degrees=True)[2]
                    l_hip_angle_adab = l_hip_angle_rot.as_euler('YZX', degrees=True)[2]
                    r_knee_angle_adab = r_knee_angle_rot.as_euler('YZX', degrees=True)[2]
                    l_knee_angle_adab = l_knee_angle_rot.as_euler('YZX', degrees=True)[2]
                    r_ankle_angle_adab = r_ankle_angle_rot.as_euler('YZX', degrees=True)[2]
                    l_ankle_angle_adab = l_ankle_angle_rot.as_euler('YZX', degrees=True)[2]

                    """
                    xyz小文字(外因性)で角度計算してた
                    """
                    # r_hip_angle = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
                    # l_hip_angle = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
                    # r_knee_angle = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
                    # l_knee_angle = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
                    # r_ankle_angle = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
                    # l_ankle_angle = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
                    
                    # # 内旋外旋
                    # r_hip_angle_inex = r_hip_angle_rot.as_euler('yzx', degrees=True)[1]
                    # l_hip_angle_inex = l_hip_angle_rot.as_euler('yzx', degrees=True)[1]
                    # r_knee_angle_inex = r_knee_angle_rot.as_euler('yzx', degrees=True)[1]
                    # l_knee_angle_inex = l_knee_angle_rot.as_euler('yzx', degrees=True)[1]
                    # r_ankle_angle_inex = r_ankle_angle_rot.as_euler('yzx', degrees=True)[1]
                    # l_ankle_angle_inex = l_ankle_angle_rot.as_euler('yzx', degrees=True)[1]
                    
                    # # 内転外転
                    # r_hip_angle_adab = r_hip_angle_rot.as_euler('yzx', degrees=True)[2]
                    # l_hip_angle_adab = l_hip_angle_rot.as_euler('yzx', degrees=True)[2]
                    # r_knee_angle_adab = r_knee_angle_rot.as_euler('yzx', degrees=True)[2]
                    # l_knee_angle_adab = l_knee_angle_rot.as_euler('yzx', degrees=True)[2]
                    # r_ankle_angle_adab = r_ankle_angle_rot.as_euler('yzx', degrees=True)[2]
                    # l_ankle_angle_adab = l_ankle_angle_rot.as_euler('yzx', degrees=True)[2]

                    angle_list_range.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle, 
                                             r_hip_angle_inex, l_hip_angle_inex, r_knee_angle_inex, l_knee_angle_inex, r_ankle_angle_inex, l_ankle_angle_inex,
                                             r_hip_angle_adab, l_hip_angle_adab, r_knee_angle_adab, l_knee_angle_adab, r_ankle_angle_adab, l_ankle_angle_adab])
                    # angle_list_range.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle])
                    
                    #骨盤と左足踵のベクトルを計算
                    pel2heel = lhee[frame_idx_in_range, :] - hip
                    pel2hee_list.append(pel2heel)


                    plot_flag = False
                    if plot_flag:
                        if original_frame_num == 0:
                            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            ax.set_zlabel("z")
                            ax.set_xlim(-1000, 1000)
                            ax.set_ylim(-1000, 1000)
                            ax.set_zlim(0, 2000)
                            #frame数を表示
                            ax.text2D(0.5, 0.01, f"frame = {original_frame_num}", transform=ax.transAxes)
                            #方向を設定
                            ax.view_init(elev=0, azim=0)

                            ax.scatter(rasi[frame_idx_in_range,:][0], rasi[frame_idx_in_range,:][1], rasi[frame_idx_in_range,:][2], label='rasi')
                            ax.scatter(lasi[frame_idx_in_range,:][0], lasi[frame_idx_in_range,:][1], lasi[frame_idx_in_range,:][2], label='lasi')
                            ax.scatter(rpsi[frame_idx_in_range,:][0], rpsi[frame_idx_in_range,:][1], rpsi[frame_idx_in_range,:][2], label='rpsi')
                            ax.scatter(lpsi[frame_idx_in_range,:][0], lpsi[frame_idx_in_range,:][1], lpsi[frame_idx_in_range,:][2], label='lpsi')
                            ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
                            ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
                            ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
                            ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
                            ax.scatter(rtoe[frame_idx_in_range,:][0], rtoe[frame_idx_in_range,:][1], rtoe[frame_idx_in_range,:][2], label='rtoe')
                            ax.scatter(ltoe[frame_idx_in_range,:][0], ltoe[frame_idx_in_range,:][1], ltoe[frame_idx_in_range,:][2], label='ltoe')
                            ax.scatter(rhee[frame_idx_in_range,:][0], rhee[frame_idx_in_range,:][1], rhee[frame_idx_in_range,:][2], label='rhee')
                            ax.scatter(lhee[frame_idx_in_range, :][0], lhee[frame_idx_in_range, :][1], lhee[frame_idx_in_range, :][2], label='lhee')
                            ax.scatter(lumbar[0], lumbar[1], lumbar[2], label='lumbar')
                            ax.scatter(hip[0], hip[1], hip[2], label='hip')
                            ax.scatter(rthigh[0], rthigh[1], rthigh[2], label='rthigh')
                            ax.scatter(lthigh[0], lthigh[1], lthigh[2], label='lthigh')
                            
                            # ax.scatter(hip_0[0], hip_0[1], hip_0[2], label='hip_0')
                            # ax.scatter(lthigh_pelvis[0], lthigh_pelvis[1], lthigh_pelvis[2], label='lthigh_0')
                            # ax.scatter(rthigh_pelvis[0], rthigh_pelvis[1], rthigh_pelvis[2], label='rthigh_0')
                            # ax.scatter((rthigh_pelvis[0] + lthigh_pelvis[0])/2, (rthigh_pelvis[1] + lthigh_pelvis[1])/2, (rthigh_pelvis[2] + lthigh_pelvis[2])/2, label='mid_thigh')
                            # ax.scatter(0, 0, 0, label='origin')


                            # ax.plot([lhee[frame_idx_in_range, :][0]- hip[0]], [lhee[frame_idx_in_range, :][1]- hip[1]], [lhee[frame_idx_in_range, :][2]- hip[2]], color='red')


                            plot_scale = 50
                            e_x_pelvis = e_x_pelvis * plot_scale
                            e_y_pelvis = e_y_pelvis * plot_scale
                            e_z_pelvis = e_z_pelvis * plot_scale
                            e_x_rthigh = e_x_rthigh * plot_scale
                            e_y_rthigh = e_y_rthigh * plot_scale
                            e_z_rthigh = e_z_rthigh * plot_scale
                            e_x_lthigh = e_x_lthigh * plot_scale
                            e_y_lthigh = e_y_lthigh * plot_scale
                            e_z_lthigh = e_z_lthigh * plot_scale
                            e_x_rshank = e_x_rshank * plot_scale
                            e_y_rshank = e_y_rshank * plot_scale
                            e_z_rshank = e_z_rshank * plot_scale
                            e_x_lshank = e_x_lshank * plot_scale
                            e_y_lshank = e_y_lshank * plot_scale
                            e_z_lshank = e_z_lshank * plot_scale
                            e_x_rfoot = e_x_rfoot * plot_scale
                            e_y_rfoot = e_y_rfoot * plot_scale
                            e_z_rfoot = e_z_rfoot * plot_scale
                            e_x_lfoot = e_x_lfoot * plot_scale
                            e_y_lfoot = e_y_lfoot * plot_scale
                            e_z_lfoot = e_z_lfoot * plot_scale
                            
                            e_x_pelvis_0 = e_x_pelvis_0 * plot_scale
                            e_y_pelvis_0 = e_y_pelvis_0 * plot_scale
                            e_z_pelvis_0 = e_z_pelvis_0 * plot_scale

                            ax.plot([hip[0], hip[0] + e_x_pelvis[0]], [hip[1], hip[1] + e_x_pelvis[1]], [hip[2], hip[2] + e_x_pelvis[2]], color='red')
                            ax.plot([hip[0], hip[0] + e_y_pelvis[0]], [hip[1], hip[1] + e_y_pelvis[1]], [hip[2], hip[2] + e_y_pelvis[2]], color='green')
                            ax.plot([hip[0], hip[0] + e_z_pelvis[0]], [hip[1], hip[1] + e_z_pelvis[1]], [hip[2], hip[2] + e_z_pelvis[2]], color='blue')

                            ax.plot([rthigh[0], rthigh[0] + e_x_rthigh[0]], [rthigh[1], rthigh[1] + e_x_rthigh[1]], [rthigh[2], rthigh[2] + e_x_rthigh[2]], color='red')
                            ax.plot([rthigh[0], rthigh[0] + e_y_rthigh[0]], [rthigh[1], rthigh[1] + e_y_rthigh[1]], [rthigh[2], rthigh[2] + e_y_rthigh[2]], color='green')
                            ax.plot([rthigh[0], rthigh[0] + e_z_rthigh[0]], [rthigh[1], rthigh[1] + e_z_rthigh[1]], [rthigh[2], rthigh[2] + e_z_rthigh[2]], color='blue')

                            ax.plot([lthigh[0], lthigh[0] + e_x_lthigh[0]], [lthigh[1], lthigh[1] + e_x_lthigh[1]], [lthigh[2], lthigh[2] + e_x_lthigh[2]], color='red')
                            ax.plot([lthigh[0], lthigh[0] + e_y_lthigh[0]], [lthigh[1], lthigh[1] + e_y_lthigh[1]], [lthigh[2], lthigh[2] + e_y_lthigh[2]], color='green')
                            ax.plot([lthigh[0], lthigh[0] + e_z_lthigh[0]], [lthigh[1], lthigh[1] + e_z_lthigh[1]], [lthigh[2], lthigh[2] + e_z_lthigh[2]], color='blue')

                            ax.plot([rshank[0], rshank[0] + e_x_rshank[0]], [rshank[1], rshank[1] + e_x_rshank[1]], [rshank[2], rshank[2] + e_x_rshank[2]], color='red')
                            ax.plot([rshank[0], rshank[0] + e_y_rshank[0]], [rshank[1], rshank[1] + e_y_rshank[1]], [rshank[2], rshank[2] + e_y_rshank[2]], color='green')
                            ax.plot([rshank[0], rshank[0] + e_z_rshank[0]], [rshank[1], rshank[1] + e_z_rshank[1]], [rshank[2], rshank[2] + e_z_rshank[2]], color='blue')

                            ax.plot([lshank[0], lshank[0] + e_x_lshank[0]], [lshank[1], lshank[1] + e_x_lshank[1]], [lshank[2], lshank[2] + e_x_lshank[2]], color='red')
                            ax.plot([lshank[0], lshank[0] + e_y_lshank[0]], [lshank[1], lshank[1] + e_y_lshank[1]], [lshank[2], lshank[2] + e_y_lshank[2]], color='green')
                            ax.plot([lshank[0], lshank[0] + e_z_lshank[0]], [lshank[1], lshank[1] + e_z_lshank[1]], [lshank[2], lshank[2] + e_z_lshank[2]], color='blue')

                            ax.plot([rfoot[0], rfoot[0] + e_x_rfoot[0]], [rfoot[1], rfoot[1] + e_x_rfoot[1]], [rfoot[2], rfoot[2] + e_x_rfoot[2]], color='red')
                            ax.plot([rfoot[0], rfoot[0] + e_y_rfoot[0]], [rfoot[1], rfoot[1] + e_y_rfoot[1]], [rfoot[2], rfoot[2] + e_y_rfoot[2]], color='green')
                            ax.plot([rfoot[0], rfoot[0] + e_z_rfoot[0]], [rfoot[1], rfoot[1] + e_z_rfoot[1]], [rfoot[2], rfoot[2] + e_z_rfoot[2]], color='blue')

                            ax.plot([lfoot[0], lfoot[0] + e_x_lfoot[0]], [lfoot[1], lfoot[1] + e_x_lfoot[1]], [lfoot[2], lfoot[2] + e_x_lfoot[2]], color='red')
                            ax.plot([lfoot[0], lfoot[0] + e_y_lfoot[0]], [lfoot[1], lfoot[1] + e_y_lfoot[1]], [lfoot[2], lfoot[2] + e_y_lfoot[2]], color='green')
                            ax.plot([lfoot[0], lfoot[0] + e_z_lfoot[0]], [lfoot[1], lfoot[1] + e_z_lfoot[1]], [lfoot[2], lfoot[2] + e_z_lfoot[2]], color='blue')
                            
                            # ax.plot([hip_0[0], hip_0[0] + e_x_pelvis_0[0]], [hip_0[1], hip_0[1] + e_x_pelvis_0[1]], [hip_0[2], hip_0[2] + e_x_pelvis_0[2]], color='red', linestyle='dashed')
                            # ax.plot([hip_0[0], hip_0[0] + e_y_pelvis_0[0]], [hip_0[1], hip_0[1] + e_y_pelvis_0[1]], [hip_0[2], hip_0[2] + e_y_pelvis_0[2]], color='green', linestyle='dashed')
                            # ax.plot([hip_0[0], hip_0[0] + e_z_pelvis_0[0]], [hip_0[1], hip_0[1] + e_z_pelvis_0[1]], [hip_0[2], hip_0[2] + e_z_pelvis_0[2]], color='blue', linestyle='dashed')

                            plt.legend()
                            plt.show()

 
                except Exception as e:
                    print(f"フレーム {original_frame_num} で予期せぬエラー: {e}。このフレームの角度をNaNとします。")
                    angle_list_range.append([np.nan] * 6) # 6つの角度すべてをNaNに

            # この範囲の角度データからデータフレームを作成
            if angle_list_range:
                angle_array_range = np.array(angle_list_range)
                # angle_df_range = pd.DataFrame(angle_array_range, columns=["R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle"], index=_range)
                angle_df_range = pd.DataFrame(angle_array_range, columns=["R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle",
                                                                           "R_Hip_InEx", "L_Hip_InEx", "R_Knee_InEx", "L_Knee_InEx", "R_Ankle_InEx", "L_Ankle_InEx",
                                                                           "R_Hip_AdAb", "L_Hip_AdAb", "R_Knee_AdAb", "L_Knee_AdAb", "R_Ankle_AdAb", "L_Ankle_AdAb"], index=_range)
                all_angle_dfs.append(angle_df_range)

        # すべての範囲の角度データフレームを結合
        if all_angle_dfs:
            angle_df = pd.concat(all_angle_dfs)
            # 元のdfのインデックスに合わせて振り直し、欠損部分はNaNで埋める
            angle_df = angle_df.reindex(full_df.index)
        else:
            print("警告: 有効な角度データが計算されませんでした。")
            continue
        
        
        # 角度データの連続性保つ
        for col in angle_df.columns:
            prev = None
            for i in angle_df.index:
                curr = angle_df.at[i, col]
                if prev is not None and pd.notna(curr) and pd.notna(prev):
                    diff = curr - prev
                    if diff > 180:
                        angle_df.at[i, col] = curr - 360
                    elif diff < -180:
                        angle_df.at[i, col] = curr + 360
                    prev = angle_df.at[i, col]
                elif pd.notna(curr):
                    prev = curr
                    
        # Hip, Knee, Ankle角度のオフセット補正
        if 'R_Hip' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Hip'] > 0:
                    angle_df.loc[frame, 'R_Hip'] = angle_df.at[frame, 'R_Hip'] - 180
                else:
                    angle_df.loc[frame, 'R_Hip'] = 180 + angle_df.at[frame, 'R_Hip']
        if 'L_Hip' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Hip'] > 0:
                    angle_df.loc[frame, 'L_Hip'] = angle_df.at[frame, 'L_Hip'] - 180
                else:
                    angle_df.loc[frame, 'L_Hip'] = 180 + angle_df.at[frame, 'L_Hip']
        if 'R_Knee' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Knee'] > 0:
                    angle_df.loc[frame, 'R_Knee'] = 180 - angle_df.at[frame, 'R_Knee']
                else:
                    angle_df.loc[frame, 'R_Knee'] = - (180 + angle_df.at[frame, 'R_Knee'])
        if 'L_Knee' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Knee'] > 0:
                    angle_df.loc[frame, 'L_Knee'] = 180 - angle_df.at[frame, 'L_Knee']
                else:
                    angle_df.loc[frame, 'L_Knee'] = - (180 + angle_df.at[frame, 'L_Knee'])
        if 'R_Ankle' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Ankle'] = 90 - angle_df.at[frame, 'R_Ankle']
        if 'L_Ankle' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Ankle'] = 90 - angle_df.at[frame, 'L_Ankle']
            
        # if 'R_Hip' in angle_df.columns:
        #     if angle_df['R_Hip'].loc[0] > 0:
        #         angle_df['R_Hip'] = angle_df['R_Hip'] - 180
        #     else:
        #         angle_df['R_Hip'] = 180 + angle_df['R_Hip']
        # if 'L_Hip' in angle_df.columns:
        #     if angle_df['L_Hip'].loc[0] > 0:
        #         angle_df['L_Hip'] = angle_df['L_Hip'] - 180
        #     else:
        #         angle_df['L_Hip'] = 180 + angle_df['L_Hip']
        # if 'R_Knee' in angle_df.columns:
        #     angle_df['R_Knee'] = 180 - angle_df['R_Knee']
        # if 'L_Knee' in angle_df.columns:
        #     angle_df['L_Knee'] = 180 - angle_df['L_Knee']
        # if 'R_Ankle' in angle_df.columns:
        #     angle_df['R_Ankle'] = 90 - angle_df['R_Ankle']
        # if 'L_Ankle' in angle_df.columns:
        #     angle_df['L_Ankle'] = 90 - angle_df['L_Ankle']

        angle_df.to_csv(tsv_file.with_name(f"{tsv_file.stem}_angles.csv"))
        print(f"角度データを保存しました: {tsv_file.with_name(f'{tsv_file.stem}_angles.csv')}")
        
        # 初期接地の算出
        pel2heel_array = np.array(pel2hee_list)
        if tsv_file.name == "sub1-0003_label.tsv":
            pel2heel = - pel2heel_array[:, 1] #y軸負方向が進行方向
        if tsv_file.name == "sub4_com_nfpa0001.tsv" or tsv_file.name == "sub4_com_nfpa0001_deleted.tsv":
            pel2heel = - pel2heel_array[:, 0] #x軸負方向が進行方向
        p2h_df = pd.DataFrame({"pel2heel": pel2heel})
        p2h_df = p2h_df.reindex(full_df.index)  # 元のdfのインデックスに合わせて振り直し、欠損部分はNaNで埋める
        
       #初期接地の算出（左足）
        p2h_df.sort_values(by="pel2heel", ascending=False, inplace=True)
        cand_ic_frame = p2h_df.index[:60]

        ic_frame_list = []
        skip_frame = set()
        for frame in cand_ic_frame:
            if frame in skip_frame:
                continue
            ic_frame_list.append(frame)
            skip_frame.update(range(frame-10, frame+10)) # 10フレーム前後をスキップ
        ic_frame_list = sorted(ic_frame_list)
        # print(f"Candidate IC frames: {cand_ic_frame}")
        print(f"IC frames: {ic_frame_list}")
        
        # 左右股関節の屈曲伸展角度をプロット
        plt.plot(angle_df['L_Hip'], label='Left Hip Flexion/Extension', color='orange')
        plt.plot(angle_df['R_Hip'], label='Right Hip Flexion/Extension', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Flexion/Extension Angles Over Time')
        # plt.ylim(-40, 40)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Hip_L_Flexion_Extension.png"))
        plt.close()
        
        # 左右膝関節の屈曲伸展角度をプロット
        plt.plot(angle_df['L_Knee'], label='Left Knee Flexion/Extension', color='orange')
        plt.plot(angle_df['R_Knee'], label='Right Knee Flexion/Extension', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Flexion/Extension Angles Over Time')
        # plt.ylim(-10, 70)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Knee_L_Flexion_Extension.png"))
        plt.close()

        # 左右足関節の底屈背屈角度をプロット
        plt.plot(angle_df['L_Ankle'], label='Left Ankle Plantarflexion/Dorsiflexion', color='orange')
        plt.plot(angle_df['R_Ankle'], label='Right Ankle Plantarflexion/Dorsiflexion', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Ankle Plantarflexion/Dorsiflexion Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Ankle_L_Plantarflexion_Dorsiflexion.png"))
        plt.close()
        
        
        # 左右股関節の内転外転角度をプロット
        plt.plot(angle_df['L_Hip_AdAb'], label='Left Hip Adduction/Abduction', color='orange')
        plt.plot(angle_df['R_Hip_AdAb'], label='Right Hip Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Hip_Adduction_Abduction.png"))
        plt.close()
        
        # 左右膝関節の内転外転角度をプロット
        plt.plot(angle_df['L_Knee_AdAb'], label='Left Knee Adduction/Abduction', color='orange')
        plt.plot(angle_df['R_Knee_AdAb'], label='Right Knee Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Knee_Adduction_Abduction.png"))
        plt.close()
        
        # 左右足関節の内転外転角度をプロット
        plt.plot(angle_df['L_Ankle_AdAb'], label='Left Ankle Adduction/Abduction', color='orange')
        plt.plot(angle_df['R_Ankle_AdAb'], label='Right Ankle Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Ankle Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Ankle_Adduction_Abduction.png"))
        plt.close()
        
        
        # 左右股関節の内旋外旋角度をプロット
        plt.plot(angle_df['L_Hip_InEx'], label='Left Hip Internal/External Rotation', color='orange')
        plt.plot(angle_df['R_Hip_InEx'], label='Right Hip Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Hip_Internal_External_Rotation.png"))
        plt.close()
        
        # 左右膝関節の内旋外旋角度をプロット
        plt.plot(angle_df['L_Knee_InEx'], label='Left Knee Internal/External Rotation', color='orange')
        plt.plot(angle_df['R_Knee_InEx'], label='Right Knee Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Knee_Internal_External_Rotation.png"))
        plt.close()
        
        # 左右足関節の内旋外旋角度をプロット
        plt.plot(angle_df['L_Ankle_InEx'], label='Left Ankle Internal/External Rotation', color='orange')
        plt.plot(angle_df['R_Ankle_InEx'], label='Right Ankle Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in ic_frame_list]
        plt.xlabel('Frame')     
        plt.ylabel('Angle (degrees)')
        plt.title('Ankle Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Ankle_Internal_External_Rotation.png"))
        plt.close()
        
    #     angle_dict[tsv_file.stem] = angle_df
        
    # print(f"angle_dict: {angle_dict}")
    # # 各被験者の角度データをプロット，RMSEを計算
    # if len(angle_dict) == 2:
        
    #     def ref_rmse(angle_dict):
    #         keys = list(angle_dict.keys())
    #         df1 = angle_dict[keys[0]]
    #         df2 = angle_dict[keys[1]]
            
    #         # 共通のフレーム範囲を特定
    #         common_index = df1.index.intersection(df2.index)
    #         df1_common = df1.loc[common_index]
    #         df2_common = df2.loc[common_index]
            
    #         joints = ['Hip', 'Knee', 'Ankle']
    #         sides = ['L', 'R']
            
    #         for joint in joints:
    #             plt.figure(figsize=(10, 6))
    #             for side in sides:
    #                 if side == "L":
    #                     color = "tab:orange"
    #                 else:
    #                     color = "tab:blue"
    #                 col_name = f"{side}_{joint}"
    #                 plt.plot(df1_common.index, df1_common[col_name], label=f'Original', color=color, linestyle='-', linewidth=2)
    #                 plt.plot(df2_common.index, df2_common[col_name], label=f'Rigid Transformation', color=color, linewidth=1, alpha=0.1)
    #                 # plt.plot(df1_common.index, df1_common[col_name], label=f'Original', color=color)
    #                 # plt.plot(df2_common.index, df2_common[col_name], label=f'Rigid Transformation', linestyle='--', color=color)

    #                 # RMSEの計算
    #                 rmse = np.sqrt(np.nanmean((df1_common[col_name] - df2_common[col_name]) ** 2))
    #                 print(f"RMSE for {col_name}: {rmse:.2f} degrees")
                    
    #             plt.xlabel('Frame')
    #             plt.ylabel('Angle (degrees)')
    #             plt.title(f'{joint} Flexion/Extension Angles Comparison')
    #             plt.ylim(-50, 100)
    #             plt.grid()
    #             plt.legend()
    #             save_path = tsv_dir / f"{keys[0]}_{keys[1]}_{joint}_Comparison.png"
    #             plt.savefig(save_path)
    #             plt.close()
    #     ref_rmse(angle_dict)  # 関節角度のRMSEを計算,プロット


if __name__ == "__main__":
    main()
