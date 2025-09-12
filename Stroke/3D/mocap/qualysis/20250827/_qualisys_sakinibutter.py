"""
元データのZ座標が汚いから先にバターワースフィルタで平滑化してから補間を試したけど、あんまり変わらなかったから使うのやめた
"""


import module_mocap as moc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as mticker


tsv_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm")
tsv_files = tsv_dir.glob("*0004*.tsv")
tpose_path = tsv_dir / "sub1-0001_ref_pos.json"

def plot_interpolation_results(dfs, labels, marker_name, output_path):
    """
    指定されたマーカーの補間結果をプロットして保存する。
    補間ステップ2と3で新たに埋められた点を強調表示する。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
    coords = ['X', 'Y', 'Z']

    original_df, step1_df, step1_2_df, step2_df, step3_df = dfs

    for i, coord in enumerate(coords):
        ax = axes[i]
        col_name = f'{marker_name} {coord}'

        if col_name not in original_df.columns:
            print(f"警告: マーカー '{marker_name}' の列が見つかりません。プロットをスキップします。")
            plt.close(fig)
            return

        # 1. 元のデータ(欠損あり)を点でプロット
        ax.plot(original_df.index, original_df[col_name], 'o', color='gray', label=labels[0], markersize=3, alpha=0.6)

        # 2. ステップ1の補間結果を基準線としてプロット
        newly_filled_step11 = step1_df[original_df[col_name].isna() & step1_df[col_name].notna()]
        if not newly_filled_step11.empty:
            ax.plot(newly_filled_step11.index, newly_filled_step11[col_name], 'o', color='orange',
                    label=f'{labels[1]} (newly filled)', markersize=3)
            
        # ステップ1_2の補間結果を基準線としてプロット
        ax.plot(step1_2_df.index, step1_2_df[col_name], '-', color='black', label=labels[1] + ' (butterworth)', alpha=0.2)

        # 3. ステップ2で「新たに」補間された点のみを抽出してプロット
        newly_filled_step2 = step2_df[step1_df[col_name].isna() & step2_df[col_name].notna()]
        if not newly_filled_step2.empty:
            ax.plot(newly_filled_step2.index, newly_filled_step2[col_name], 'o', color='blue',
                    label=f'{labels[2]} (newly filled)', markersize=3)

        # 4. ステップ3で「新たに」補間された点のみを抽出してプロット
        newly_filled_step3 = step3_df[step2_df[col_name].isna() & step3_df[col_name].notna()]
        if not newly_filled_step3.empty:
            ax.plot(newly_filled_step3.index, newly_filled_step3[col_name], 'o', color='red',
                    label=f'{labels[3]} (newly filled)', markersize=3)

        ax.set_title(f'{marker_name} - {coord} coordinate', fontsize=14)
        ax.set_ylabel('Position (mm)')
        ax.legend(fontsize=12)

        # 20フレームごとに薄い縦のグリッド線を追加
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.8, color='lightgray')


    axes[-1].set_xlabel('Frame')
    fig.suptitle(f'Interpolation Step-by-Step Check for {marker_name}', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.savefig(output_path)
    plt.close(fig)
    print(f"プロット画像を保存しました: '{output_path.name}'")

def main():
    for itsv, tsv_file in enumerate(tsv_files):
        print(f"Processing {itsv+1}/{len(list(tsv_dir.glob('*.tsv')))}: {tsv_file.name}")
        full_df = moc.read_tsv(tsv_file)  #tsvファイルの読み込み
        target_df = full_df.copy()
        # print(f"target_df: {target_df}")
        nan_df = target_df.replace(0, np.nan)  #0をNaNに置き換え
        # 補間処理1 欠損が20フレーム以下の区間をスプライン補間
        interpolated1_df = moc.interpolate_short_gaps(nan_df, max_gap_size=20, method='spline', order=3)
        
        
        
        # 補間処理1_2 欠損がない範囲をbutterworthフィルタで平滑化
        interpolated1_2_df = moc.butterworth_filter_no_nan_gaps(interpolated1_df, cutoff=6, order=4, fs=120) #4次のバターワースローパスフィルタ
        
        
        # 補間処理2 参考点が3点以上ある骨盤の座標を補間
        target_markers_to_fill = ['RASI', 'LASI', 'RPSI', 'LPSI']  # 補間したいターゲット
        interpolated2_df = moc.interpolate_pelvis_rigid_body(interpolated1_2_df, tpose_path, target_markers_to_fill)
        # 補間処理3 骨盤補間後に欠損が20フレーム以下の区間を再度スプライン補間
        interpolated3_df = moc.interpolate_short_gaps(interpolated2_df, max_gap_size=20, method='spline', order=3)

        interpolated3_df.to_csv(tsv_file.with_name(f"{tsv_file.stem}_interpolated.csv"))











        print("\n--- 補間結果のプロットを開始 ---")
        plot_dfs = [nan_df, interpolated1_df, interpolated1_2_df, interpolated2_df, interpolated3_df]
        plot_labels = ['Original Data (with Gaps)', 'Step 1: Spline Interpolation', 'Step 1_2: Butterworth Filter', 'Step 2: Rigid Body Fitting', 'Step 3: Final Spline Interpolation']

        # プロットしたいマーカーをリストで指定
        markers_to_plot = ['RASI', 'LASI', 'RPSI', 'LPSI']

        for marker in markers_to_plot:
            if f'{marker} X' in full_df.columns:
                output_filename = tsv_dir / f"{tsv_file.stem}_{marker}_interpolation_check.png"
                plot_interpolation_results(plot_dfs, plot_labels, marker, output_filename)
            else:
                print(f"マーカー '{marker}' はファイルに存在しないため、プロットをスキップします。")
        print("--- 補間結果のプロットが完了 ---\n")



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




        # # フィルタ処理 バターワースローパスフィルタ
        # butter_df = interpolated3_df.copy()
        # butter_df = moc.butterworth_filter(butter_df, cutoff=6, order=4, fs=120) #4次のバターワースローパスフィルタ

        # butter_df.to_csv(tsv_file.with_name(f"{tsv_file.stem}_interpolated_filtered.csv"))



        # 各有効範囲の角度計算結果を保存するリスト
        all_angle_dfs = []

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

            # ループの範囲を現在の有効範囲 '_range' に限定
            for frame_idx_in_range, original_frame_num in enumerate(_range):
                try:
                    # 座標系定義の計算 (インデックスは範囲内でのインデックス frame_idx_in_range を使用)
                    # print(f"Frame {frame}:")
                    d_asi = np.linalg.norm(rasi[frame_idx_in_range,:] - lasi[frame_idx_in_range,:])
                    d_leg = (np.linalg.norm(rank[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]) + np.linalg.norm(lank[frame_idx_in_range, :] - lasi[frame_idx_in_range,:]) / 2)
                    r = 0.012 #使用したマーカー径

                    h = 1.76 #被験者(kasa)の身長

                    k = h/1.7
                    beta = 0.1 * np.pi #[rad]
                    theta = 0.496 #[rad]
                    c = 0.115 * d_leg - 0.00153
                    x_dis = 0.1288 * d_leg - 0.04856

                    # skycom + davis
                    x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    y_rthigh = +(c * np.sin(theta) - d_asi/2)
                    y_lthigh = -(c * np.sin(theta)- d_asi/2)
                    z_rthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
                    z_lthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
                    rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
                    lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

                    hip_0 = (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2
                    lumbar = (0.47 * (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2 + 0.53 * (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

                    #骨盤節座標系（原点はhip）
                    e_y0_pelvis = lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]
                    e_z_pelvis = (lumbar - hip_0)/np.linalg.norm(lumbar - hip_0)
                    e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
                    e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
                    rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

                    transformation_matrix = np.array([[e_x_pelvis[0], e_y_pelvis[0], e_z_pelvis[0], hip_0[0]],
                                                        [e_x_pelvis[1], e_y_pelvis[1], e_z_pelvis[1], hip_0[1]],
                                                        [e_x_pelvis[2], e_y_pelvis[2], e_z_pelvis[2], hip_0[2]],
                                                        [0,       0,       0,       1]])

                    #モーキャプの座標系に変換してもう一度計算
                    rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
                    lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
                    hip = (rthigh + lthigh) / 2

                    e_y0_pelvis = lthigh - rthigh
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
                    r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
                    l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
                    r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)
                    l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
                    r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
                    l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)

                    r_hip_angle_rot = R.from_matrix(r_hip_realative_rotation)
                    l_hip_angle_rot = R.from_matrix(l_hip_realative_rotation)
                    r_knee_angle_rot = R.from_matrix(r_knee_realative_rotation)
                    l_knee_angle_rot = R.from_matrix(l_knee_realative_rotation)
                    r_ankle_angle_rot = R.from_matrix(r_ankle_realative_rotation)
                    l_ankle_angle_rot = R.from_matrix(l_ankle_realative_rotation)

                    # 回転行列から回転角を計算
                    r_hip_angle = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
                    l_hip_angle = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
                    r_knee_angle = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
                    l_knee_angle = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
                    r_ankle_angle = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
                    l_ankle_angle = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]

                    # 角度範囲を調整
                    # 角度が負の場合は360を足して正の値に変換
                    r_hip_angle = 360 + r_hip_angle if r_hip_angle < 0 else r_hip_angle
                    l_hip_angle = 360 + l_hip_angle if l_hip_angle < 0 else l_hip_angle
                    r_knee_angle = 360 + r_knee_angle if r_knee_angle < 0 else r_knee_angle
                    l_knee_angle = 360 + l_knee_angle if l_knee_angle < 0 else l_knee_angle
                    r_ankle_angle = 360 + r_ankle_angle if r_ankle_angle < 0 else r_ankle_angle
                    l_ankle_angle = 360 + l_ankle_angle if l_ankle_angle < 0 else l_ankle_angle

                    # 各角度について特定の範囲に変換
                    r_hip_angle = 180 - r_hip_angle if r_hip_angle > 100 else r_hip_angle
                    l_hip_angle = 180 - l_hip_angle if l_hip_angle > 100 else l_hip_angle
                    r_knee_angle = 180 - r_knee_angle if r_knee_angle < 180 else r_knee_angle - 180
                    l_knee_angle = 180 - l_knee_angle if l_knee_angle < 180 else l_knee_angle - 180
                    r_ankle_angle = 90 - r_ankle_angle if r_ankle_angle < 180 else 270 - r_ankle_angle
                    l_ankle_angle = 90 - l_ankle_angle if l_ankle_angle < 180 else 270 - l_ankle_angle

                    angle_list_range.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle])

                except Exception as e:
                    print(f"フレーム {original_frame_num} で予期せぬエラー: {e}。このフレームの角度をNaNとします。")
                    angle_list_range.append([np.nan] * 6) # 6つの角度すべてをNaNに

            # この範囲の角度データからデータフレームを作成
            if angle_list_range:
                angle_array_range = np.array(angle_list_range)
                angle_df_range = pd.DataFrame(angle_array_range, columns=["R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle"], index=_range)
                all_angle_dfs.append(angle_df_range)

        # すべての範囲の角度データフレームを結合
        if all_angle_dfs:
            angle_df = pd.concat(all_angle_dfs)
            # 元のdfのインデックスに合わせて振り直し、欠損部分はNaNで埋める
            angle_df = angle_df.reindex(full_df.index)
        else:
            print("警告: 有効な角度データが計算されませんでした。")
            continue

        angle_df.to_csv(tsv_file.with_name(f"{tsv_file.stem}_angles.csv"))
        print(f"角度データを保存しました: {tsv_file.with_name(f'{tsv_file.stem}_angles.csv')}")
        
        
        # 左右股関節の屈曲伸展角度をプロット
        plt.plot(angle_df['R_Hip'], label='Right Hip Flexion/Extension', color='blue')
        plt.plot(angle_df['L_Hip'], label='Left Hip Flexion/Extension', color='orange')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Flexion/Extension Angles Over Time')
        plt.legend()
        # plt.ylim(-50, 100)
        plt.show()
        plt.close()
        
        
        # 左右膝関節の屈曲伸展角度をプロット
        plt.plot(angle_df['R_Knee'], label='Right Knee Flexion/Extension', color='blue')
        plt.plot(angle_df['L_Knee'], label='Left Knee Flexion/Extension', color='orange')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Flexion/Extension Angles Over Time')
        plt.legend()
        # plt.ylim(-10, 150)
        plt.show()
        plt.close()
        
        # 左右足関節の底屈背屈角度をプロット
        plt.plot(angle_df['R_Ankle'], label='Right Ankle Plantarflexion/Dorsiflexion', color='blue')
        plt.plot(angle_df['L_Ankle'], label='Left Ankle Plantarflexion/Dorsiflexion', color='orange')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Ankle Plantarflexion/Dorsiflexion Angles Over Time')
        plt.legend()
        # plt.ylim(-50, 50)
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
