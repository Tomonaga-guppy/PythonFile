import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import traceback

# COCO-25フォーマットのキーポイント名とインデックス
KEYPOINTS_MAP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

def load_3d_data_from_json(json_path, person_key='person_1'):
    """
    指定されたJSONファイルから、特定の人物の3Dキーポイントデータを読み込む。
    'raw_processed'と'final'の両方のデータを読み込む。

    Args:
        json_path (Path): 3D結果が含まれるJSONファイルのパス。
        person_key (str): 読み込む人物のキー (例: 'person_1')。

    Returns:
        dict: 'raw_processed'と'final'のデータをNumPy配列で格納した辞書。
              データが見つからない場合はNoneを返す。
    """
    try:
        with open(json_path, 'r') as f:
            all_frames_data = json.load(f)

        data_sets = {'raw_processed': [], 'final': []}
        data_keys = {
            'raw_processed': 'points_3d_raw_processed',
            'final': 'points_3d_final'
        }

        for frame_data in all_frames_data:
            if person_key in frame_data:
                for set_name, data_key in data_keys.items():
                    points = frame_data[person_key].get(data_key)
                    if points:
                        data_sets[set_name].append(np.array(points))
                    else:
                        # データがないフレームはNaNで埋める
                        data_sets[set_name].append(np.full((25, 3), np.nan))

        # リストが空でないことを確認
        if not data_sets['raw_processed'] or not data_sets['final']:
            print(f"警告: {person_key} のデータが見つかりませんでした。")
            return None

        return {
            'raw_processed': np.array(data_sets['raw_processed']),
            'final': np.array(data_sets['final'])
        }

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {json_path}")
        return None
    except Exception as e:
        print(f"エラー: JSONファイルの読み込み中に問題が発生しました: {e}")
        traceback.print_exc()
        return None


def find_initial_contact_3d(points_3d, prominence=10, distance=30):
    """
    3Dキーポイントデータから踵接地（Initial Contact）のタイミングを検出する。
    骨盤と踵の進行方向（Z軸）の距離が極大になる点をICとする。

    Args:
        points_3d (np.ndarray): 3Dキーポイントデータ (frames, 25, 3)。
        prominence (float): ピークの際立ちの閾値。
        distance (int): ピーク間の最小距離（フレーム数）。

    Returns:
        dict: 左右の足のICフレーム番号のリストを格納した辞書。
    """
    ic_frames = {'R': [], 'L': []}
    sides = {'R': 'RHeel', 'L': 'LHeel'}

    midhip_z = points_3d[:, KEYPOINTS_MAP['MidHip'], 2]
    relative_dist_dict = {'R': [], 'L': []}

    for side, heel_name in sides.items():
        heel_z = points_3d[:, KEYPOINTS_MAP[heel_name], 2]

        # 骨盤に対する踵の相対的な前方位置
        relative_dist = heel_z - midhip_z
        relative_dist_dict[side] = relative_dist

        # 距離が極大になる点を検出
        peaks, _ = find_peaks(relative_dist, prominence=prominence, distance=distance)
        ic_frames[side] = peaks.tolist()



    return ic_frames, relative_dist_dict


def calc_joint_angles_3d(points_3d):
    """
    3D座標から関節角度を計算する（3Dベクトル間の角度を直接計算）

    Args:
        points_3d (np.ndarray): 3Dキーポイントデータ (frames, 25, 3)。

    Returns:
        pd.DataFrame: 計算された関節角度（度数）
    """

    def calculate_3d_vector_angle(v1, v2):
        """
        2つの3Dベクトル間の角度をarctan2で計算

        Args:
            v1: 第1ベクトル (frames, 3)
            v2: 第2ベクトル (frames, 3)

        Returns:
            np.ndarray: ベクトル間の角度（度数）
        """
        # ベクトルの長さを計算
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)

        # 正規化（0除算を避ける）
        v1_unit = np.divide(v1, v1_norm, out=np.zeros_like(v1), where=v1_norm!=0)
        v2_unit = np.divide(v2, v2_norm, out=np.zeros_like(v2), where=v2_norm!=0)

        # 内積を計算
        dot_product = np.sum(v1_unit * v2_unit, axis=1)

        # 外積を計算（大きさ）
        cross_product = np.cross(v1_unit, v2_unit)
        cross_magnitude = np.linalg.norm(cross_product, axis=1)

        # arctan2を使って角度を計算（符号付き）
        angle_rad = np.arctan2(cross_magnitude, dot_product)

        # 度数に変換
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_signed_3d_vector_angle(v1, v2, reference_normal=None):
        """
        2つの3Dベクトル間の符号付き角度を計算

        Args:
            v1: 第1ベクトル (frames, 3) - 近位セグメント
            v2: 第2ベクトル (frames, 3) - 遠位セグメント
            reference_normal: 参照法線ベクトル（屈曲方向の判定用）

        Returns:
            np.ndarray: 符号付き角度（度数）
        """
        # ベクトルの長さを計算
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)

        # 正規化
        v1_unit = np.divide(v1, v1_norm, out=np.zeros_like(v1), where=v1_norm!=0)
        v2_unit = np.divide(v2, v2_norm, out=np.zeros_like(v2), where=v2_norm!=0)

        # 内積を計算
        dot_product = np.sum(v1_unit * v2_unit, axis=1)

        # 外積を計算
        cross_product = np.cross(v1_unit, v2_unit)
        cross_magnitude = np.linalg.norm(cross_product, axis=1)

        # 角度を計算
        angle_rad = np.arctan2(cross_magnitude, dot_product)

        # 符号の判定（参照法線との内積で決定）
        if reference_normal is not None:
            # 外積と参照法線の内積で符号を決定
            sign = np.sum(cross_product * reference_normal, axis=1)
            sign = np.sign(sign)
            angle_rad = angle_rad * sign

        # 度数に変換
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_joint_flexion_extension(proximal_point, joint_point, distal_point):
        """
        関節の屈曲-伸展角度を3Dベクトル角度として計算

        Args:
            proximal_point: 近位点の座標 (frames, 3)
            joint_point: 関節点の座標 (frames, 3)
            distal_point: 遠位点の座標 (frames, 3)

        Returns:
            np.ndarray: 屈曲-伸展角度（度数）
        """
        # セグメントベクトルを計算
        proximal_segment = joint_point - proximal_point  # 近位セグメント（関節へ向かう）
        distal_segment = distal_point - joint_point      # 遠位セグメント（関節から離れる）

        # 2つのセグメント間の角度を計算
        angle = calculate_3d_vector_angle(proximal_segment, distal_segment)

        # 180度から引いて関節角度にする（真っ直ぐが180度、屈曲すると角度が小さくなる）
        joint_angle = 180.0 - angle

        return joint_angle

    def calculate_hip_angle_with_trunk_3d(trunk_point1, trunk_point2, hip_point, knee_point):
        """
        体幹基準での股関節角度を3Dベクトル角度として計算

        Args:
            trunk_point1, trunk_point2: 体幹の2点 (frames, 3)
            hip_point: 股関節点 (frames, 3)
            knee_point: 膝関節点 (frames, 3)

        Returns:
            np.ndarray: 股関節角度（度数）
        """
        # 体幹ベクトル
        trunk_vector = trunk_point2 - trunk_point1

        # 大腿ベクトル
        thigh_vector = knee_point - hip_point

        # 2つのベクトル間の角度を計算
        angle = calculate_3d_vector_angle(trunk_vector, thigh_vector)

        return angle

    # 全3D座標を使用
    p = points_3d  # (frames, 25, 3)

    # 股関節の屈曲伸展角度
    trunk_vector = p[:, KEYPOINTS_MAP['MidHip']] - p[:, KEYPOINTS_MAP['Neck']]
    l_thigh_vector = p[:, KEYPOINTS_MAP['LKnee']] - p[:, KEYPOINTS_MAP['LHip']]
    l_hip_flexion_angle = calculate_3d_vector_angle(l_thigh_vector, trunk_vector)
    l_hip_flexion_angle = l_hip_flexion_angle - 180

    r_thigh_vector = p[:, KEYPOINTS_MAP['RKnee']] - p[:, KEYPOINTS_MAP['RHip']]
    r_hip_flexion_angle = calculate_3d_vector_angle(r_thigh_vector, trunk_vector)
    r_hip_flexion_angle = r_hip_flexion_angle - 180

    # 膝関節の屈曲伸展角度
    l_shank_vector = p[:, KEYPOINTS_MAP['LAnkle']] - p[:, KEYPOINTS_MAP['LKnee']]
    l_knee_flexion_angle = calculate_3d_vector_angle(l_thigh_vector, l_shank_vector)
    l_knee_flexion_angle = l_knee_flexion_angle - 180

    r_shank_vector = p[:, KEYPOINTS_MAP['RAnkle']] - p[:, KEYPOINTS_MAP['RKnee']]
    r_knee_flexion_angle = calculate_3d_vector_angle(r_thigh_vector, r_shank_vector)
    r_knee_flexion_angle = r_knee_flexion_angle - 180

    # 足関節の背屈底屈角度
    l_foot_vector = (p[:, KEYPOINTS_MAP['LBigToe']] + p[:, KEYPOINTS_MAP['LSmallToe']]) - p[:, KEYPOINTS_MAP['LAnkle']]
    l_ankle_dorsiflexion_angle = calculate_3d_vector_angle(l_shank_vector, l_foot_vector)
    l_ankle_dorsiflexion_angle = l_ankle_dorsiflexion_angle - 90

    r_foot_vector = (p[:, KEYPOINTS_MAP['RBigToe']] + p[:, KEYPOINTS_MAP['RSmallToe']]) - p[:, KEYPOINTS_MAP['RAnkle']]
    r_ankle_dorsiflexion_angle = calculate_3d_vector_angle(r_shank_vector, r_foot_vector)
    r_ankle_dorsiflexion_angle = r_ankle_dorsiflexion_angle - 90

    # 股関節の外転内転角度
    l_hip_abduction_angle = calculate_3d_vector_angle(l_thigh_vector, trunk_vector)


    # 結果をデータフレームに格納
    angles_dict = {
        # 主要関節角度（体幹基準・関節角度）
        'L_Hip_Flexion': l_hip_flexion_angle,
        'R_Hip_Flexion': r_hip_flexion_angle,
        'L_Knee_Flexion': l_knee_flexion_angle,
        'R_Knee_Flexion': r_knee_flexion_angle,
        'L_Ankle_Dorsiflexion': l_ankle_dorsiflexion_angle,
        'R_Ankle_Dorsiflexion': r_ankle_dorsiflexion_angle,
    }

    return pd.DataFrame(angles_dict)


def plot_gait_angles_comparison(angles_dict, ic_frames_dict, start_frame, end_frame, title, save_path):
    """
    raw_processedとfinalの歩行角度を同じグラフで比較プロットする

    Args:
        angles_dict: {'raw_processed': angles_df1, 'final': angles_df2}
        ic_frames_dict: {'raw_processed': ic_frames1, 'final': ic_frames2}
        start_frame, end_frame: 表示範囲
        title: グラフタイトル
        save_path: 保存パス
    """

def plot_gait_angles_comparison(angles_dict, ic_frames_dict, start_frame, end_frame, title, save_path):
    """
    raw_processedとfinalの歩行角度を同じグラフで比較プロットする

    Args:
        angles_dict: {'raw_processed': angles_df1, 'final': angles_df2}
        ic_frames_dict: {'raw_processed': ic_frames1, 'final': ic_frames2}
        start_frame, end_frame: 表示範囲
        title: グラフタイトル
        save_path: 保存パス
    """
    # デバッグ: 利用可能なカラムを確認
    for data_type, angles_df in angles_dict.items():
        print(f"Available columns for {data_type}: {list(angles_df.columns)}")

    joint_names = ['Hip', 'Knee', 'Ankle']
    sides = ['L', 'R']

    # 色設定: raw_processedは薄い色、finalは濃い色
    colors = {
        'raw_processed': {'L': 'lightblue', 'R': 'lightcoral'},
        'final': {'L': 'blue', 'R': 'red'}
    }

    # ラインスタイル設定
    line_styles = {
        'raw_processed': '--',  # 破線
        'final': '-'           # 実線
    }

    ic_colors = {'L': 'lightblue', 'R': 'lightcoral'}

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(15, 12), sharex=True)
    if len(joint_names) == 1:
        axes = [axes]  # 単一プロットの場合のための調整

    for i, joint in enumerate(joint_names):
        ax = axes[i]

        # 各データタイプ（raw_processed, final）をプロット
        for data_type, angles_df in angles_dict.items():
            for side in sides:
                # 正しい列名の形式に修正
                if joint == 'Hip':
                    col_name = f"{side}_Hip_Flexion"
                elif joint == 'Knee':
                    col_name = f"{side}_Knee_Flexion"
                elif joint == 'Ankle':
                    col_name = f"{side}_Ankle_Dorsiflexion"
                else:
                    col_name = f"{side}_{joint}"

                # カラムが存在するかチェック
                if col_name in angles_df.columns:
                    label = f"{col_name}_{data_type}"
                    ax.plot(angles_df.index, angles_df[col_name],
                           label=label,
                           color=colors[data_type][side],
                           linestyle=line_styles[data_type],
                           alpha=0.7 if data_type == 'raw_processed' else 1.0,
                           lw=2)
                else:
                    print(f"Warning: Column '{col_name}' not found in {data_type} DataFrame")

        # IC（初期接触）の縦線を描画（finalデータのICを使用）
        if 'final' in ic_frames_dict:
            ic_frames = ic_frames_dict['final']
            # ic_framesをstart_frameとend_frameの範囲内に制限
            filtered_ic_frames = {}
            for side, frames in ic_frames.items():
                filtered_frames = [frame for frame in frames if start_frame <= frame <= end_frame]
                filtered_ic_frames[side] = filtered_frames

            for side_ic, frames in filtered_ic_frames.items():
                for frame in frames:
                    ax.axvline(x=frame, color=ic_colors[side_ic], linestyle=':', alpha=0.7,
                              label=f'IC_{side_ic}' if frame == frames[0] else "")

        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{joint} Angle (Comparison)')
        ax.grid(True, linestyle=':', alpha=0.6)

        # 凡例の重複を除去
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

    # X軸ラベルは最下段のみ
    axes[-1].set_xlabel('Frame Number')

    plt.tight_layout()
    plt.suptitle(title, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """メイン処理"""
    # --- 設定 ---
    # 解析したいデータが含まれるディレクトリを指定してください
    base_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera0-2")

    # try_1_spline.pyが出力したJSONファイルを指定
    json_file = base_dir / "3d_gait_analysis_spline_v1" / "thera0-2_3d_results_spline.json"
    subject_name = base_dir.parent.name
    thera_name = base_dir.name

    # 結果を出力するディレクトリ
    output_dir = base_dir / "gait_angle_analysis_from_3d"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 解析対象の人物
    person_key = 'person_1'

    if base_dir.parent.name == "sub1" and base_dir.name == "thera0-2":
        analysis_start_frame = 0
        analysis_end_frame = 400  # 必要に応じて終了フレームを設定
    elif base_dir.parent.name == "sub1" and base_dir.name == "thera1-1":
        analysis_start_frame = 0
        analysis_end_frame = 400  # 必要に応じて終了フレームを設定
    else:
        analysis_start_frame = 0
        analysis_end_frame = 400  # 必要に応じて終了フレームを設定

    print(f"処理を開始します: {base_dir.name}")
    print(f"入力ファイル: {json_file}")

    # 1. 3Dデータの読み込み
    points_data_dict = load_3d_data_from_json(json_file, person_key)
    if not points_data_dict:
        print("データが読み込めなかったため、処理を終了します。")
        return

    # 'raw_processed'と'final'の両方のデータセットに対して処理を実行
    angles_dict = {}
    ic_frames_dict = {}

    for data_type, points_3d in points_data_dict.items():
        print(f"\n--- '{data_type}' データの解析 ---")

        if np.isnan(points_3d).all():
            print(f"'{data_type}' データが全てNaNのためスキップします。")
            continue

        # 2. 歩行周期（踵接地）の算出
        ic_frames, relative_dist_dict = find_initial_contact_3d(points_3d)
        print("踵接地タイミングを検出しました:")
        print(f"  - 右足 (R): {ic_frames['R']}")
        print(f"  - 左足 (L): {ic_frames['L']}")

        if data_type == 'raw_processed':  # 1回だけ保存
            plt.plot(relative_dist_dict['R'], label='R')
            plt.plot(relative_dist_dict['L'], label='L')
            plt.xlabel('Frame')
            plt.ylabel('Relative Distance')
            plt.title('Relative Distance over Time')
            plt.legend()
            plt.savefig(base_dir / "relative_distance_plot.png")
            plt.close()

        if subject_name == "sub1" and thera_name == "thera0-2":
            start_frame = ic_frames['R'][3]
            end_frame = ic_frames['R'][7]
        else:
            start_frame = ic_frames['R'][0]
            end_frame = ic_frames['R'][-1]

        print(f"解析範囲: {start_frame} 〜 {end_frame} フレーム")

        # 3. 関節角度の計算
        angles_df = calc_joint_angles_3d(points_3d)
        print("関節角度を計算しました。")
        angles_df = angles_df.loc[start_frame:end_frame]

        # データを保存
        angles_dict[data_type] = angles_df
        ic_frames_dict[data_type] = ic_frames

    # 両方のデータが揃ったら比較プロットを作成
    if len(angles_dict) == 2:
        mocap_dir = base_dir / "mocap"
        if mocap_dir.exists():
            mocap_files = list(mocap_dir.glob("*.csv"))
            if mocap_files:
                mocap_path = mocap_files[0]
                print(f"MOCAPファイルを発見: {mocap_path}")
            else:
                print(f"警告: MOCAPディレクトリにCSVファイルが見つかりません: {mocap_dir}")
                mocap_path = None
        else:
            print(f"警告: MOCAPディレクトリが存在しません: {mocap_dir}")
            mocap_path = None

        # 4. 比較グラフの描画と保存
        plot_title = f"Joint Angles Comparison (raw_processed vs final) - {base_dir.name} ({person_key})"
        save_path = output_dir / f"{base_dir.name}_{person_key}_angles_comparison.png"
        plot_gait_angles_comparison(angles_dict, ic_frames_dict, start_frame, end_frame, plot_title, save_path)

    print("\n全ての処理が完了しました。")

if __name__ == '__main__':
    main()
