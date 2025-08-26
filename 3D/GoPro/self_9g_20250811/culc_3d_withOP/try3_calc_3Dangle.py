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
    'raw'と'raw_filt'の両方のデータを読み込む。

    Args:
        json_path (Path): 3D結果が含まれるJSONファイルのパス。
        person_key (str): 読み込む人物のキー (例: 'person_1')。

    Returns:
        dict: 'raw'と'raw_filt'のデータをNumPy配列で格納した辞書。
              データが見つからない場合はNoneを返す。
    """
    try:
        with open(json_path, 'r') as f:
            all_frames_data = json.load(f)

        data_sets = {'raw': [], 'raw_filt': []}
        data_keys = {
            'raw': 'points_3d_raw',
            'raw_filt': 'points_3d_raw_filt'
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
        if not data_sets['raw'] or not data_sets['raw_filt']:
            print(f"警告: {person_key} のデータが見つかりませんでした。")
            return None

        return {
            'raw': np.array(data_sets['raw']),
            'raw_filt': np.array(data_sets['raw_filt'])
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
    
    r_thigh_vector = p[:, KEYPOINTS_MAP['RKnee']] - p[:, KEYPOINTS_MAP['RHip']]
    r_hip_flexion_angle = calculate_3d_vector_angle(r_thigh_vector, trunk_vector)

    # 膝関節の屈曲伸展角度
    l_shank_vector = p[:, KEYPOINTS_MAP['LAnkle']] - p[:, KEYPOINTS_MAP['LKnee']]
    l_knee_flexion_angle = calculate_3d_vector_angle(l_thigh_vector, l_shank_vector)
    
    r_shank_vector = p[:, KEYPOINTS_MAP['RAnkle']] - p[:, KEYPOINTS_MAP['RKnee']]
    r_knee_flexion_angle = calculate_3d_vector_angle(r_thigh_vector, r_shank_vector)

    # 足関節の背屈底屈角度
    l_foot_vector = (p[:, KEYPOINTS_MAP['LBigToe']] + p[:, KEYPOINTS_MAP['LSmallToe']]) - p[:, KEYPOINTS_MAP['LAnkle']]
    l_ankle_dorsiflexion_angle = calculate_3d_vector_angle(l_shank_vector, l_foot_vector)

    r_foot_vector = (p[:, KEYPOINTS_MAP['RBigToe']] + p[:, KEYPOINTS_MAP['RSmallToe']]) - p[:, KEYPOINTS_MAP['RAnkle']]
    r_ankle_dorsiflexion_angle = calculate_3d_vector_angle(r_shank_vector, r_foot_vector)
    
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


def plot_gait_angles(angles_df, ic_frames, title, save_path):
    """
    歩行角度をプロットする
    """
    # デバッグ: 利用可能なカラムを確認
    print(f"Available columns: {list(angles_df.columns)}")
    
    joint_names = ['Hip', 'Knee', 'Ankle']
    colors = {'L': 'blue', 'R': 'red'}
    ic_colors = {'L': 'lightblue', 'R': 'lightcoral'}
    
    fig, axes = plt.subplots(len(joint_names), 1, figsize=(15, 12), sharex=True)
    if len(joint_names) == 1:
        axes = [axes]  # 単一プロットの場合のための調整
    
    for i, joint in enumerate(joint_names):
        ax = axes[i]
        
        # 正しい大文字でsideを定義
        sides = ['L', 'R']  # 大文字に修正
        
        for side in sides:
            col_name = f"{joint}_{side}"
            
            # カラムが存在するかチェック
            if col_name in angles_df.columns:
                ax.plot(angles_df.index, angles_df[col_name], label=col_name, color=colors[side], lw=2)
            else:
                print(f"Warning: Column '{col_name}' not found in DataFrame")
        
        # IC（初期接触）の縦線を描画
        for side_ic, frames in ic_frames.items():
            for frame in frames:
                ax.axvline(x=frame, color=ic_colors[side_ic], linestyle='--', alpha=0.7, 
                          label=f'IC_{side_ic}' if frame == frames[0] else "")
        
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{joint} Angle')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 凡例の重複を除去
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    # X軸ラベルは最下段のみ
    axes[-1].set_xlabel('Frame Number')
    
    plt.tight_layout()
    plt.suptitle(title, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gait_angles_comparison(angles_spline_df, angles_final_df, ic_frames, title, save_path):
    """
    フィルターありなしの角度を同時にプロットする関数（関節別に分離）
    """
    joint_names = ['Hip', 'Knee', 'Ankle']
    joint_config = {
        'Hip': {'ylim': (-30, 60), 'color_spline': 'lightblue', 'color_filter': 'blue'},
        'Knee': {'ylim': (-10, 80), 'color_spline': 'lightcoral', 'color_filter': 'red'},
        'Ankle': {'ylim': (-40, 30), 'color_spline': 'lightgreen', 'color_filter': 'green'}
    }
    
    # 関節別に3つの個別グラフを作成
    for joint in joint_names:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle(f'{title} - {joint} Joint', fontsize=16, fontweight='bold')
        
        sides = ['L', 'R']  # 大文字に修正
        side_styles = {'L': '-', 'R': '--'}  # 左は実線、右は破線
        frames = np.arange(len(angles_spline_df))
        
        for side in sides:
            # スプライン補間のみのデータをプロット
            spline_col = f'{joint}_{side}'
            if spline_col in angles_spline_df.columns:
                ax.plot(angles_spline_df.index, angles_spline_df[spline_col], 
                       linestyle=side_styles[side], color=joint_config[joint]['color_spline'], 
                       linewidth=2, alpha=0.7, label=f'{side}_spline')
            
            # フィルター適用後のデータをプロット
            filter_col = f'{joint}_{side}'
            if filter_col in angles_final_df.columns:
                ax.plot(angles_final_df.index, angles_final_df[filter_col], 
                       linestyle=side_styles[side], color=joint_config[joint]['color_filter'], 
                       linewidth=2.5, label=f'{side}_filtered')
        
        # IC（初期接触）の縦線を描画
        ic_colors = {'L': 'purple', 'R': 'orange'}
        for side_ic, frames in ic_frames.items():
            for frame in frames:
                ax.axvline(x=frame, color=ic_colors[side_ic], linestyle=':', 
                          alpha=0.8, linewidth=2, 
                          label=f'IC_{side_ic}' if frame == frames[0] else "")
        
        # 軸の設定
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_title(f'{joint} Joint Angle (Spline vs Filtered)', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Y軸の範囲を関節別に設定
        ax.set_ylim(joint_config[joint]['ylim'])
        
        # 凡例の重複を除去
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        # 関節別にファイルを保存
        joint_save_path = save_path.parent / f"{save_path.stem}_{joint}_comparison.png"
        plt.savefig(joint_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {joint}関節の比較グラフを保存: {joint_save_path}")

def main():
    """メイン処理"""
    # --- 設定 ---
    # 解析したいデータが含まれるディレクトリを指定してください
    base_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera0-2")
    
    # try_1_spline.pyが出力したJSONファイルを指定
    json_file = base_dir / "3d_gait_analysis" / "thera0-2_3d_results.json"

    # 結果を出力するディレクトリ
    output_dir = base_dir / "gait_angle_analysis_from_3d"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 解析対象の人物
    person_key = 'person_1'

    print(f"処理を開始します: {base_dir.name}")
    print(f"入力ファイル: {json_file}")

    # 1. 3Dデータの読み込み
    points_data_dict = load_3d_data_from_json(json_file, person_key)
    if not points_data_dict:
        print("データが読み込めなかったため、処理を終了します。")
        return

    # 'raw'と'raw_filt'の両方のデータセットに対して処理を実行
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

        plt.plot(relative_dist_dict['R'], label='R')
        plt.plot(relative_dist_dict['L'], label='L')
        plt.xlabel('Frame')
        plt.ylabel('Relative Distance')
        plt.title('Relative Distance over Time')
        plt.legend()
        plt.show()

        # 3. 関節角度の計算
        angles_df = calc_joint_angles_3d(points_3d)
        print("関節角度を計算しました。")
        
        # 4. グラフの描画と保存
        plot_title = f"Joint Angles and Gait Cycle - {base_dir.name} ({person_key}, {data_type})"
        save_path = output_dir / f"{base_dir.name}_{person_key}_{data_type}_angles.png"
        plot_gait_angles(angles_df, ic_frames, plot_title, save_path)

    print("\n全ての処理が完了しました。")

    # --- 比較グラフの処理 ---
    # スプライン補間とフィルター適用後のデータを比較するグラフを作成
    for subject_dir in base_dir.glob("sub*"):
        if not subject_dir.is_dir():
            continue
        
        thera_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")])
        
        for thera_dir in thera_dirs:
            if sub_dir.name != "sub1" and thera_dir.name != "thera0-2":
                continue
                
            print(f"\n処理開始: {thera_dir}")
            
            # 出力ディレクトリの作成（parents=Trueを追加）
            output_dir = thera_dir / "gait_angle_analysis_from_3d"
            output_dir.mkdir(parents=True, exist_ok=True)  # ← parents=True を追加
            
            # 各人物のデータに対して処理
            data = {}
            for person_json in (thera_dir / "3d_gait_analysis").glob("*_3d_results.json"):
                person_key = person_json.stem.split("_")[0]
                person_data = load_3d_data_from_json(person_json, person_key)
                if person_data:
                    data[person_key] = person_data
            
            # 各人物のスプライン補間+フィルターと最適化（組み換え）＋フィルター適用後の角度データを計算
            for person_key, person_data in data.items():
                if person_key.startswith('person_'):
                    person_output_dir = output_dir / person_key
                    person_output_dir.mkdir(parents=True, exist_ok=True)  # ← parents=True を追加
                    
                    # スプライン補間のみのデータを処理
                    spline_points = person_data['points_3d_flit']
                    angles_spline_df, ic_frames_spline = calculate_joint_angles_and_ic(spline_points, frame_rate=60)
                    spline_save_path = person_output_dir / f"{person_key}_flit_angles.csv"
                    angles_spline_df.to_csv(spline_save_path, index=False)
                    
                    # フィルター適用後のデータを処理
                    final_points = person_data['points_3d_final']
                    angles_final_df, ic_frames_final = calculate_joint_angles_and_ic(final_points, frame_rate=60)
                    final_save_path = person_output_dir / f"{person_key}_final_angles.csv"
                    angles_final_df.to_csv(final_save_path, index=False)
                    
                    # 比較グラフの作成
                    plot_title = f"Joint Angles Comparison - {thera_dir.name} ({person_key})"
                    plot_save_path = person_output_dir / f"{person_key}_angles_comparison"
                    
                    plot_gait_angles_comparison(
                        angles_spline_df, angles_final_df, ic_frames_final, 
                        plot_title, plot_save_path
                    )
                    
                    print(f"  ✓ {person_key}の処理完了")

if __name__ == '__main__':
    main()
