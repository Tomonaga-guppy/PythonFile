import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def main():
    # --- 1. パラメータ設定 ---
    # プロジェクトのルートディレクトリ
    root_dir = Path(r"G:\gait_pattern")

    # 内部パラメータディレクトリ
    int_cali_dir = root_dir / "int_cali" / "9g_20250807_6x5"
    
    # 外部パラメータ用画像があるディレクトリ
    ext_cali_dir = root_dir / "stereo_cali" / "9g_20250811"

    # カメラディレクトリ名
    camera_directions = ['fl', 'fr']

    # 内部パラメータファイル名
    intrinsic_filename = "camera_params.json"

    # 外部キャリブレーション用画像の入ったフォルダ名
    ext_img_folder_name = "cali_imgs"

    # チェッカーボードの物理的な設定
    checker_pattern = (5, 4)  # (width, height)
    square_size = 35.0  # mm単位
    print(f"チェッカーボードのパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")

    # チェッカーボードの3D座標を準備
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    # 終了基準
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- 2. 各カメラの外部パラメータを計算 ---
    print("\n各カメラの外部パラメータを計算します...")
    
    for direction in camera_directions:
        print(f"\n{'='*80}")
        print(f"カメラ '{direction}' の外部パラメータを計算中...")
        print(f"{'='*80}")
        
        # --- 2.1 内部パラメータの読み込み ---
        try:
            params_file = int_cali_dir / direction / intrinsic_filename
            with open(params_file, 'r') as f:
                camera_params = json.load(f)
                mtx = np.array(camera_params['intrinsics'])
                dist = np.array(camera_params['distortion'])
                print(f"カメラ ({direction}) の内部パラメータを読み込みました。")
        except FileNotFoundError as e:
            print(f"エラー: 内部パラメータファイルが見つかりません。")
            print(f"-> {params_file}")
            print("-> 1_culc_intparams_SB.py を先に実行してください。")
            continue
        except KeyError as e:
            print(f"エラー: 内部パラメータファイルのキーが不正です: {e}")
            continue

        # --- 2.2 キャリブレーション画像の読み込み ---
        img_dir = ext_cali_dir / direction / ext_img_folder_name
        if not img_dir.exists():
            print(f"エラー: キャリブレーション画像ディレクトリが見つかりません。")
            print(f"-> {img_dir}")
            continue

        img_paths = sorted(list(img_dir.glob("*.png")))
        if not img_paths:
            print(f"エラー: キャリブレーション用の画像が見つかりません。")
            print(f"-> {img_dir}")
            continue

        print(f"{len(img_paths)} 枚の画像からチェッカーボードを検出します...")

        # --- 2.3 チェッカーボードの検出 ---
        objpoints = []  # 3D点
        imgpoints = []  # 2D点
        img_size = None
        successful_images = []

        for img_path in tqdm(img_paths, desc=f"チェッカーボード検出中 ({direction})"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 画像を読み込めません: {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = gray.shape[::-1]

            # チェッカーボードコーナーの検出
            ret, corners = cv2.findChessboardCorners(gray, checker_pattern, None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                successful_images.append(img_path)

        print(f"\nチェッカーボードが検出されたのは {len(objpoints)} 枚でした。")

        if len(objpoints) < 5:
            print(f"エラー: 外部パラメータ計算に必要な画像が少なすぎます ({len(objpoints)}枚)。5枚以上を推奨します。")
            continue

        # --- 2.4 外部パラメータの計算 (solvePnP を使用) ---
        print(f"\n{len(objpoints)} 枚の画像で外部パラメータを計算します...")

        rvecs, tvecs = [], []
        reprojection_errors = []

        # 各画像に対して solvePnP を実行
        for i in range(len(objpoints)):
            ret, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)

            if ret:
                rvecs.append(rvec)
                tvecs.append(tvec)
                
                # 再投影エラーを計算
                imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
                reprojection_errors.append(error)

        if len(rvecs) < 1:
            print("エラー: 有効な画像で外部パラメータを計算できませんでした。")
            continue

        print(f"成功した {len(rvecs)} 枚の外部パラメータを平均化します。")
        
        # --- 2.5 外部パラメータの平均化 ---
        # 計算された外部パラメータを平均化
        rvec_avg = np.mean(rvecs, axis=0)
        tvec_avg = np.mean(tvecs, axis=0)
        avg_reprojection_error = np.mean(reprojection_errors)

        # 回転ベクトルを回転行列に変換
        R_avg, _ = cv2.Rodrigues(rvec_avg)

        # --- 2.6 結果の表示と保存 ---
        print(f"\n【カメラ '{direction}' の外部パラメータ計算結果】")
        print("\n回転行列 (R):")
        print(R_avg)
        print("\n並進ベクトル (T) [mm]:")
        print(tvec_avg.flatten())
        print(f"\n回転ベクトル (rodrigues): {rvec_avg.flatten()}")
        print(f"平均再投影エラー: {avg_reprojection_error:.4f} pixels")

        # 完全なカメラパラメータを作成
        camera_params_with_extrinsics = {
            # 内部パラメータを含む
            **camera_params,
            # 外部パラメータを追加
            'extrinsics': {
                'rotation_matrix': R_avg.tolist(),
                'translation_vector': tvec_avg.flatten().tolist(),
                'rotation_euler_angles': rvec_avg.flatten().tolist()
            },
            'calibration_statistics': {
                'used_images_count': len(objpoints),
                'total_images_count': len(img_paths),
                'average_reprojection_error': float(avg_reprojection_error),
                'individual_reprojection_errors': [float(e) for e in reprojection_errors]
            },
            'checkerboard_info': {
                'pattern': checker_pattern,
                'square_size_mm': square_size,
                'image_size': img_size
            },
            'method': 'solvePnP_averaged',
            'successful_images': [str(p.name) for p in successful_images]
        }

        # 出力先ディレクトリを作成
        output_dir = ext_cali_dir / direction
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "camera_params_with_extrinsics_op.json"

        with open(output_file, 'w') as f:
            json.dump(camera_params_with_extrinsics, f, indent=4)

        print(f"\n外部パラメータを含むカメラパラメータを {output_file} に保存しました。")

    # --- 3. 全体の結果をまとめて保存 ---
    print(f"\n{'='*80}")
    print("全カメラの外部パラメータ計算が完了しました。")
    
    all_camera_params = {}
    for direction in camera_directions:
        params_file = ext_cali_dir / direction / "camera_params_with_extrinsics.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                all_camera_params[direction] = json.load(f)

    if all_camera_params:
        summary_file = ext_cali_dir / "all_camera_parameters.json"
        with open(summary_file, 'w') as f:
            json.dump(all_camera_params, f, indent=4)
        
        print(f"全カメラのパラメータをまとめて保存: {summary_file}")

    print("\n後続の3D復元処理 (6_3d_reconstruction.py) でこれらのファイルを使用します。")
    print("\n処理が完了しました。")


if __name__ == '__main__':
    main()