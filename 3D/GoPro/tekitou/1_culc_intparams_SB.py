import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

def generate3Dgrid(CheckerBoardParams):
    """
    チェッカーボードの3D座標を生成する
    """
    dimensions = CheckerBoardParams['dimensions']  # (width, height)
    square_size = CheckerBoardParams['squareSize']

    # 3D座標の準備
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp *= square_size

    return objp

def detect_chessboard_corners_sb(image, checker_pattern, square_size):
    """
    cv2.findChessboardCornersSBWithMeta を使用した高精度コーナー検出

    Args:
        image: 入力画像
        checker_pattern: チェッカーボードパターン (width, height)
        square_size: 正方形のサイズ

    Returns:
        ret: 検出成功フラグ
        corners2: 検出されたコーナー座標
        objp: 対応する3D座標
        meta: メタデータ
    """
    gray_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SBWithMetaを使用した高精度検出
    ret, corners, meta = cv2.findChessboardCornersSBWithMeta(
        gray_color,
        checker_pattern,
        cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
    )

    if ret:
        # メタデータから実際に検出されたパターンサイズを取得
        actual_pattern = meta.shape[::-1]  # (width, height)に変換

        # 実際のパターンに基づいて3D座標を生成
        CheckerBoardParams = {
            'dimensions': actual_pattern,
            'squareSize': square_size
        }
        objp = generate3Dgrid(CheckerBoardParams)

        # SBWithMetaはサブピクセル精度で検出するため、cornerSubPixは不要
        corners2 = corners

        return ret, corners2, objp, meta, actual_pattern
    else:
        return ret, None, None, None, None

def main():
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_35")
    directions = ['sagi']
    # directions = ['fl', 'fr']
    checker_pattern = (5, 4)  # (width, height) - 期待するパターン
    square_size = 35  # mm単位

    print(f"チェッカーボードのパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")

    for direction in directions:
        print(f"\n{'='*80}")
        print(f"カメラ: '{direction}' のキャリブレーションを開始します")
        print(f"{'='*80}")

        cali_img_folder = root_dir / direction / "cali_imgs"
        cali_imgs = sorted(list(cali_img_folder.glob("*.png")))
        cali_imgs = [p for p in cali_imgs if not p.name.startswith("excluded_")]

        # successフォルダの準備
        success_folder = root_dir / direction / "success"
        success_folder.mkdir(parents=True, exist_ok=True)

        if not cali_imgs:
            print(f"エラー: {cali_img_folder} に有効な画像ファイルが見つかりません。")
            continue

        objpoints = []
        imgpoints = []
        img_paths_used = []
        pattern_info = []  # 検出されたパターン情報を保存

        print("ステップ1: 全画像からチェッカーボードのコーナーを検出します (SB方式)...")
        successful_detections = 0

        for cali_img_path in tqdm(cali_imgs, desc=f"[{direction}] SBコーナー検出中"):
            img = cv2.imread(str(cali_img_path))
            if img is None:
                print(f"警告: 画像を読み込めません: {cali_img_path}")
                continue

            # SBWithMetaを使用した検出
            ret, corners2, objp, meta, actual_pattern = detect_chessboard_corners_sb(
                img, checker_pattern, square_size
            )

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners2)
                img_paths_used.append(cali_img_path)
                pattern_info.append(actual_pattern)
                successful_detections += 1

                # 検出成功画像をsuccessフォルダに保存
                corner_img = cv2.drawChessboardCorners(
                    img.copy(), actual_pattern, corners2, ret
                )
                cv2.imwrite(str(success_folder / cali_img_path.name), corner_img)

                print(f"  成功: {cali_img_path.name} - パターン: {actual_pattern[0]}x{actual_pattern[1]}")
            else:
                print(f"  失敗: {cali_img_path.name} - チェッカーボードが検出できません")

        if len(objpoints) < 10:
            print(f"エラー: コーナー検出に成功した画像が少なすぎます ({len(objpoints)}枚)。")
            print("最低10枚以上の成功画像が必要です。")
            continue

        print(f"\n{len(objpoints)}枚の画像でコーナー検出に成功しました。")

        # パターンサイズの統計を表示
        unique_patterns = list(set(pattern_info))
        print(f"検出されたパターンサイズの種類: {len(unique_patterns)}")
        for pattern in unique_patterns:
            count = pattern_info.count(pattern)
            print(f"  {pattern[0]}x{pattern[1]}: {count}枚")

        # 画像サイズを取得
        first_img = cv2.imread(str(img_paths_used[0]))
        image_size = (first_img.shape[1], first_img.shape[0])  # (width, height)

        # ステップ2: 初期キャリブレーション
        print("\nステップ2: 初期キャリブレーションを実行し、各画像の誤差を評価します...")
        calib_flags = cv2.CALIB_RATIONAL_MODEL

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None, flags=calib_flags
            )
        except cv2.error as e:
            print(f"初期キャリブレーションエラー: {e}")
            print("標準モデルで再試行します...")
            calib_flags = 0  # 標準モデル
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None, flags=calib_flags
            )

        if not ret:
            print("初期キャリブレーションに失敗しました。")
            continue

        print(f"初期キャリブレーション完了 - RMS誤差: {ret:.4f} pixels")

        # ステップ3: ユーザーによる目視での画像除外
        print("\nステップ3: 最終キャリブレーションから除外する画像を、目視で選択してください。")
        per_image_errors = []

        for i in range(len(objpoints)):
            try:
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                per_image_errors.append(error)
            except cv2.error as e:
                print(f"警告: 画像 {i} の誤差計算に失敗: {e}")
                per_image_errors.append(float('inf'))

        error_data = sorted(zip(per_image_errors, img_paths_used, pattern_info),
                           key=lambda x: x[1].name)

        # グラフの生成
        graph_path = root_dir / direction / f"reprojection_errors_sb_{direction}.png"
        graph_labels = [f"[{i}] {item[1].name}\n({item[2][0]}x{item[2][1]})"
                       for i, item in enumerate(error_data)]
        graph_errors = [item[0] if item[0] != float('inf') else 0 for item in error_data]

        plt.figure(figsize=(16, 8))
        colors = ['red' if error == float('inf') else 'blue' for error in [item[0] for item in error_data]]
        plt.bar(range(len(error_data)), graph_errors, color=colors)
        plt.ylabel("Reprojection Error (pixels)")
        plt.xlabel("Image Files ([index] filename + pattern size)")
        plt.title(f"Per-Image Reprojection Errors for Camera '{direction}' (SB Method)")
        plt.xticks(range(len(error_data)), graph_labels, rotation=90)
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n判断材料として、誤差のグラフを保存しました: {graph_path}")

        print("\n--- 画像除外リスト (SB方式) ---")
        for i, (error, path, pattern) in enumerate(error_data):
            error_str = f"{error:.4f}" if error != float('inf') else "ERROR"
            print(f"  [{i:2d}] {path.name} (誤差: {error_str} pixels, パターン: {pattern[0]}x{pattern[1]})")

        # ユーザー入力による除外選択
        excluded_indices = []
        while True:
            try:
                print("\n最終キャリブレーションから『除外する』画像の番号をカンマ区切りで入力してください。")
                print("エラーが発生した画像（ERROR表示）は自動的に除外されます。")
                user_input = input("何も除外しない場合は、そのままEnterキーを押してください: ")
                if not user_input:
                    break
                excluded_indices = [int(i.strip()) for i in user_input.split(',')]
                if all(0 <= i < len(error_data) for i in excluded_indices):
                    break
                else:
                    print("エラー: 範囲外の番号が入力されました。")
            except ValueError:
                print("エラー: 不正な入力です。")

        # エラー画像を自動的に除外リストに追加
        auto_excluded = [i for i, (error, _, _) in enumerate(error_data) if error == float('inf')]
        all_excluded = list(set(excluded_indices + auto_excluded))

        # ファイルのリネーム処理
        if all_excluded:
            print(f"\n以下の{len(all_excluded)}枚の画像を除外対象としてファイル名を変更します:")
            for i in all_excluded:
                path_to_exclude_original = error_data[i][1]
                path_to_exclude_success = success_folder / path_to_exclude_original.name

                # 元の画像をリネーム
                new_name = f"excluded_{path_to_exclude_original.name}"
                new_path_original = path_to_exclude_original.with_name(new_name)
                if path_to_exclude_original.exists() and not new_path_original.exists():
                    path_to_exclude_original.rename(new_path_original)
                    exclude_reason = "ERROR" if i in auto_excluded else "USER"
                    print(f"  - [{exclude_reason}] {path_to_exclude_original.name} -> {new_name}")

                # successフォルダ内の画像をリネーム
                new_path_success = path_to_exclude_success.with_name(new_name)
                if path_to_exclude_success.exists() and not new_path_success.exists():
                    path_to_exclude_success.rename(new_path_success)

        # 除外されなかったデータを整理
        original_indices_map = {path: i for i, (_, path, _) in enumerate(zip(per_image_errors, img_paths_used, pattern_info))}
        objpoints_clean, imgpoints_clean = [], []
        all_indices, excluded_set = set(range(len(error_data))), set(all_excluded)
        kept_indices = sorted(list(all_indices - excluded_set))

        print(f"\n以下の{len(kept_indices)}枚の画像が最終キャリブレーションに使用されます:")
        kept_data = [error_data[i] for i in kept_indices]
        for error, path, pattern in sorted(kept_data, key=lambda x: x[1].name):
            print(f"  - {path.name} (誤差: {error:.4f}, パターン: {pattern[0]}x{pattern[1]})")
            idx = original_indices_map[path]
            objpoints_clean.append(objpoints[idx])
            imgpoints_clean.append(imgpoints[idx])

        if len(objpoints_clean) < 10:
            print(f"警告: 残った優良画像が少なすぎます ({len(objpoints_clean)}枚)。")
            if not objpoints_clean:
                print("使用可能な画像がなくなったため、処理を中断します。")
                continue

        # ステップ4: 最終キャリブレーション
        print(f"\nステップ4: {len(objpoints_clean)}枚の画像で最終キャリブレーション (SB方式) を実行します...")

        try:
            ret_final, final_mtx, final_dist, _, _ = cv2.calibrateCamera(
                objpoints_clean, imgpoints_clean, image_size, None, None, flags=calib_flags
            )
        except cv2.error as e:
            print(f"最終キャリブレーションエラー: {e}")
            continue

        if not ret_final:
            print("最終キャリブレーションに失敗しました。")
            continue

        # 最終結果の表示と保存
        print("\n" + "="*60)
        print("【最終的なキャリブレーション結果 (SB方式)】")
        print("="*60)
        print(f"再投影誤差(RMS): {ret_final:.4f} pixels")
        print(f"使用画像数: {len(objpoints_clean)}枚")
        print(f"画像サイズ: {image_size[0]} x {image_size[1]}")
        print(f"キャリブレーションフラグ: {calib_flags}")
        print(f"\n内部パラメータ行列:")
        print(f"  fx = {final_mtx[0,0]:.2f}")
        print(f"  fy = {final_mtx[1,1]:.2f}")
        print(f"  cx = {final_mtx[0,2]:.2f}")
        print(f"  cy = {final_mtx[1,2]:.2f}")
        print(f"\n歪み係数 ({len(final_dist.flatten())}個):")
        for i, coeff in enumerate(final_dist.flatten()):
            print(f"  d{i+1} = {coeff:.6f}")

        # 結果をJSONファイルに保存
        result_file = root_dir / direction / "camera_params.json"
        result_data = {
            "calibration_method": "SBWithMeta",
            "model_type": "rational" if calib_flags == cv2.CALIB_RATIONAL_MODEL else "standard",
            "reprojection_error_rms": float(ret_final),
            "intrinsics": final_mtx.tolist(),
            "distortion": final_dist.tolist(),
            "image_width": image_size[0],
            "image_height": image_size[1],
            "used_images_count_final": len(objpoints_clean),
            "total_images_processed": len(cali_imgs),
            "excluded_images_indices": all_excluded,
            "auto_excluded_count": len(auto_excluded),
            "user_excluded_count": len(excluded_indices),
            "expected_checkerboard_pattern": checker_pattern,
            "square_size_mm": square_size,
            "detected_pattern_types": unique_patterns,
            "calibration_flags": int(calib_flags)
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=4)

        print(f"\nSB方式による高精度パラメータを保存しました: {result_file}")
        print("="*60)

if __name__ == '__main__':
    main()