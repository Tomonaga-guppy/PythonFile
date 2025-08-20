import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_49d5")
    directions = ['fr']
    # directions = ['fl', 'fr']
    checker_pattern = (5, 4)
    square_size = 49.5  # mm単位

    print(f"チェッカーボードのパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")

    # 3D座標の準備 (標準モデル用の形式)
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    for direction in directions:
        print(f"\n{'='*80}")
        print(f"カメラ: '{direction}' の標準キャリブレーションを開始します (除外モード)")
        print(f"{'='*80}")

        cali_img_folder = root_dir / direction / "cali_imgs"
        cali_imgs = sorted(list(cali_img_folder.glob("*.png")))
        cali_imgs = [p for p in cali_imgs if not p.name.startswith("excluded_")]

        # --- ここから変更点: successフォルダの準備 ---
        success_folder = root_dir / direction / "success"
        success_folder.mkdir(parents=True, exist_ok=True)
        # --- 変更点ここまで ---

        if not cali_imgs:
            print(f"エラー: {cali_img_folder} に有効な画像ファイルが見つかりません。")
            continue

        objpoints = []
        imgpoints = []
        img_paths_used = []

        print("ステップ1: 全画像からチェッカーボードのコーナーを検出します...")
        for cali_img_path in tqdm(cali_imgs, desc=f"[{direction}] コーナー検出中"):
            img = cv2.imread(str(cali_img_path))
            gray_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray_color, checker_pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray_color, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                img_paths_used.append(cali_img_path)

                # --- ここから変更点: 検出成功画像をsuccessフォルダに保存 ---
                corner_img = cv2.drawChessboardCorners(img.copy(), checker_pattern, corners2, ret)
                cv2.imwrite(str(success_folder / cali_img_path.name), corner_img)
                # --- 変更点ここまで ---

        if len(objpoints) < 10:
            print(f"エラー: コーナー検出に成功した画像が少なすぎます ({len(objpoints)}枚)。")
            continue

        print(f"\n{len(objpoints)}枚の画像でコーナー検出に成功しました。")
        image_size = gray_color.shape[::-1]

        # ステップ2: 初期キャリブレーション
        print("\nステップ2: 初期キャリブレーションを実行し、各画像の誤差を評価します...")
        calib_flags = cv2.CALIB_RATIONAL_MODEL
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None, flags=calib_flags)

        if not ret:
            print("初期キャリブレーションに失敗しました。")
            continue

        # ステップ3: ユーザーによる目視での画像除外
        print("\nステップ3: 最終キャリブレーションから除外する画像を、目視で選択してください。")
        per_image_errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            per_image_errors.append(error)

        error_data = sorted(zip(per_image_errors, img_paths_used), key=lambda x: x[1].name)

        graph_path = root_dir / direction / f"reprojection_errors_std_{direction}.png"
        graph_labels = [f"[{i}] {item[1].name}" for i, item in enumerate(error_data)]
        graph_errors = [item[0] for item in error_data]

        plt.figure(figsize=(15, 6))
        plt.bar(range(len(error_data)), graph_errors, tick_label=graph_labels)
        plt.ylabel("Reprojection Error (pixels)")
        plt.xlabel("Image Files ([index] filename)")
        plt.title(f"Per-Image Reprojection Errors for Camera '{direction}' (Standard Model)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
        print(f"\n判断材料として、誤差のグラフを保存しました: {graph_path}")

        print("\n--- 画像除外リスト ---")
        for i, (error, path) in enumerate(error_data):
            print(f"  [{i:2d}] {path.name} (誤差: {error:.4f} pixels)")

        excluded_indices = []
        while True:
            try:
                print("\n最終キャリブレーションから『除外する』画像の番号をカンマ区切りで入力してください。")
                user_input = input("何も除外しない場合は、そのままEnterキーを押してください: ")
                if not user_input: break
                excluded_indices = [int(i.strip()) for i in user_input.split(',')]
                if all(0 <= i < len(error_data) for i in excluded_indices): break
                else: print("エラー: 範囲外の番号が入力されました。")
            except ValueError: print("エラー: 不正な入力です。")

        # --- ここから変更点: successフォルダ内のファイルもリネーム ---
        if excluded_indices:
            print("\n以下の画像を除外対象としてファイル名を変更します:")
            for i in excluded_indices:
                path_to_exclude_original = error_data[i][1]
                path_to_exclude_success = success_folder / path_to_exclude_original.name

                # 元の画像をリネーム
                new_name = f"excluded_{path_to_exclude_original.name}"
                new_path_original = path_to_exclude_original.with_name(new_name)
                if path_to_exclude_original.exists() and not new_path_original.exists():
                    path_to_exclude_original.rename(new_path_original)
                    print(f"  - 元ファイルをリネーム: {path_to_exclude_original.name} -> {new_name}")

                # successフォルダ内の画像をリネーム
                new_path_success = path_to_exclude_success.with_name(new_name)
                if path_to_exclude_success.exists() and not new_path_success.exists():
                    path_to_exclude_success.rename(new_path_success)
                    print(f"  - success内ファイルをリネーム: {path_to_exclude_success.name} -> {new_name}")
        # --- 変更点ここまで ---

        # 除外されなかったデータを整理
        original_indices_map = {path: i for i, path in enumerate(img_paths_used)}
        objpoints_clean, imgpoints_clean = [], []
        all_indices, excluded_set = set(range(len(error_data))), set(excluded_indices)
        kept_indices = sorted(list(all_indices - excluded_set))

        print("\n以下の画像が最終キャリブレーションに使用されます:")
        kept_paths = [error_data[i][1] for i in kept_indices]
        for path in sorted(kept_paths, key=lambda p: p.name):
            print(f"  - {path.name}")
            idx = original_indices_map[path]
            objpoints_clean.append(objpoints[idx])
            imgpoints_clean.append(imgpoints[idx])

        if len(objpoints_clean) < 10:
            print(f"警告: 残った優良画像が少なすぎます ({len(objpoints_clean)}枚)。")
            if not objpoints_clean:
                print("使用可能な画像がなくなったため、処理を中断します。")
                continue

        # ステップ4: 最終キャリブレーション
        print(f"\nステップ4: {len(objpoints_clean)}枚の画像で最終キャリブレーションを実行します...")
        ret, final_mtx, final_dist, _, _ = cv2.calibrateCamera(
            objpoints_clean, imgpoints_clean, image_size, None, None, flags=calib_flags)

        if not ret:
            print("最終キャリブレーションに失敗しました。")
            continue

        # 最終結果の表示と保存
        print("\n【最終的なキャリブレーション結果】")
        print(f"再投影誤差(RMS): {ret:.4f} pixels")
        print(f"内部パラメータ行列 (fx, fy, cx, cy):")
        print(f"  fx={final_mtx[0,0]:.2f}, fy={final_mtx[1,1]:.2f}, cx={final_mtx[0,2]:.2f}, cy={final_mtx[1,2]:.2f}")
        print(f"歪み係数 (k1,k2,p1,p2,k3,k4,k5,k6): \n  {final_dist.flatten()}")

        result_file = root_dir / direction / "camera_params.json"
        result_data = {
            "model_type": "standard_rational", "reprojection_error_rms": ret,
            "intrinsics": final_mtx.tolist(), "distortion": final_dist.tolist(),
            "image_width": image_size[0], "image_height": image_size[1],
            "used_images_count_final": len(objpoints_clean),
            "excluded_images_indices": excluded_indices,
            "used_checkerboard_pattern": checker_pattern,
            "used_square_size_mm": square_size,
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"\n手動選択による標準モデルのパラメータを保存しました: {result_file}")

if __name__ == '__main__':
    main()