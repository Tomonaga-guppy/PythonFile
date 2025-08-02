import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# --- 設定項目 ---
PARAMS_BASE_PATH = Path("G:/gait_pattern/20250717_br/camera_parameters")
EXT_CALI_BASE_PATH = Path("G:/gait_pattern/20250717_br/ext_cali")
CAMERA_NAMES = ["fl", "fr"]

# --- 関数定義 ---

def reset_extrinsic_parameters():
    """
    ユーザーに確認し、必要であれば外部パラメータ(CameraMatrix)をリセットする。
    """
    while True:
        # ユーザーにリセットの確認を求める
        answer = input("前回の外部パラメータをリセットしますか？ (y/n): ").lower().strip()

        if answer == 'y':
            print("\n外部パラメータをリセットしています...")

            # リセット用の単位行列を定義
            reset_camera_matrix = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=np.float64)

            # カメラごとに処理
            for i, camera_name in enumerate(CAMERA_NAMES):
                file_path = PARAMS_BASE_PATH / f"{i}.xml"

                if not file_path.exists():
                    print(f"警告: {file_path} が見つかりません。スキップします。")
                    continue

                # 1. 既存のパラメータをすべて読み込む
                fs_read = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_READ)
                if not fs_read.isOpened():
                    print(f"エラー: {file_path} を読み込めませんでした。")
                    continue

                intrinsics = fs_read.getNode("Intrinsics").mat()
                distortion = fs_read.getNode("Distortion").mat()
                # CameraMatrixInitialがない場合も考慮
                cam_matrix_initial_node = fs_read.getNode("CameraMatrixInitial")
                camera_matrix_initial = cam_matrix_initial_node.mat() if not cam_matrix_initial_node.empty() else np.zeros((3, 4))
                fs_read.release()

                # 2. CameraMatrixをリセットして、すべてのデータを書き戻す
                fs_write = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_WRITE)
                fs_write.write("CameraMatrix", reset_camera_matrix) # ここでリセット
                fs_write.write("Intrinsics", intrinsics)
                fs_write.write("Distortion", distortion)
                fs_write.write("CameraMatrixInitial", camera_matrix_initial)
                fs_write.release()
                print(f" {file_path} の CameraMatrix をリセットしました。")

            print("リセットが完了しました。\n")
            break # ループを抜ける

        elif answer == 'n':
            print("\nリセットをスキップして、現在のパラメータで処理を続行します。\n")
            break # ループを抜ける
        else:
            print("無効な入力です。'y' または 'n' で入力してください。")


def undistort_images():
    """
    内部パラメータを読み込み、画像を歪み補正する。
    """
    print("--- ステップ1: 画像の歪み補正を開始します ---")
    for i, camera_name in enumerate(CAMERA_NAMES):
        serial_num = str(i)
        file_name = PARAMS_BASE_PATH / f"{serial_num}.xml"

        # XMLファイルからパラメータを読み込む
        fs = cv2.FileStorage(str(file_name), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"エラー: ファイル '{file_name}' を開けませんでした。")
            continue

        intrinsics = fs.getNode("Intrinsics").mat()
        distortion = fs.getNode("Distortion").mat()
        fs.release()

        # 歪み補正する画像のフォルダと保存先フォルダを設定
        imgs_folder = EXT_CALI_BASE_PATH / camera_name / "8x6" / "cali_frames"
        undistorted_img_folder = EXT_CALI_BASE_PATH / "cali_ud" / camera_name # カメラごとにサブフォルダを作成
        undistorted_img_folder.mkdir(parents=True, exist_ok=True) # 保存先フォルダを作成

        imgs = sorted(list(imgs_folder.glob("*.png"))) # ファイル順を安定させるためにソート

        if not imgs:
            print(f"警告: {imgs_folder} に .png ファイルが見つかりません。")
            continue

        # プログレスバーを表示しながら歪み補正を実行
        for img_path in tqdm(imgs, desc=f"歪み補正中 ({camera_name})"):
            img = cv2.imread(str(img_path))
            if img is not None:
                undistorted_img = cv2.undistort(img, intrinsics, distortion)
                # ファイル名を連番にする（例：fl_0000.png, fr_0000.png）
                new_name = f"{camera_name}_{img_path.stem.split('_')[-1].zfill(4)}.png"
                cv2.imwrite(str(undistorted_img_folder / new_name), undistorted_img)
    print("画像の歪み補正が完了しました。\n")


# --- メインの実行部分 ---
if __name__ == "__main__":
    # 最初に、外部パラメータをリセットするかどうか確認する
    reset_extrinsic_parameters()

    # 次に、画像の歪み補正を実行する
    undistort_images()

    print(">>> すべてのプロセスが完了しました。")