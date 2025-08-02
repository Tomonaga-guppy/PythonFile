import cv2
import numpy as np
from pathlib import Path

# XMLファイル名
file_name = Path("G:/gait_pattern/20250717_br/camera_parameters/18079958.xml")

# FileStorageオブジェクトを使用してファイルを読み込みモードで開く
fs = cv2.FileStorage(str(file_name), cv2.FILE_STORAGE_READ)

# エラーハンドリング: ファイルが開けたか確認
if not fs.isOpened():
    print(f"エラー: ファイル '{file_name}' を開けませんでした。")
else:
    # 各ノード名（キー）を指定してデータをNumPy配列として取得
    camera_matrix = fs.getNode("CameraMatrix").mat()
    intrinsics = fs.getNode("Intrinsics").mat()
    distortion = fs.getNode("Distortion").mat()
    camera_matrix_initial = fs.getNode("CameraMatrixInitial").mat()

    # FileStorageオブジェクトを解放
    fs.release()

    # 取得したパラメータを表示
    print("## カメラパラメータの取得結果")

    print("\n### 内部パラメータ行列 (Intrinsics):")
    print("この3x3行列は、カメラの内部的な特性（焦点距離、主点）を表します。")
    print(intrinsics)
    # f_x, f_y, c_x, c_y を抽出して表示
    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    c_x = intrinsics[0, 2]
    c_y = intrinsics[1, 2]
    print(f" - 焦点距離 (fx, fy): ({f_x:.2f}, {f_y:.2f})")
    print(f" - 主点 (cx, cy): ({c_x:.2f}, {c_y:.2f})")


    print("\n### 歪み係数 (Distortion):")
    print("このベクトルは、レンズの歪み（放射状歪み、接線方向歪みなど）を表す係数を含みます。")
    print(distortion)

    print("\n### カメラ行列 (CameraMatrix):")
    print("この3x4射影行列は、3Dの点を2D画像平面に投影します。[R|t] の形式で回転と並進を含みますが、このファイルでは初期値として単位行列とゼロベクトルが設定されています。")
    print(camera_matrix)

    print("\n### 初期カメラ行列 (CameraMatrixInitial):")
    print("キャリブレーションの初期値として使用されたカメラ行列です。")
    print(camera_matrix_initial)

test_img = cv2.imread("G:/gait_pattern/20250717_br/int_cali/fl/8x6/cali_frames_all/frame_00.png")
if test_img is not None:
    print("\n### テスト画像の読み込みに成功しました。画像サイズ:", test_img.shape)

# 読み込んだ画像を歪み補正
undistorted_img = cv2.undistort(test_img, intrinsics, distortion)
if undistorted_img is not None:
    print("### 歪み補正に成功しました。")
    # cv2.imshow("Original Image", test_img)
    # cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.imwrite("G:/gait_pattern/20250717_br/int_cali/fl/8x6/undistorted_frame_00.png", undistorted_img)

    cv2.destroyAllWindows()
else:
    print("### 歪み補正に失敗しました。")