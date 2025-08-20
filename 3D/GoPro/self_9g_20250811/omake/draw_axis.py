import cv2
import numpy as np
from pathlib import Path

# --- パラメータ設定 ---

# 2. チェッカーボードの交点の数（内側）
# パターンが6x5の場合、内側の交点の数は (6-1) x (5-1) = 5x4 となります。
checkerboard_size = (5, 4)

# 3. チェッカーボードの正方形の1辺の長さ (mm)
square_size = 35.0

# 4. 描画する座標軸の長さ (mm)
axis_length = square_size * 3

# --- 処理の実行 ---

def main():
    for direction in ['fl', 'fr']:
        image_path = Path(fr"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35\{direction}\cali_imgs\0031.png")
        # 画像を読み込む
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"エラー: 画像ファイル '{image_path}' を読み込めませんでした。")
            return

        # 処理のためにグレースケール画像に変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェッカーボードの3D座標を準備 (Z=0の平面と仮定)
        # (0,0,0), (35,0,0), (70,0,0), ..., (35*4, 35*3, 0)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp = objp * square_size

        # チェッカーボードのコーナーを検出
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # コーナーが検出された場合
        if ret:
            print("チェッカーボードのコーナーを検出しました。")

            # コーナーの精度を向上させる
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # ----- カメラのポーズを推定 (solvePnP) -----
            # 注意: 正確な座標軸を描画するには、事前にカメラキャリブレーションで得られた
            # カメラ行列と歪み係数が必要です。
            # ここでは、簡易的な仮の値を設定します。

            # 画像サイズから仮のカメラ行列を作成
            height, width = img.shape[:2]
            # 仮の焦点距離を画像の幅とする
            focal_length = width
            camera_matrix = np.array([
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            # 歪みは無いものと仮定
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            # 3Dオブジェクトポイントと2D画像ポイントから回転と並進ベクトルを計算
            _, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)

            # 座標軸を描画
            # OpenCV 4.7.0以降では cv2.drawFrameAxes が推奨されます
            try:
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
            except AttributeError:
                # 古いバージョンのOpenCVの場合
                from cv2.aruco import drawAxis
                drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

            print("座標軸を描画しました。")

            # 検出したコーナーを描画（オプション）
            cv2.drawChessboardCorners(img, checkerboard_size, corners_subpix, ret)

        else:
            print("エラー: チェッカーボードのコーナーを検出できませんでした。")
            print("チェッカーボードのサイズ設定や画像の品質を確認してください。")


        # # 結果の画像を表示
        # cv2.imshow('Checkerboard with Axes', img)
        # print("画像が表示されています。何かキーを押すとウィンドウが閉じます。")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 結果をファイルに保存
        output_path = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35") / f"{direction}_checkerboard_axes.png"
        cv2.imwrite(str(output_path), img)
        print(f"結果を '{output_path}' に保存しました。")


if __name__ == '__main__':
    main()