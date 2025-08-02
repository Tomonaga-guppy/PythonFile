# チェッカーボードを検出する適当なプログラム

import cv2
import numpy as np

def detect_checkerboard(image_path):
    """
    画像からチェッカーボードのコーナーを検出して表示する関数

    Args:
        image_path (str): 入力画像のファイルパス
    """
    # チェスボードの内側のコーナー数を指定 (横の数, 縦の数)
    # 提供された画像は8x6のマス目なので、内側のコーナーは (8-1)x(6-1) = 7x5 となります。
    pattern_size = (7, 5)

    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' を読み込めませんでした。")
        print("ファイルパスが正しいか、画像が破損していないか確認してください。")
        return

    # 高速化と精度の向上のため、グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを探す
    # ret: コーナーが見つかったかどうか (True/False)
    # corners: 検出されたコーナーの座標
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # コーナーが検出された場合の処理
    if ret:
        print("✅ チェスボードのコーナーが正常に検出されました。")

        # さらに正確なコーナー位置を計算するための設定
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # サブピクセル精度でコーナー位置を最適化
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 検出したコーナーを元の画像に描画
        cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)

        # 結果をウィンドウに表示
        cv2.imshow('Checkerboard Detection Result', img)

        # 結果をファイルに保存
        output_path = 'checkerboard_detected.png'
        cv2.imwrite(output_path, img)
        print(f"検出結果を '{output_path}' に保存しました。")

        # 'q'キーが押されるまで表示を待つ
        print("結果ウィンドウで 'q' キーを押すと終了します。")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # コーナーが検出されなかった場合の処理
    else:
        print("❌ チェスボードのコーナーを検出できませんでした。")
        print("画像が鮮明であるか、または pattern_size の設定が正しいか確認してください。")

    # すべてのウィンドウを閉じる
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ステップ1で保存した画像ファイル名を指定してください
    input_image = r"C:\Users\Tomson\.vscode\PythonFile\props\checker_board\test_img.png"
    detect_checkerboard(input_image)