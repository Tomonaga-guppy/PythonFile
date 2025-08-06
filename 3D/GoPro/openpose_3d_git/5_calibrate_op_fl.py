"""
OpenPoseのキャリブレーションを実行
"""
import subprocess
from pathlib import Path
import os
import sys

# --- 設定項目 ---

# 成功したコマンドプロンプトのパス（作業ディレクトリ）
OPENPOSE_EXEC_DIR = Path(r"C:\Users\Tomson\openpose\build\x64\Release")

# 実行するプログラム名
OPENPOSE_EXECUTABLE_NAME = "calibration.exe"

# 画像とパラメータを保存するベースディレクトリ
BASE_PATH = Path(r"G:\gait_pattern\20250717_br")
IMAGE_BASE_PATH = BASE_PATH / "int_cali"
PARAMS_BASE_PATH = BASE_PATH / "camera_parameters"

# チェスボードの設定
SQUARE_SIZE_MM = "35"  # チェスボードのマスのサイズ (mm)
BOARD_SQUARES_X = "5"  # チェスボードの横の"内側の"角の数
BOARD_SQUARES_Y = "4"  # チェスボードの縦の"内側の"角の数
# 新しいフラグ用に "XxY" 形式の文字列を作成
grid_corners = f"{BOARD_SQUARES_X}x{BOARD_SQUARES_Y}"

# カメラの設定
CAMERA_NAMES = ["fl"]
CAMERA_SERIAL_PARAMS = str(0)

# 外部キャリブレーション用の画像フォルダ
EXTRINSIC_IMAGE_PATH = BASE_PATH / "ext_cali" / "cali_merge"

# --- 実行関数の定義 ---

def run_command(command_list):
    """コマンドをリストとして受け取り、出力をリアルタイムで表示しながら実行する"""
    print("--- [EXECUTING COMMAND] ---")
    print(f"Command: {subprocess.list2cmdline(command_list)}")
    print(f"Working Directory: {os.getcwd()}")
    print("---------------------------")

    # Popenを使ってプロセスを開始し、出力をパイプで受け取る
    # stderr=subprocess.STDOUT でエラー出力も標準出力にまとめる
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1 # 1行ずつバッファリングする
    )

    # プロセスの出力をリアルタイムで一行ずつ読み取って表示
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line) # print()より高速なsys.stdout.writeを使用
            sys.stdout.flush()

    # プロセスの終了を待つ
    return_code = process.wait()

    # 終了コードを確認して結果を表示
    if return_code == 0:
        print("\n[SUCCESS] コマンドが正常に実行されました。\n")
    else:
        print(f"\n[ERROR] コマンドの実行に失敗しました (終了コード: {return_code})\n")


def calibrate_intrinsics():
    """ステップ1: 各カメラの内部パラメータをキャリブレーション"""
    print(">>> ステップ1: 内部パラメータのキャリブレーションを開始します...")

    original_cwd = os.getcwd() # 元のディレクトリを保存
    try:
        # 成功したコマンドプロンプトと同じ場所に移動
        os.chdir(OPENPOSE_EXEC_DIR)

        for camera_name in CAMERA_NAMES:
            print(f"--- カメラ '{camera_name}' の処理を開始 ---")
            image_dir = IMAGE_BASE_PATH / camera_name / "6x5" / "cali_frames_all"
            output_param_folder = PARAMS_BASE_PATH

            if not image_dir.is_dir():
                print(f"[WARNING] 画像フォルダが見つかりません: {image_dir}. このカメラをスキップします。")
                continue

            # コマンドリストの先頭はプログラム名のみにする
            command_list = [
                OPENPOSE_EXECUTABLE_NAME,
                '--mode', '1',
                '--calibration_image_dir', str(image_dir),
                '--camera_parameter_folder', str(output_param_folder),
                '--grid_square_size_mm', SQUARE_SIZE_MM,
                '--grid_number_inner_corners', grid_corners,
                '--camera_serial_number', CAMERA_SERIAL_PARAMS
            ]
            run_command(command_list)
    finally:
        os.chdir(original_cwd) # 必ず元のディレクトリに戻る

if __name__ == "__main__":
    # 必要なディレクトリを作成
    IMAGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
    PARAMS_BASE_PATH.mkdir(parents=True, exist_ok=True)

    print(f"画像は '{IMAGE_BASE_PATH}' 以下に、")
    print(f"パラメータは '{PARAMS_BASE_PATH}' に保存されます。")
    print("-" * 20)

    # # ステップ1: 内部パラメータのキャリブレーション
    calibrate_intrinsics()

    print(">>> すべてのキャリブレーションプロセスが完了しました。")
