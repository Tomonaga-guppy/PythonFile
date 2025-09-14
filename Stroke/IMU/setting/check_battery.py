import serial
import struct

def get_battery_status(ser):
    """
    IMUセンサーにバッテリー状態取得コマンドを送信し、結果を表示する。
    """
    if not ser.is_open:
        print("シリアルポートが開かれていません。")
        return

    # --- check_volume.pyからの変更点 1 ---
    # コマンドを「バッテリ状態取得」用に変更
    header = 0x9A
    cmd = 0x3B  # バッテリ状態取得コマンド (仕様書 4.44)
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option
    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    ser.read(100) # コマンド送信前にバッファをクリア
    ser.write(command)
    
    # --- check_volume.pyからの変更点 2 ---
    # レスポンスの長さを5バイトに変更
    # Header(1), Command Code(1), 電圧(2), 残量(1) の合計5バイト
    response = ser.read(5)

    # レスポンスの確認と処理
    # --- check_volume.pyからの変更点 3 ---
    # レスポンスコードとデータ解析処理をバッテリー状態取得用に変更
    if len(response) == 5 and response[1] == 0xBB: # レスポンスコードは 0xBB
        # パラメータ部分を抽出
        params = response[2:]
        
        # 電圧値 (2バイト、リトルエンディアン) をデコード
        # struct.unpack('<H', ...) を使って2バイトをunsigned shortとして解釈
        voltage_raw = struct.unpack('<H', params[0:2])[0]
        voltage = voltage_raw / 100.0  # 仕様書より 0.01V 単位
        
        # バッテリー残量 (1バイト)
        level = params[2] # 1%単位

        print(f"バッテリー電圧：{voltage:.2f} V")
        print(f"バッテリー残量：{level} %")
    else:
        print("レスポンスが正しくありません。")
        print("受信データ：", " ".join(f"{b:02x}" for b in response))

# --- 以下は check_volume.py と同様の接続処理 ---
# 使用例
ser = serial.Serial()
# COMポートの番号をユーザーに入力させる
port_num = input("シリアルポート番号を入力してください (例: 3): ")
ser.port = "COM" + port_num
ser.timeout = 2.0  # タイムアウトを少し長めに設定
ser.baudrate = 115200

try:
    ser.open()
    print(f"シリアルポートを開きました。 ({ser.port})")
    get_battery_status(ser)
except serial.SerialException as e:
    print(f"シリアルポートを開くことができませんでした。({ser.port})\nエラー: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("シリアルポートを閉じました。")
