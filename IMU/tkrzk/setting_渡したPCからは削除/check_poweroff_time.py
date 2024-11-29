import serial

def get_poweroff_time_setting(ser):
    if not ser.is_open:
        print("シリアルポートが開かれていません。")
        return

    # ブザー音量設定取得コマンド
    header = 0x9A
    cmd = 0x51  # ブザー音量設定取得コマンド
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option
    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    # コマンド送信
    ser.read(100)
    ser.write(command)
    response = ser.read(3)  # レスポンスは3バイト（Header、Command Code、ブザー音量）

    # レスポンスの確認と処理
    if len(response) == 3 and response[1] == 0xD1:
        volume = response[2]
        print(f"オートパワーオフ時間: {volume}")  # オートパワーオフ時間（0~20）0:無効、1~20:時間（分）

    else:
        print("レスポンスが正しくありません。")
        print("レスポンス：", response)

# 使用例
ser = serial.Serial()
ser.port = "COM" + input("シリアルポートを入力してください:COM")
ser.timeout = 1.0
ser.baudrate = 115200

try:
    ser.open()
    print(f"シリアルポートを開きました。 ({ser.port})")
    get_poweroff_time_setting(ser)
except Exception as e:
    print(f"シリアルポートを開くことができませんでした。 {ser.port}\nエラー: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("シリアルポートを閉じました。")
