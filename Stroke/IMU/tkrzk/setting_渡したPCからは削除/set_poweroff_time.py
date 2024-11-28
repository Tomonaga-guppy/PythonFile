import serial

def set_auto_power_off(ser, time):
    # ブザー音量設定コマンド
    header = 0x9A
    cmd = 0x50  # ブザー音量設定コマンド
    parameter = time  # オートパワーオフ時間（0~20）0:無効、1~20:時間（分）

    # チェックサムの計算
    check = header ^ cmd ^ parameter
    # コマンドリスト作成
    command = bytearray([header, cmd, parameter, check])

    # コマンド送信
    ser.read(100)
    ser.write(command)
    response = ser.read(3)  # コマンドレスポンスは2バイト（HeaderとCommand Code）

    # レスポンスの確認
    if len(response) == 3 and response[1] == 0x8F:
        print(f"オートパワーオフが設定されました。時間: {time}")
    else:
        print("レスポンスが正しくありません。オートパワーオフ設定に失敗しました。response:", response)

# 使用例
ser = serial.Serial()
ser.port = "COM" + input("シリアルポートを入力してください:COM")
ser.timeout = 1.0
ser.baudrate = 115200

try:
    ser.open()
    print(f"シリアルポートを開きました。 {ser.port}")
    set_auto_power_off(ser, 0)  # オートパワーオフを0:無効に設定
except Exception as e:
    print(f"シリアルポートを開くことができませんでした。 ({ser.port})\nエラー: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("シリアルポートを閉じました。")
