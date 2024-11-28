import serial

def set_buzzer_volume(ser, volume):
    # volumeが0, 1, 2の範囲内であることを確認
    if volume not in [0, 1, 2]:
        print("エラー: 音量は0（消音）、1（小）、2（大）のいずれかでなければなりません。")
        return

    # ブザー音量設定コマンド
    header = 0x9A
    cmd = 0x32  # ブザー音量設定コマンド
    parameter = volume  # ブザー音量（0, 1, 2）

    # チェックサムの計算
    check = header ^ cmd ^ parameter
    # コマンドリスト作成
    command = bytearray([header, cmd, parameter, check])

    # コマンド送信
    ser.read(100)
    ser.write(command)
    response = ser.read(2)  # コマンドレスポンスは2バイト（HeaderとCommand Code）
    command_code = 0x8F  # ブザー音量設定コマンドのコマンドコード

    # レスポンスの確認
    if len(response) == 2 and response[1] == command_code:
        print(f"ブザー音量が設定されました。音量レベル: {volume}")
    else:
        print("レスポンスが正しくありません。音量設定に失敗しました。response:", response)

# 使用例
ser = serial.Serial()
ser.port = "COM" + input("シリアルポートを入力してください:COM")
ser.timeout = 1.0
ser.baudrate = 115200

try:
    ser.open()
    print(f"シリアルポートを開きました。 {ser.port}")
    # 音量を設定（0: 消音、1: 小、2: 大）
    set_buzzer_volume(ser, 2)  # 音量を「大」に設定
except Exception as e:
    print(f"シリアルポートを開くことができませんでした。 ({ser.port})\nエラー: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("シリアルポートを閉じました。")
