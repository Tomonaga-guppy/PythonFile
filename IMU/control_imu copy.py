import serial
import time
import struct
import ctypes

def read_sensor_data(ser):
    while True:
        # データを1バイトずつ読み取る
        str = ser.read(1)

        # データが受信できていない場合はタイムアウト
        if len(str) == 0:
            print("データが受信されていません。")
            continue

        # ヘッダが 0x9A であることを確認
        if ord(str) == 0x9A:
            # コマンドの種類を読み取る
            cmd = ser.read(1)

            if len(cmd) == 0:
                continue

            if ord(cmd) == 0x80:  # 加速度・角速度データ
                # 4バイトのTickTimeを読み取る
                tick_time = ser.read(4)
                tick_time_ms = struct.unpack('<I', tick_time)[0]
                print(f"TickTime: {tick_time_ms} ms")

                # 加速度データ X, Y, Z を読み取る (各3バイト)
                acc_x = read_3byte_signed(ser)
                acc_y = read_3byte_signed(ser)
                acc_z = read_3byte_signed(ser)
                print(f"加速度データ (X, Y, Z): {acc_x * 0.1} mg, {acc_y * 0.1} mg, {acc_z * 0.1} mg")

                # 角速度データ X, Y, Z を読み取る (各3バイト)
                gyro_x = read_3byte_signed(ser)
                gyro_y = read_3byte_signed(ser)
                gyro_z = read_3byte_signed(ser)
                print(f"角速度データ (X, Y, Z): {gyro_x * 0.01} dps, {gyro_y * 0.01} dps, {gyro_z * 0.01} dps")

            elif ord(cmd) == 0x81:  # 地磁気データ
                # 4バイトのTickTimeを読み取る
                tick_time = ser.read(4)
                tick_time_ms = struct.unpack('<I', tick_time)[0]
                print(f"TickTime: {tick_time_ms} ms")

                # 地磁気データ X, Y, Z を読み取る (各3バイト)
                mag_x = read_3byte_signed(ser)
                mag_y = read_3byte_signed(ser)
                mag_z = read_3byte_signed(ser)
                print(f"地磁気データ (X, Y, Z): {mag_x * 0.1} uT, {mag_y * 0.1} uT, {mag_z * 0.1} uT")

            else:
                pass
                # print(f"未対応のコマンドを受信: {hex(ord(cmd))}")

        else:
            pass
            # print(f"不正なヘッダ: {hex(ord(str))}")

def read_3byte_signed(ser):
    """3バイトの符号付きデータを読み取って、符号付き整数として返す"""
    data = ser.read(3)
    if len(data) != 3:
        return 0

    # 3バイトのデータを4バイトに変換（符号拡張）
    if data[2] & 0x80:  # 負の数の場合
        data4 = b'\xFF'
    else:
        data4 = b'\x00'

    value = data[0] + (data[1] << 8) + (data[2] << 16) + (ord(data4) << 24)
    return ctypes.c_int(value).value

def main():
    # シリアルポートの設定
    ser = serial.Serial()
    ser.port = "COM7"  # デバイスに応じて変更
    ser.timeout = 1.0
    ser.baudrate = 115200

    # シリアルポートを開く
    ser.open()

    # 加速度、角速度、地磁気のデータを読み取るループを開始
    try:
        read_sensor_data(ser)
    finally:
        # シリアルポートを閉じる
        ser.close()

if __name__ == '__main__':
    main()


# samplingrateは100くらいがよくある

