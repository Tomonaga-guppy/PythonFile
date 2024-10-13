import serial
import time
import struct
import ctypes

def configure_accelgyro(ser):
    # 加速度、角速度計測の設定
    header = 0x9A
    cmd = 0x16  # 加速度計測設定コマンド
    data1 = 0x0A  # 計測周期 10ms
    data2 = 0x01  # 送信データの平均回数 1回
    data3 = 0x01  # 記録データの平均回数 1回

    # チェックサムの計算
    check = header ^ cmd ^ data1 ^ data2 ^ data3

    # コマンドリストを作成
    command = bytearray([header, cmd, data1, data2, data3, check])

    # コマンド送信
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(100)
    print(f"加速度レスポンス: {response}")

def configure_magnetic(ser):
    # 地磁気計測の設定
    header = 0x9A
    cmd = 0x18  # 地磁気計測設定コマンド
    data1 = 0x0A  # 計測周期 10ms
    data2 = 0x01  # 送信データの平均回数 1回
    data3 = 0x01  # 記録データの平均回数 1回

    # チェックサムの計算
    check = header ^ cmd ^ data1 ^ data2 ^ data3

    # コマンドリストを作成
    command = bytearray([header, cmd, data1, data2, data3, check])

    # コマンド送信
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(100)
    print(f"地磁気レスポンス: {response}")

def start_measurement(ser):
    # 計測開始コマンド
    header = 0x9A
    cmd = 0x13  # 計測開始/予約コマンド
    smode = 0x00  # 相対時間指定
    syear = 0x00  # 開始年（2000年からの経過年数）
    smonth = 0x01  # 開始月
    sday = 0x01  # 開始日
    shour = 0x00  # 開始時
    smin = 0x00  # 開始分
    ssec = 0x00  # 開始秒
    emode = 0x00  # 相対時間指定
    eyear = 0x00  # 終了年（2000年からの経過年数）
    emonth = 0x01  # 終了月
    eday = 0x01  # 終了日
    ehour = 0x00  # 終了時
    emin = 0x00  # 終了分
    esec = 0x00  # 終了秒

    # チェックサムの計算
    check = header ^ cmd ^ smode ^ syear ^ smonth ^ sday ^ shour ^ smin ^ ssec
    check ^= emode ^ eyear ^ emonth ^ eday ^ ehour ^ emin ^ esec

    # コマンドリスト作成
    command = bytearray([header, cmd, smode, syear, smonth, sday, shour, smin, ssec,
                         emode, eyear, emonth, eday, ehour, emin, esec, check])

    # コマンド送信
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(100)
    print(f"計測開始レスポンス: {response}")

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
    ser.port = "COMfaefa"  # デバイスに応じて変更
    ser.timeout = 1.0
    ser.baudrate = 115200

    # シリアルポートを開く
    ser.open()

    # 加速度、角速度、地磁気の計測設定を行う
    configure_accelgyro(ser)
    configure_magnetic(ser)

    # 計測を開始する
    start_measurement(ser)

    print("全てのセンサーの計測設定が完了し、計測を開始しました。")

    # 加速度、角速度、地磁気のデータを読み取るループを開始
    try:
        read_sensor_data(ser)
    finally:
        # シリアルポートを閉じる
        ser.close()

if __name__ == '__main__':
    main()
