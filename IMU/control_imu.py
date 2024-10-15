import serial
import time
import struct
import ctypes
import csv
from datetime import datetime

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
    ser.read(100)
    ser.write(command)
    response = ser.read(3)
    # print(f"\n加速度レスポンス: {response}")

    if len(response) == 3:
        result = response[2]
        if result == 0:
            print("加速度の設定が正常に完了しました。")
        else:
            # print("加速度の設定に失敗しました。")
            pass
    else:
        # print("加速度のレスポンスを確認してください。")
        pass

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
    ser.read(100)
    ser.write(command)
    response = ser.read(3)
    # print(f"地磁気レスポンス: {response}")

    if len(response) == 3:
        result = response[2]
        if result == 0:
            print("地磁気の設定が正常に完了しました。")
        else:
            # print("地磁気の設定に失敗しました。")
            pass
    else:
        # print("地磁気のレスポンスを確認してください。")
        pass

def send_sync_signal(ser, level):
    # 外部拡張端子の出力レベルを設定（HighまたはLow）
    header = 0x9A
    cmd = 0x30  # 外部拡張端子設定コマンド
    data1 = int(level)  # 外部端子1を High (9) か Low (8) に設定
    data2 = 0x00  # 外部端子2は未使用
    data3 = 0x00  # 外部端子3は未使用
    data4 = 0x00  # 外部端子4は未使用

    # チェックサムの計算
    check = header ^ cmd ^ data1 ^ data2 ^ data3 ^ data4

    # コマンドリスト作成
    command = bytearray([header, cmd, data1, data2, data3, data4, check])

    # コマンド送信
    ser.write(command)

def set_device_time(ser):
    # ヘッダとコマンドコードを定義
    header = 0x9A
    cmd = 0x11  # 時刻設定コマンド

    # 現在の時刻を取得
    current_time = datetime.now()
    # 各パラメータの設定
    year = current_time.year - 2000  # 年 (2000年からの経過年数)
    month = current_time.month  # 月
    day = current_time.day  # 日
    hour = current_time.hour  # 時
    minute = current_time.minute  # 分
    second = current_time.second  # 秒
    millisecond = int(current_time.microsecond / 1000)  # ミリ秒に変換 (マイクロ秒を1000で割る)

    # チェックサムの計算
    # 年、月、日、時、分、秒、ミリ秒の各値を使って計算
    check = header ^ cmd ^ year ^ month ^ day ^ hour ^ minute ^ second
    check ^= (millisecond & 0xFF) ^ (millisecond >> 8)  # ミリ秒は2バイトなので下位と上位を分けて計算

    # コマンドリストを作成
    command = bytearray([header, cmd, year, month, day, hour, minute, second])
    command += struct.pack('<H', millisecond)  # ミリ秒を2バイトでパック
    command.append(check)  # チェックサムを最後に追加

    # コマンド送信
    ser.read(100)  # バッファをクリア
    ser.write(command)  # コマンドを送信

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
    ser.read(100)
    ser.write(command)
    response = ser.read(15)
    # print(f"\n計測開始レスポンス: {response}")

    if len(response) == 15:
        header = response[0]
        cmd_code = response[1]
        setting_status = response[2]
        start_year = 2000 + response[3]
        start_month = response[4]
        start_day = response[5]
        start_hour = response[6]
        start_minute = response[7]
        start_second = response[8]
        end_year = 2000 + response[9]
        end_month = response[10]
        end_day = response[11]
        end_hour = response[12]
        end_minute = response[13]
        end_second = response[14]

        # print(f"コマンドコード: {cmd_code}")
        # print(f"計測時刻設定状態: {setting_status}")
        print(f"開始時刻: {start_year}/{start_month}/{start_day} {start_hour}:{start_minute}:{start_second}")
        # print(f"終了時刻: {end_year}/{end_month}/{end_day} {end_hour}:{end_minute}:{end_second}")
        send_sync_pulse(ser)

    if len(response) != 15:
        print(f"Byte数が不正です。 {len(response)}")

def stop_measurement(ser):
    # 計測停止コマンド
    header = 0x9A
    cmd = 0x15  # 計測停止/計測予約クリアコマンド
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option

    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    # コマンド送信
    ser.read(100)
    ser.write(command)

    print(f"\n計測を停止しました。")

def get_entry_count(ser):
    # 計測データ記録エントリ件数取得コマンド
    header = 0x9A
    cmd = 0x36  # 計測データ記録エントリ件数取得コマンド
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option

    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    ser.read(100)
    # コマンド送信
    ser.write(command)
    response = ser.readline()

    # print(f"\nエントリ件数レスポンス: {response}")
    if response[1] == 0xB6:
        entry_count = response[2]
        print(f"有効なエントリの件数: {entry_count}")
        return entry_count
    else:
        print("エントリ件数の取得に失敗しました。")
        return 0

def read_entry(ser, entry_number):
    # 計測データ記録メモリ読み出しコマンドを作成
    header = 0x9A
    cmd = 0x39  # 計測データ記録メモリ読み出しコマンド
    entry = entry_number  # 読み出したいエントリ番号（1～80）

    # チェックサムの計算
    check = header ^ cmd ^ entry
    # コマンドリストを作成
    command = bytearray([header, cmd, entry, check])

    # コマンド送信
    ser.reset_input_buffer()  # 受信バッファをクリア
    ser.write(command)

    response = b''  # レスポンスデータを格納する変数

    time.sleep(1)  # データの受信を待つ
    while ser.in_waiting > 0:
        response += ser.read(100)
        time.sleep(0.01)

    # # レスポンスの内容を表示
    # print(f"レスポンス: {response}")


    accel_gyro_data = []
    geomagnetic_data = []

    i = 0
    while i < len(response):
        if response[i] == 0x9a:
            if response[i + 1] == 0x80 and i + 24 <= len(response):
                # # 加速度角速度データ
                timestamp = struct.unpack('<I', response[i + 2:i + 6])[0]
                accel_x = parse_3byte_signed(response[i + 6:i + 9])
                accel_y = parse_3byte_signed(response[i + 9:i + 12])
                accel_z = parse_3byte_signed(response[i + 12:i + 15])
                gyro_x = parse_3byte_signed(response[i + 15:i + 18])
                gyro_y = parse_3byte_signed(response[i + 18:i + 21])
                gyro_z = parse_3byte_signed(response[i + 21:i + 24])
                accel_gyro_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                i += 24
            elif response[i + 1] == 0x81 and i + 12 <= len(response):
                # 地磁気データ
                timestamp = struct.unpack('<I', response[i + 2:i + 6])[0]
                geo_x = parse_3byte_signed(response[i + 6:i + 9])
                geo_y = parse_3byte_signed(response[i + 9:i + 12])
                geo_z = parse_3byte_signed(response[i + 12:i + 15])
                geomagnetic_data.append([timestamp, geo_x, geo_y, geo_z])
                i += 15
            else:
                # 予期しないデータブロックの場合は次に進む
                i += 1
        else:
            # 予期しないデータの場合は次に進む
            i += 1

    print(f"内部メモリから全てのデータを取得しました。")
    return accel_gyro_data, geomagnetic_data

def save_to_csv(accel_gyro_data, geomagnetic_data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Timestamp", "X", "Y", "Z", "Gyro_X", "Gyro_Y", "Gyro_Z"])

        # 書き込み: 加速度・角速度データ
        for data in accel_gyro_data:
            writer.writerow(["ags"] + data)

        # 書き込み: 地磁気データ
        for data in geomagnetic_data:
            writer.writerow(["geo"] + data)


def read_sensor_data(ser):
    try:
        print("計測中")
        while True:
            pass
            # # データを1バイトずつ読み取る
            # str = ser.read(1)

            # # データが受信できていない場合はタイムアウト
            # if len(str) == 0:
            #     print("データが受信されていません。")
            #     continue

            # # ヘッダが 0x9A であることを確認
            # if ord(str) == 0x9A:
            #     # コマンドの種類を読み取る
            #     cmd = ser.read(1)

            #     if len(cmd) == 0:
            #         continue

            #     if ord(cmd) == 0x80:  # 加速度・角速度データ
            #         send_sync_pulse(ser)  # 同期パルスを送信

            #         # 4バイトのTickTimeを読み取る
            #         tick_time = ser.read(4)
            #         tick_time_ms = struct.unpack('<I', tick_time)[0]
            #         print(f"TickTime: {tick_time_ms} ms")

            #         # 加速度データ X, Y, Z を読み取る (各3バイト)
            #         acc_x = read_3byte_signed(ser)
            #         acc_y = read_3byte_signed(ser)
            #         acc_z = read_3byte_signed(ser)
            #         print(f"加速度データ (X, Y, Z): {acc_x} mg, {acc_y} mg, {acc_z} mg")

            #         # 角速度データ X, Y, Z を読み取る (各3バイト)
            #         gyro_x = read_3byte_signed(ser)
            #         gyro_y = read_3byte_signed(ser)
            #         gyro_z = read_3byte_signed(ser)
            #         print(f"角速度データ (X, Y, Z): {gyro_x} dps, {gyro_y} dps, {gyro_z} dps")

            #     elif ord(cmd) == 0x81:  # 地磁気データ
            #         # 4バイトのTickTimeを読み取る
            #         tick_time = ser.read(4)
            #         tick_time_ms = struct.unpack('<I', tick_time)[0]
            #         print(f"TickTime: {tick_time_ms} ms")

            #         # 地磁気データ X, Y, Z を読み取る (各3バイト)
            #         mag_x = read_3byte_signed(ser)
            #         mag_y = read_3byte_signed(ser)
            #         mag_z = read_3byte_signed(ser)
            #         print(f"地磁気データ (X, Y, Z): {mag_x} uT, {mag_y} uT, {mag_z} uT")

    except KeyboardInterrupt:
        pass

def parse_3byte_signed(data):
    """3バイトの符号付きデータを整数として解釈する"""
    if len(data) != 3:
        raise ValueError("データ長が3バイトでありません")

    # 3バイトのデータを4バイトに変換（符号拡張）
    if data[2] & 0x80:  # 負の数の場合
        data4 = 0xFF  # 負の数の場合、符号拡張として 0xFF
    else:
        data4 = 0x00  # 正の数の場合、符号拡張として 0x00

    # 4バイトの整数として結合
    value = data[0] + (data[1] << 8) + (data[2] << 16) + (data4 << 24)

    return ctypes.c_int(value).value

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

def clear_measurement_data(ser):
    # 計測データ記録クリアコマンド
    header = 0x9A
    cmd = 0x35  # 計測データ記録クリアコマンド
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option
    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    # コマンド送信
    ser.reset_input_buffer()
    ser.write(command)

def send_sync_pulse(ser):
    send_sync_signal(ser, level=8)  # Low出力 (8)
    time.sleep(0.0001)  # 少しの間Lowに保持
    send_sync_signal(ser, level=9)  # High出力 (9)

def main():
    # シリアルポートの設定
    ser = serial.Serial()
    port = "COM" + input("接続するポート番号を入力:COM")
    ser.port = port  # デバイスに応じて変更
    ser.timeout = 1.0
    ser.baudrate = 115200

    # シリアルポートを開く
    ser.open()

    set_device_time(ser)

    # 加速度、角速度、地磁気の計測設定を行う
    configure_accelgyro(ser)
    configure_magnetic(ser)

    # 計測を開始する
    start_measurement(ser)
    print("計測設定が完了し、計測を開始しました。\n")

    # 加速度、角速度、地磁気のデータを読み取るループを開始
    try:
        read_sensor_data(ser)
    finally:
        stop_measurement(ser)

        while ser.in_waiting > 0:  # バッファ内のデータをクリア 重要！
            ser.readline()
            time.sleep(0.01)

        # 最新の計測記録を取得し、CSVに保存
        entry_count = get_entry_count(ser)
        if entry_count > 0:
            accel_gyro_data, geomagnetic_data = read_entry(ser, entry_count)
            # CSVに保存
            save_to_csv(accel_gyro_data, geomagnetic_data, 'sensor_data.csv')
            print("計測データをCSVに保存しました。")

        # 計測データの記録をクリア
        # clear_measurement_data(ser)

        send_sync_signal(ser, level=9)  # 外部端子1をHigh出力


        # シリアルポートを閉じる
        ser.close()
        print("計測を終了し、シリアルポートを閉じました。")

if __name__ == '__main__':
    main()
