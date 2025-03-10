import serial
import time
import struct
import ctypes
import csv
import threading
from datetime import datetime

#各スレッドで終了をするためのイベント
stop_event = threading.Event()

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

def configure_accelgyro(ser, port):
    # 成功したかどうかのフラグを作成
    flag = False

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
            print(f"加速度の設定が正常に完了しました {port}")
        flag = True
    else:
        print(f"加速度の設定に失敗しました {port}")
        flag = False

    return flag

def configure_magnetic(ser, port):
    # 成功したかどうかのフラグを作成
    flag = False

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
            print(f"地磁気の設定が正常に完了しました {port}")
            flag = True
    else:
        print(f"地磁気の設定に失敗しました {port}")
        flag = False

    return flag

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

def start_measurement(ser, port, sync_port):
    # 成功したかどうかのフラグを作成
    flag = False

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

        print(f"開始時刻: {start_year}/{start_month}/{start_day} {start_hour}:{start_minute}:{start_second} {port}")
        flag = True

        if port == sync_port:
            send_sync_pulse(ser)

    else:
        print(f"時刻設定時のレスポンスが正しくありません。 {(response)}")
        flag = False

    return flag


def stop_measurement(ser, port):
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
    print(f"\n計測を停止しました {port}")

def read_sensor_data(ser, port):  #元々は値の表示用だったけどいまは停止イベントを受け取るだけ
    try:
        flag = True
        print(f"計測中 {port}")
        while not stop_event.is_set():
            pass
    except KeyboardInterrupt:
        flag = False

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

def send_sync_pulse(ser):
    send_sync_signal(ser, level=8)  # Low出力 (8)
    time.sleep(0.0001)  # 少しの間Lowに保持
    send_sync_signal(ser, level=9)  # High出力 (9)

def get_entry_count(ser, port):
    # 計測データ記録エントリ件数取得コマンド
    header = 0x9A
    cmd = 0x36  # 計測データ記録エントリ件数取得コマンド
    option = 0x00  # 固定

    # チェックサムの計算
    check = header ^ cmd ^ option

    # コマンドリスト作成
    command = bytearray([header, cmd, option, check])

    # コマンド送信
    ser.read(100)
    ser.write(command)
    response = ser.readline()

    time.sleep(0.1)  # データの受信を待つ
    print(f"\nエントリ件数レスポンス: {response}")
    if response[1] == 0xB6:
        entry_count = response[2]
        print(f"有効なエントリの件数: {entry_count} {port}")
        return entry_count
    else:
        print("エントリ件数の取得に失敗しました {port}")
        return 0

def read_entry(ser, entry_number, port):
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

    print(f"内部メモリから全てのデータを取得しました {port}")

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
    response = ser.read(2)  # コマンドレスポンスは 2 バイト（Header と Command Code）

    if len(response) == 2 and response[1] == 0x8F:
        result = ser.read(1)  # コマンド受付結果を読む
        if result == b'\x00':
            print("計測データの記録クリアが正常に完了しました。")
        else:
            print("計測データの記録クリアに失敗しました。")
    else:
        print("レスポンスが正しくありません。")



def run_imu_on_port(port, barrier, sync_port):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200
    try:
        ser.open()
    except Exception as e:
        print(f"シリアルポートを開くことができませんでした。 ({port})")
        return

    print(f"{port} is open")

    try :
        # 時刻の設定
        set_device_time(ser)
        # 加速度の設定
        flag = configure_accelgyro(ser, port)
        if not flag:
            raise Exception(f"加速度の設定に失敗しました。 計測を終了します({port})")
        # 地磁気の設定
        flag = configure_magnetic(ser, port)
        if not flag:
            raise Exception(f"地磁気の設定に失敗しました。 計測を終了します({port})")

        barrier.wait(timeout=3)  # 他のスレッドと同期

        # 計測を開始する
        flag = start_measurement(ser, port, sync_port)
        if not flag:
            raise Exception(f"計測の開始に失敗しました。 計測を終了します({port})")

        print(f"計測設定が完了し、計測を開始しました。 ({port})\n")

        # 加速度、角速度、地磁気のデータを読み取るループを開始
        flag = read_sensor_data(ser, port)
        if not flag:
            raise Exception(f"計測を終了しました。 ({port})")

    except Exception:
        stop_measurement(ser, port)
        ser.close

def read_memory(port, sync_port):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200
    ser.open()

    if port == sync_port:
        while ser.in_waiting > 0:  # バッファ内のデータをクリア 重要！
            ser.readline()
            time.sleep(0.01)

    # 最新の計測記録を取得し、CSVに保存
    entry_count = get_entry_count(ser, port)
    if entry_count > 0:
        accel_gyro_data, geomagnetic_data = read_entry(ser, entry_count, port)
        # CSVに保存
        save_to_csv(accel_gyro_data, geomagnetic_data, f'sensor_data_{port}.csv')
        print(f"計測データをCSVに保存しました。 ({port})")

    # 計測データの記録をクリア
    # clear_measurement_data(ser)

    send_sync_signal(ser, level=9)  # 外部端子1をHigh出力

    # シリアルポートを閉じる
    ser.close()
    print(f"計測を終了し、シリアルポートを閉じました。 ({port})")

def main():
    sync_port = "COM" + input("同期用IMU AP09181497 のポート番号を入力:COM")
    subject_port = "COM" + input("患者用IMU RP1B001592 のポート番号を入力:COM")
    therapist_port = "COM" + input("療法士用IMU RPRP1B001593 のポート番号を入力:COM")
    ports = [sync_port, subject_port, therapist_port]
    threads = []
    threads_post = []
    barrier = threading.Barrier(len(ports))  # スレッド間の同期のためのバリアを作成

    ### 計測処理 ########################################################################################
    # 各シリアルポートに対してスレッドを作成し、実行
    for port in ports:
        thread = threading.Thread(target=run_imu_on_port, args=(port,barrier, sync_port))
        threads.append(thread)

    try:
        for thread in threads:
            thread.start()
       # スレッドが実行中はメインスレッドはそのまま継続する
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.00001)  # 少し待機しながらスレッドの終了を待つ
    except KeyboardInterrupt:
        stop_event.set()  # 終了のイベントをセット

        for thread in threads:  #全てのスレッドが終了するまで待機
            thread.join()

        print("計測を終了しました")

    # ### 内部メモリの書き出し##############################################################################
    # print("計測を終了しました。IMUメモリの書き出しを行います")
    # for port in ports:
    #     thread_post = threading.Thread(target=read_memory, args=(port,sync_port,))
    #     threads_post.append(thread_post)

    # for thread_post in threads_post:
    #     thread_post.start()

    # print("メモリの書き出しを行っています 少々お待ちください")

    # for thread_post in threads_post:
    #     thread_post.join()

    # print("\n全てのIMUで書き出しを終了しました\n ")

if __name__ == '__main__':
    main()
