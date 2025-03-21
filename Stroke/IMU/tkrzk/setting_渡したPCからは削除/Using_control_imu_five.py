import serial
import time
import struct
import ctypes
import csv
import multiprocessing
from datetime import datetime
from pathlib import Path
import json

#各スレッドで終了をするためのイベント
stop_event = multiprocessing.Event()

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
    data1 = 0x0A  # 計測周期 10ms  100Hz
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
            # print(f"加速度の設定が正常に完了しました {port}")
            pass
        flag = True
    else:
        # print(f"加速度の設定に失敗しました {port}")
        flag = False

    return flag

def configure_magnetic(ser, port):
    # 成功したかどうかのフラグを作成
    flag = False

    # 地磁気計測の設定
    header = 0x9A
    cmd = 0x18  # 地磁気計測設定コマンド
    data1 = 0x0A  # 計測周期 10ms 100Hz
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
            # print(f"地磁気の設定が正常に完了しました {port}")
            pass
            flag = True
    else:
        # print(f"地磁気の設定に失敗しました {port}")
        flag = False

    return flag

# def send_sync_signal(ser, level):
#     # 外部拡張端子の出力レベルを設定（HighまたはLow）
#     header = 0x9A
#     cmd = 0x30  # 外部拡張端子設定コマンド
#     data1 = int(level)  # 外部端子1を High (9) か Low (8) に設定
#     data2 = 0x00  # 外部端子2は未使用
#     data3 = 0x00  # 外部端子3は未使用
#     data4 = 0x00  # 外部端子4は未使用

#     # チェックサムの計算
#     check = header ^ cmd ^ data1 ^ data2 ^ data3 ^ data4

#     # コマンドリスト作成
#     command = bytearray([header, cmd, data1, data2, data3, data4, check])

#     # コマンド送信
#     ser.write(command)

def start_measurement(ser, port):
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
    start_time = ""

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

        start_time = f"{start_year}-{start_month}-{start_day}-{start_hour}-{start_minute}-{start_second}"
        print(f"{port}の計測が開始されました : {start_time}")
        flag = True

    else:
        print(f"時刻設定時のレスポンスが正しくありません。 {(response)}")
        flag = False

    return flag, start_time


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
    print(f"計測を停止しました {port}")



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
    # print(f"エントリ件数レスポンス: {response} {port}")
    if response[1] == 0xB6:
        entry_count = response[2]
        # print(f"有効なエントリの件数: {entry_count} {port}")
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
    # ser.reset_input_buffer()  # 受信バッファをクリア

    while ser.in_waiting > 0:  # バッファ内のデータをクリア 重要！
        ser.read(100)
        time.sleep(0.01)

    ser.write(command)

    response = b''  # レスポンスデータを格納する変数

    time.sleep(1)  # データの受信を待つ
    # print(f"受信バッファの初期データ数: {port}, {ser.in_waiting}")
    # ser.in_waitingは受信データのバイト数を返す
    while ser.in_waiting > 0:
        read_data = ser.read(min(ser.in_waiting, 128))
        response += read_data
        time.sleep(0.01)


    # time.sleep(1)  # データの受信を待つ
    # while ser.in_waiting > 0:
    #     response += ser.read(100)
    #     time.sleep(0.01)

    # # レスポンスの内容を表示
    # print(f"レスポンス: {response}")
    # print(f"内部メモリから全てのデータを取得しました {port}")

    accel_gyro_data = []
    geomagnetic_data = []

    # print(f"{port}の合計データ数（バイト）: {len(response)}")

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

    del response

    return accel_gyro_data, geomagnetic_data

def save_to_csv(accel_gyro_data, geomagnetic_data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 列名を指定
        writer.writerow([
            "Type", "Timestamp_Acc", "Acc_X 0.1[mG]", "Acc_Y 0.1[mG]", "Acc_Z 0.1[mG]", "Gyro_X 0.01[dps]", "Gyro_Y 0.01[dps]", "Gyro_Z 0.01[dps]",
            "Type", "Timestamp_Mag", "Mag_X 0.1[μT]", "Mag_Y 0.1[μT]", "Mag_Z 0.1[μT]"
        ])

        # 加速度・角速度データと地磁気データの数が同じと仮定
        max_len = max(len(accel_gyro_data), len(geomagnetic_data))

        for i in range(max_len):
            # accel_gyro_dataがある場合はそのデータを使用し、なければ空欄を埋める
            if i < len(accel_gyro_data):
                ag_data = ["ags"] + accel_gyro_data[i]
            else:
                ag_data = ["ags", "", "", "", "", "", "", ""]

            # geomagnetic_dataがある場合はそのデータを使用し、なければ空欄を埋める
            if i < len(geomagnetic_data):
                geo_data = ["geo"] + geomagnetic_data[i]
            else:
                geo_data = ["geo", "", "", "", ""]

            # データを横に結合してCSVに書き込み
            writer.writerow(ag_data + geo_data)

def parse_3byte_signed(data):
    """3バイトの符号付きデータを整数として解釈する"""
    if len(data) != 3:
        print(f"データ長: {data}")
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
            # print("計測データの記録クリアが正常に完了しました。")
            pass
        else:
            # print("計測データの記録クリアに失敗しました。")
            pass
    else:
        print("レスポンスが正しくありません。")



def run_imu_on_port(port, barrier, start_queue):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200
    start_time = ""
    try:
        ser.open()
    except Exception as e:
        print(f"シリアルポートを開くことができませんでした。 ({port})")
        return

    print(f"{port} のポートを開きました。")

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

        print(f"{port} IMUの設定が完了しました。")

        barrier.wait(timeout=30)  # 他のスレッドと同期

        # 計測を開始する
        flag, start_time = start_measurement(ser, port)
        if not flag:
            raise Exception(f"計測の開始に失敗しました。 計測を終了します({port})")

        try:
            while not stop_event.is_set():  # 終了イベントがセットされるまで待機
                time.sleep(0.01)
        except KeyboardInterrupt:
            raise Exception

    except Exception:
        stop_measurement(ser, port)
        start_queue.put((port, start_time))
        ser.close()

def read_save_memory(port, port_dict, start_time_dict, save_path):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200
    ser.open()

    while ser.in_waiting > 0:  # バッファ内のデータをクリア 重要！
        ser.readline()
        time.sleep(0.01)

    # 最新の計測記録を取得し、CSVに保存
    entry_count = get_entry_count(ser, port)
    if entry_count > 0:
        accel_gyro_data, geomagnetic_data = read_entry(ser, entry_count, port)
        # CSVに保存
        save_to_csv(accel_gyro_data, geomagnetic_data, save_path)
        del accel_gyro_data, geomagnetic_data
        # print(f"計測データをCSVに保存しました。 ({port})")

    # 計測データの記録をクリア
    if entry_count > 40:
        clear_measurement_data(ser)

    # シリアルポートを閉じる
    ser.close()
    # print(f"計測を終了し、シリアルポートを閉じました。 ({port})")

def main(ports, port_dict, save_dir):
    #計測用の変数を初期化
    threads = []
    barrier = multiprocessing.Barrier(len(ports))  # スレッド間の同期のためのバリアを作成
    start_queue = multiprocessing.Queue()  # 開始時刻を取得するためのキューを作成
    start_time_dict = {}

    ### 計測処理 ########################################################################################
    # 各シリアルポートに対してスレッドを作成し、実行
    for port in ports:
        thread = multiprocessing.Process(target=run_imu_on_port, args=(port, barrier, start_queue, ))
        threads.append(thread)

    try:
        for thread in threads:
            thread.start()

        # start = time.time()

       # スレッドが実行中はメインスレッドはそのまま継続する
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.00001)  # 少し待機しながらスレッドの終了を待つ
    except KeyboardInterrupt:
        stop_event.set()  # 終了のイベントをセット

        for thread in threads:  #全てのスレッドが終了するまで待機
            thread.join()

        # end = time.time()
        # print(f"計測時間: {end - start}秒")
        # print("計測を終了しました")

        while not start_queue.empty():
            port, start_time = start_queue.get()
            start_time_dict[port] = start_time

    ### 内部メモリの書き出し##############################################################################

    print("メモリの書き出しを5台分行います 少々お待ちください")
    start = time.time()
    for i, port in enumerate(ports):
        save_path = save_dir / f'sensor_data_{port_dict[port]}_{start_time_dict[port]}.csv'
        read_save_memory(port, port_dict, start_time_dict, save_path)
        print(f"{i+1}/{len(ports)} 計測データを{save_path}に保存しました {port}")

    end = time.time()
    # print(f"保存にかかった時間: {end - start}秒")
    print("\n全てのIMUで書き出しを終了しました ウィンドウを閉じて終了してください")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 入力はメインプロセスでのみ実行
    if multiprocessing.current_process().name == "MainProcess":
        # root_dir = Path(r"C:\Users\zutom\OneDrive\デスクトップ\IMU\data")
        root_dir = Path(r"C:\Users\BRLAB\Desktop\data\IMU")
        reuse_port_flag = "a"
        while reuse_port_flag != "y" and reuse_port_flag != "n":
            reuse_port_flag = input("前回のポート番号を再利用しますか？(y/n): ")
            if reuse_port_flag == "y":  #前回のポート番号を読み込んで再利用
                port_dict_file = root_dir / "port_dict.json"
                try:
                    with open(port_dict_file, "r") as file:
                        port_dict = json.load(file)
                    ports = list(port_dict.keys())
                    sync_port = ports[0]
                    sub_port = ports[1]
                    thera_port = ports[2]
                    thera_rhand_port = ports[3]
                    thera_lhand_port = ports[4]
                except json.decoder.JSONDecodeError:
                    print("前回のポート番号が正常に読み込めませんでした。nを選択してポート番号を入力してください")
                    pass

            elif reuse_port_flag == "n":  #新たにポート番号を入力
                # tkzkの場合
                sync_port_num = input("同期用IMU AP09181356 のポート番号を入力:COM")
                sub_port_num = input("患者腰用IMU AP09182459 のポート番号を入力:COM")
                thera_port_num = input("療法士腰用IMU AP09182460 のポート番号を入力:COM")
                thera_rhand_port_num = input("療法士右手用IMU AP09182461 のポート番号を入力:COM")
                thera_lhand_port_num = input("療法士左手用IMU AP09182462 のポート番号を入力:COM")

                # otの場合
                # sync_port_num = input("同期用IMU AP09181497 のポート番号を入力:COM")
                # sub_port_num = input("患者腰用IMU AP09181498 のポート番号を入力:COM")
                # thera_port_num = input("療法士腰用IMU AP09181354 のポート番号を入力:COM")
                # thera_rhand_port_num = input("療法士右手用IMU AP09181355 のポート番号を入力:COM")
                # thera_lhand_port_num = input("療法士左手用IMU AP09181357 のポート番号を入力:COM")

                sync_port = "COM" + sync_port_num
                sub_port = "COM" + sub_port_num
                thera_port = "COM" + thera_port_num
                thera_rhand_port = "COM" + thera_rhand_port_num
                thera_lhand_port = "COM" + thera_lhand_port_num

                ports = [sync_port, sub_port, thera_port, thera_rhand_port, thera_lhand_port]
                ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
                port_dict = dict(zip(ports, ports_name))

            else:
                print("yまたはnを入力してください")
                pass


        check_port = "a"
        while check_port != "y":
            check_port = input(f"""ポート番号の確認
    同期用IMU AP09181356 : {sync_port}
    患者腰用IMU AP09182459 : {sub_port}
    療法士腰用IMU AP09182460 : {thera_port}
    療法士右手用IMU AP09182461 : {thera_rhand_port}
    療法士左手用IMU AP09182462 : {thera_lhand_port}
上記のポート番号で正しいですか？(y:/n): """)

            # print(f"以下のポート番号が正しいか確認してください")
            # print(f"同期用IMU AP09181356 : {sync_port}")
            # print(f"患者腰用IMU AP09182459 : {sub_port}")
            # print(f"療法士腰用IMU AP09182460 : {thera_port}")
            # print(f"療法士右手用IMU AP09182461 : {thera_rhand_port}")
            # print(f"療法士左手用IMU AP09182462 : {thera_lhand_port}")
            # check_port = input(f"以上の接続で良いですか？(y:計測条件の入力/n:ポート番号の再設定): ")

            if check_port == "y":
                pass
            elif check_port == "n":
                # tkzkの場合
                sync_port_num = input("同期用IMU AP09181356 のポート番号を入力:COM")
                sub_port_num = input("患者腰用IMU AP09182459 のポート番号を入力:COM")
                thera_port_num = input("療法士腰用IMU AP09182460 のポート番号を入力:COM")
                thera_rhand_port_num = input("療法士右手用IMU AP09182461 のポート番号を入力:COM")
                thera_lhand_port_num = input("療法士左手用IMU AP09182462 のポート番号を入力:COM")

                # otの場合
                # sync_port_num = input("同期用IMU AP09181497 のポート番号を入力:COM")
                # sub_port_num = input("患者腰用IMU AP09181498 のポート番号を入力:COM")
                # thera_port_num = input("療法士腰用IMU AP09181354 のポート番号を入力:COM")
                # thera_rhand_port_num = input("療法士右手用IMU AP09181355 のポート番号を入力:COM")
                # thera_lhand_port_num = input("療法士左手用IMU AP09181357 のポート番号を入力:COM")

                sync_port = "COM" + sync_port_num
                sub_port = "COM" + sub_port_num
                thera_port = "COM" + thera_port_num
                thera_rhand_port = "COM" + thera_rhand_port_num
                thera_lhand_port = "COM" + thera_lhand_port_num

                ports = [sync_port, sub_port, thera_port, thera_rhand_port, thera_lhand_port]
                ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
                port_dict = dict(zip(ports, ports_name))
            else:
                print("yまたはnを入力してください")

        #ポート番号を再利用するためjsonファイルに保存
        port_dict_file = root_dir / "port_dict.json"
        with open(port_dict_file, "w") as file:
            json.dump(port_dict, file, indent=4, ensure_ascii=False)  # indentはインデントの際のスペースの数、ensure_ascii=Falseで日本語をそのまま保存

        # 計測条件の入力、保存先のディレクトリを作成
        current_date = datetime.now().strftime('%Y%m%d')
        condition = input("計測条件を入力してください: ")
        save_dir = root_dir / current_date / condition

        new_save_dir = save_dir
        i = 1
        while new_save_dir.exists():
            new_save_dir = save_dir.with_name(f"{condition}_{i}")
            i += 1
        new_save_dir.mkdir(parents=True, exist_ok=False)

    main(ports, port_dict, new_save_dir)