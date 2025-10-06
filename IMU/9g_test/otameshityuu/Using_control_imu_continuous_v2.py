"""
連続計測対応版 v2
シンプルな設計で確実に動作する実装
"""

import serial
import time
import struct
import ctypes
import csv
import multiprocessing
from datetime import datetime
from pathlib import Path
import json

def set_device_time(ser):
    # ヘッダとコマンドコードを定義
    header = 0x9A
    cmd = 0x11  # 時刻設定コマンド

    # 現在の時刻を取得
    current_time = datetime.now()
    # 各パラメータの設定
    year = current_time.year - 2000
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    millisecond = int(current_time.microsecond / 1000)

    # チェックサムの計算
    check = header ^ cmd ^ year ^ month ^ day ^ hour ^ minute ^ second
    check ^= (millisecond & 0xFF) ^ (millisecond >> 8)

    # コマンドリストを作成
    command = bytearray([header, cmd, year, month, day, hour, minute, second])
    command += struct.pack('<H', millisecond)
    command.append(check)

    # コマンド送信
    ser.read(100)
    ser.write(command)

def configure_accelgyro(ser, port):
    flag = False
    header = 0x9A
    cmd = 0x16
    data1 = 0x0A
    data2 = 0x01
    data3 = 0x01
    check = header ^ cmd ^ data1 ^ data2 ^ data3
    command = bytearray([header, cmd, data1, data2, data3, check])

    ser.read(100)
    ser.write(command)
    response = ser.read(3)

    if len(response) == 3:
        result = response[2]
        if result == 0:
            pass
        flag = True
    else:
        flag = False

    return flag

def configure_magnetic(ser, port):
    flag = False
    header = 0x9A
    cmd = 0x18
    data1 = 0x0A
    data2 = 0x01
    data3 = 0x01
    check = header ^ cmd ^ data1 ^ data2 ^ data3
    command = bytearray([header, cmd, data1, data2, data3, check])

    ser.read(100)
    ser.write(command)
    response = ser.read(3)

    if len(response) == 3:
        result = response[2]
        if result == 0:
            pass
            flag = True
    else:
        flag = False

    return flag

def start_measurement(ser, port):
    flag = False
    header = 0x9A
    cmd = 0x13
    smode = 0x00
    syear = 0x00
    smonth = 0x01
    sday = 0x01
    shour = 0x00
    smin = 0x00
    ssec = 0x00
    emode = 0x00
    eyear = 0x00
    emonth = 0x01
    eday = 0x01
    ehour = 0x00
    emin = 0x00
    esec = 0x00

    check = header ^ cmd ^ smode ^ syear ^ smonth ^ sday ^ shour ^ smin ^ ssec
    check ^= emode ^ eyear ^ emonth ^ eday ^ ehour ^ emin ^ esec

    command = bytearray([header, cmd, smode, syear, smonth, sday, shour, smin, ssec,
                         emode, eyear, emonth, eday, ehour, emin, esec, check])

    ser.read(100)
    ser.write(command)
    time.sleep(0.1)
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

def build_command(command_code, params=b''):
    header = 0x9A
    command_data = bytes([header, command_code]) + params
    bcc = 0
    for byte in command_data:
        bcc ^= byte
    return command_data + bytes([bcc])

def send_fire_and_forget(ser, command_code, params=b''):
    ser.read(100)
    command_to_send = build_command(command_code, params)
    ser.write(command_to_send)

def set_expantion_terminal(ser):
    params_1E = bytes([10, 1, 1, 1, 1])
    send_fire_and_forget(ser, 0x1E, params_1E)

def set_voltage(ser, params):
    params_30 = bytes(params)
    send_fire_and_forget(ser, 0x30, params_30)

def stop_measurement(ser, port):
    header = 0x9A
    cmd = 0x15
    option = 0x00
    check = header ^ cmd ^ option
    command = bytearray([header, cmd, option, check])

    ser.read(100)
    ser.write(command)
    print(f"計測を停止しました {port}")

def get_entry_count(ser, port):
    header = 0x9A
    cmd = 0x36
    option = 0x00
    check = header ^ cmd ^ option
    command = bytearray([header, cmd, option, check])

    ser.read(100)
    ser.write(command)
    response = ser.readline()

    time.sleep(0.1)
    if len(response) > 1 and response[1] == 0xB6:
        entry_count = response[2]
        return entry_count
    else:
        print(f"エントリ件数の取得に失敗しました {port}")
        return 0

def read_entry(ser, entry_number, port):
    header = 0x9A
    cmd = 0x39
    entry = entry_number
    check = header ^ cmd ^ entry
    command = bytearray([header, cmd, entry, check])

    while ser.in_waiting > 0:
        ser.read(100)
        time.sleep(0.01)

    ser.write(command)
    response = b''
    time.sleep(1)

    while ser.in_waiting > 0:
        read_data = ser.read(min(ser.in_waiting, 128))
        response += read_data
        time.sleep(0.01)

    accel_gyro_data = []
    geomagnetic_data = []
    extension_data = []

    i = 0
    while i < len(response):
        if response[i] == 0x9a:
            if i + 1 < len(response):
                if response[i + 1] == 0x80 and i + 24 <= len(response):
                    timestamp = struct.unpack('<I', response[i + 2:i + 6])[0]
                    accel_x = parse_3byte_signed(response[i + 6:i + 9])
                    accel_y = parse_3byte_signed(response[i + 9:i + 12])
                    accel_z = parse_3byte_signed(response[i + 12:i + 15])
                    gyro_x = parse_3byte_signed(response[i + 15:i + 18])
                    gyro_y = parse_3byte_signed(response[i + 18:i + 21])
                    gyro_z = parse_3byte_signed(response[i + 21:i + 24])
                    accel_gyro_data.append([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                    i += 24
                elif response[i + 1] == 0x81 and i + 15 <= len(response):
                    timestamp = struct.unpack('<I', response[i + 2:i + 6])[0]
                    geo_x = parse_3byte_signed(response[i + 6:i + 9])
                    geo_y = parse_3byte_signed(response[i + 9:i + 12])
                    geo_z = parse_3byte_signed(response[i + 12:i + 15])
                    geomagnetic_data.append([timestamp, geo_x, geo_y, geo_z])
                    i += 15
                elif response[i + 1] == 0x84 and i + 11 <= len(response):
                    timestamp = struct.unpack('<I', response[i + 2:i + 6])[0]
                    port_status = response[i + 6]
                    port0 = (port_status >> 0) & 1
                    port1 = (port_status >> 1) & 1
                    port2 = (port_status >> 2) & 1
                    port3 = (port_status >> 3) & 1
                    ad0 = struct.unpack('<H', response[i + 7:i + 9])[0]
                    ad1 = struct.unpack('<H', response[i + 9:i + 11])[0]
                    extension_data.append([timestamp, port0, port1, port2, port3, ad0, ad1])
                    i += 11
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    del response
    return accel_gyro_data, geomagnetic_data, extension_data

def save_to_csv(accel_gyro_data, geomagnetic_data, extension_data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Type", "Timestamp_Acc", "Acc_X 0.1[mG]", "Acc_Y 0.1[mG]", "Acc_Z 0.1[mG]", "Gyro_X 0.01[dps]", "Gyro_Y 0.01[dps]", "Gyro_Z 0.01[dps]",
            "Type", "Timestamp_Mag", "Mag_X 0.1[μT]", "Mag_Y 0.1[μT]", "Mag_Z 0.1[μT]",
            "Type", "Timestamp_Ext", "Port0", "Port1", "Port2", "Port3", "AD0", "AD1"
        ])

        max_len = max(len(accel_gyro_data), len(geomagnetic_data), len(extension_data))

        for i in range(max_len):
            if i < len(accel_gyro_data):
                ag_data = ["ags"] + accel_gyro_data[i]
            else:
                ag_data = ["ags", "", "", "", "", "", "", ""]

            if i < len(geomagnetic_data):
                geo_data = ["geo"] + geomagnetic_data[i]
            else:
                geo_data = ["geo", "", "", "", ""]

            if i < len(extension_data):
                ext_data = ["ext data"] + extension_data[i]
            else:
                ext_data = ["ext data", "", "", "", "", "", "", ""]

            writer.writerow(ag_data + geo_data + ext_data)

def parse_3byte_signed(data):
    if len(data) != 3:
        print(f"データ長: {data}")
        raise ValueError("データ長が3バイトでありません")

    if data[2] & 0x80:
        data4 = 0xFF
    else:
        data4 = 0x00

    value = data[0] + (data[1] << 8) + (data[2] << 16) + (data4 << 24)
    return ctypes.c_int(value).value

def clear_measurement_data(ser):
    header = 0x9A
    cmd = 0x35
    option = 0x00
    check = header ^ cmd ^ option
    command = bytearray([header, cmd, option, check])

    ser.reset_input_buffer()
    ser.write(command)
    response = ser.read(2)

    if len(response) == 2 and response[1] == 0x8F:
        result = ser.read(1)
        if result == b'\x00':
            pass
        else:
            pass
    else:
        print("レスポンスが正しくありません。")


def run_imu_on_port(port, barrier, start_queue, stop_event):
    """
    各IMUデバイスの計測を制御する関数
    """
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200

    # シリアルポート接続
    i = 1
    while not ser.is_open:
        if i == 10:
            print(f"シリアルポートを開くことができませんでした。 ({port})")
            return
        try:
            ser.open()
        except:
            time.sleep(0.1)
        i += 1

    print(f"{port} のポートを開きました。")

    try:
        # 初期設定(加速度、地磁気、外部拡張端子)
        flag = configure_accelgyro(ser, port)
        if not flag:
            raise Exception(f"加速度の設定に失敗しました。 ({port})")
        
        flag = configure_magnetic(ser, port)
        if not flag:
            raise Exception(f"地磁気の設定に失敗しました。 ({port})")

        set_expantion_terminal(ser)
        set_voltage(ser, params=[9, 1, 10, 10])
        print(f"{port} IMUの初期設定が完了しました。")

        # 全デバイスの初期設定完了を待つ
        barrier.wait(timeout=60)

        # 時刻の設定
        set_device_time(ser)

        # 全デバイスの時刻設定完了を待つ
        barrier.wait(timeout=60)

        # 計測を開始する
        flag, start_time = start_measurement(ser, port)
        if not flag:
            raise Exception(f"計測の開始に失敗しました。 ({port})")

        # 全デバイスの計測開始完了を待つ
        barrier.wait(timeout=60)

        # 計測開始の合図として端子1をLow(8)に1秒間設定
        set_voltage(ser, params=[8, 1, 0, 0])
        time.sleep(1)
        # 端子1をHigh(9)に戻す
        set_voltage(ser, params=[9, 1, 0, 0])

        # 計測中(stop_eventがセットされるまで待機)
        stop_event.wait()

        # 計測停止
        stop_measurement(ser, port)
        start_queue.put((port, start_time))

    except Exception as e:
        print(f"エラーが発生しました({port}): {e}")
        set_voltage(ser, params=[9, 1, 0, 0])
    finally:
        ser.close()
        print(f"{port} のシリアルポートを閉じました。")


def read_save_memory(port, port_dict, start_time_dict, save_path):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200

    i = 1
    while not ser.is_open:
        try:
            ser.open()
        except:
            time.sleep(1)
        i += 1

    while ser.in_waiting > 0:
        ser.readline()
        time.sleep(0.01)

    entry_count = get_entry_count(ser, port)
    if entry_count > 0:
        accel_gyro_data, geomagnetic_data, extension_data = read_entry(ser, entry_count, port)
        save_to_csv(accel_gyro_data, geomagnetic_data, extension_data, save_path)
        del accel_gyro_data, geomagnetic_data, extension_data

    # # 計測データの記録をクリア
    # if entry_count > 40:
    #     clear_measurement_data(ser)

    ser.close()
    return True


def main(ports, port_dict, root_dir):
    """
    メイン処理関数
    """
    measurement_count = 0
    continue_measurement = True
    sub_num = ""
    thera_num = ""

    while continue_measurement:
        measurement_count += 1

        # 計測条件の入力と保存先作成
        current_date = datetime.now().strftime('%Y%m%d')

        if measurement_count == 1:
            sub_num = "sub" + input("被験者番号を入力してください: sub")
            thera_num = "thera" + input("介助者番号を入力してください(介助なしの場合は0を入力): thera")

        record_num = "-" + input(f"この条件での撮影回数を入力してください({measurement_count}回目の計測) :")
        save_dir = root_dir / current_date / sub_num / (thera_num + record_num)

        new_save_dir = save_dir
        i = 2
        while new_save_dir.exists():
            new_save_dir = save_dir.with_name(f"{thera_num+record_num}_{i}")
            i += 1
        new_save_dir.mkdir(parents=True, exist_ok=False)

        print(f"データは{new_save_dir}に保存されます")

        imu_save_folder = new_save_dir / "IMU"
        imu_save_folder.mkdir(parents=True, exist_ok=True)

        # gopro_fl_path = new_save_dir / "fl"
        # gopro_fr_path = new_save_dir / "fr"
        # gopro_front_path = new_save_dir / "front"
        # gopro_sagi_path = new_save_dir / "sagi"
        # gopro_fl_path.mkdir(parents=True, exist_ok=True)
        # gopro_fr_path.mkdir(parents=True, exist_ok=True)
        # gopro_front_path.mkdir(parents=True, exist_ok=True)
        # gopro_sagi_path.mkdir(parents=True, exist_ok=True)

        port_dict_file2 = imu_save_folder / f"port_dict_check.json"
        with open(port_dict_file2, "w") as file:
            json.dump(port_dict, file, indent=4, ensure_ascii=False)

        # 計測用の変数を初期化
        threads = []
        barrier = multiprocessing.Barrier(len(ports))
        start_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        start_time_dict = {}

        ### 計測処理 ########################################################################################
        print(f"\n計測準備中...")
        
        # 各シリアルポートに対してプロセスを作成し、実行
        for port in ports:
            thread = multiprocessing.Process(
                target=run_imu_on_port,
                args=(port, barrier, start_queue, stop_event)
            )
            threads.append(thread)
            thread.start()

        # プロセスが計測開始するまで少し待つ
        time.sleep(7)  # 初期設定 + 時刻設定 + 計測開始に十分な時間
        
        print(f"\n{measurement_count}回目の計測を開始しました。")
        print("計測を停止するにはCtrl+Cを押してください。\n")

        try:
            # Ctrl+Cを待つ
            while any(thread.is_alive() for thread in threads):
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n計測を停止します...")
            stop_event.set()  # 計測停止のイベントをセット

            # 全てのスレッドが終了するまで待機
            for thread in threads:
                thread.join(timeout=5)

            # start_queueからstart_timeを取得
            while not start_queue.empty():
                port, start_time = start_queue.get()
                start_time_dict[port] = start_time

        ### 内部メモリの書き出し##############################################################################
        print("\nメモリの書き出しを5台分行います 少々お待ちください")
        start_time_save = time.time()
        
        for i, port in enumerate(ports):
            if port not in start_time_dict:
                print(f'{i+1}/{len(ports)} 計測開始時刻が不明なため、データを保存できませんでした {port}')
                continue
            save_path = imu_save_folder / f'{port_dict[port]}_{start_time_dict[port]}.csv'
            flag = read_save_memory(port, port_dict, start_time_dict, save_path)
            if flag:
                print(f'{i+1}/{len(ports)} 計測データを "{save_path}" に保存しました')
            else:
                print(f'{i+1}/{len(ports)} 計測データは保存できませんでした {port}')

        end_time_save = time.time()
        print(f"\n{measurement_count}回目の計測データの書き出しを完了しました")

        # 連続計測の確認
        continue_input = ""
        while continue_input not in ["y", "n"]:
            continue_input = input("\n連続で計測を続けますか？(y/n): ")
            if continue_input == "y":
                print("\n次の計測を開始します")
            elif continue_input == "n":
                continue_measurement = False
                print("\n全ての計測を終了します ウィンドウを閉じて終了してください")
            else:
                print("yまたはnを入力してください")

    # 最終的な終了処理
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if multiprocessing.current_process().name == "MainProcess":
        root_dir = Path(r"C:\Users\zutom\Desktop\IMUtool\test_data")
        # root_dir = Path(r"C:\Users\BRLAB\Desktop\data")  #tkrzk
        # root_dir = Path(r"C:\Users\tus\Desktop\data")  #ota
        
        reuse_port_flag = "a"
        while reuse_port_flag != "y" and reuse_port_flag != "n":
            reuse_port_flag = input("前回のポート番号を再利用しますか？(y/n): ")
            if reuse_port_flag == "y":
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
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    print("前回のポート番号が正常に読み込めませんでした。nを選択してポート番号を入力してください")
                    reuse_port_flag = "n"
                    pass

            if reuse_port_flag == "n":
                sync_port_num = input("同期用IMU AP0421533 のポート番号を入力:COM")
                sub_port_num = input("患者腰用IMU AP0421538 のポート番号を入力:COM")
                thera_port_num = input("療法士腰用IMU AP04215339 のポート番号を入力:COM")
                thera_rhand_port_num = input("療法士右手用IMU AP0421540 のポート番号を入力:COM")
                thera_lhand_port_num = input("療法士左手用IMU AP0421541 のポート番号を入力:COM")

                sync_port = "COM" + sync_port_num
                sub_port = "COM" + sub_port_num
                thera_port = "COM" + thera_port_num
                thera_rhand_port = "COM" + thera_rhand_port_num
                thera_lhand_port = "COM" + thera_lhand_port_num

                # ports = [sync_port]
                # ports_name = ["sync"]
                ports = [sync_port, sub_port, thera_port, thera_rhand_port, thera_lhand_port]
                ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
                port_dict = dict(zip(ports, ports_name))

        check_port = "a"
        while check_port != "y":
            print("\nポート番号の確認")
            print(f"    同期用IMU AP0421533 : {sync_port}")
            print(f"    患者腰用IMU AP0421538 : {sub_port}")
            print(f"    療法士腰用IMU AP0421539 : {thera_port}")
            print(f"    療法士右手用IMU AP0421540 : {thera_rhand_port}")
            print(f"    療法士左手用IMU AP0421541 : {thera_lhand_port}")
            check_port = input("上記のポート番号で正しいですか？(y/n): ")

            if check_port == "y":
                pass
            elif check_port == "n":
                sync_port_num = input("同期用IMU AP04212533 のポート番号を入力:COM")
                sub_port_num = input("患者腰用IMU AP0421538 のポート番号を入力:COM")
                thera_port_num = input("療法士腰用IMU AP0421539 のポート番号を入力:COM")
                thera_rhand_port_num = input("療法士右手用IMU AP0421540 のポート番号を入力:COM")
                thera_lhand_port_num = input("療法士左手用IMU AP0421541 のポート番号を入力:COM")

                sync_port = "COM" + sync_port_num
                sub_port = "COM" + sub_port_num
                thera_port = "COM" + thera_port_num
                thera_rhand_port = "COM" + thera_rhand_port_num
                thera_lhand_port = "COM" + thera_lhand_port_num

                # ports = [sync_port]
                # ports_name = ["sync"]
                ports = [sync_port, sub_port, thera_port, thera_rhand_port, thera_lhand_port]
                ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
                port_dict = dict(zip(ports, ports_name))
            else:
                print("yまたはnを入力してください")

        port_dict_file = root_dir / "port_dict.json"
        with open(port_dict_file, "w") as file:
            json.dump(port_dict, file, indent=4, ensure_ascii=False)

        main(ports, port_dict, root_dir)
