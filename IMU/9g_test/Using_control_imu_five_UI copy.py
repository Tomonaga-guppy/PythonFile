"""
theading使ってやってみたやつだけど連続して記録すると時刻などからおかしくなりうまくいかない
"""

import serial
import time
import struct
import ctypes
import csv
import threading
import queue
from datetime import datetime
from pathlib import Path
import json

# 各スレッドで共有する停止イベント
stop_event = threading.Event()

# -----------------------------------------------------------------------------
# IMUへのコマンド送信やデータ解析を行うヘルパー関数 (一部修正あり)
# -----------------------------------------------------------------------------
def set_device_time(ser):
    header = 0x9A
    cmd = 0x11
    current_time = datetime.now()
    year = current_time.year - 2000
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    millisecond = int(current_time.microsecond / 1000)
    check = header ^ cmd ^ year ^ month ^ day ^ hour ^ minute ^ second
    check ^= (millisecond & 0xFF) ^ (millisecond >> 8)
    command = bytearray([header, cmd, year, month, day, hour, minute, second])
    command += struct.pack('<H', millisecond)
    command.append(check)
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
    if len(response) == 3 and response[2] == 0:
        flag = True
    else:
        print(f"加速度の設定に失敗しました {port}")
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
    if len(response) == 3 and response[2] == 0:
        flag = True
    else:
        print(f"地磁気の設定に失敗しました {port}")
    return flag

def start_measurement(ser, port):
    flag = False
    header = 0x9A
    cmd = 0x13
    smode, syear, smonth, sday, shour, smin, ssec = 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00
    emode, eyear, emonth, eday, ehour, emin, esec = 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00
    check = header ^ cmd ^ smode ^ syear ^ smonth ^ sday ^ shour ^ smin ^ ssec
    check ^= emode ^ eyear ^ emonth ^ eday ^ ehour ^ emin ^ esec
    command = bytearray([header, cmd, smode, syear, smonth, sday, shour, smin, ssec,
                         emode, eyear, emonth, eday, ehour, emin, esec, check])
    
    ser.reset_input_buffer()
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(15)
    start_time = ""

    if len(response) == 15 and response[0] == 0x9A and response[1] == 0x93:
        start_year = 2000 + response[3]
        start_month, start_day, start_hour, start_minute, start_second = response[4], response[5], response[6], response[7], response[8]
        start_time = f"{start_year}-{start_month:02d}-{start_day:02d}-{start_hour:02d}-{start_minute:02d}-{start_second:02d}"
        print(f"{port}の計測が開始されました : {start_time}")
        flag = True
    else:
        print(f"[{port}] 計測開始時の応答が正しくありません。 (response: {response.hex()})")

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
    # 【修正点 1】バッファクリアのタイムアウトを3秒に延長
    header = 0x9A
    cmd = 0x15
    option = 0x00
    check = header ^ cmd ^ option
    command = bytearray([header, cmd, option, check])

    t_start = time.time()
    while ser.in_waiting > 0:
        ser.read(ser.in_waiting)
        time.sleep(0.05)
        if time.time() - t_start > 3: # タイムアウトを3秒に延長
            print(f"[{port}] バッファのクリアにタイムアウトしました。")
            break
            
    ser.write(command)
    time.sleep(0.2)
    response = ser.read(3)
    if not (len(response) == 3 and response[0] == 0x9A and response[1] == 0x8F):
        print(f"[{port}] 計測停止への応答が予期せぬものでした: {response.hex()}")
    print(f"計測を停止しました {port}")

def get_entry_count(ser, port):
    header = 0x9A
    cmd = 0x36
    option = 0x00
    check = header ^ cmd ^ option
    command = bytearray([header, cmd, option, check])
    ser.reset_input_buffer()
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(4)
    if len(response) == 4 and response[0] == 0x9A and response[1] == 0xB6:
        bcc = response[0] ^ response[1] ^ response[2]
        if bcc != response[3]:
            print(f"[{port}] エントリ件数応答のチェックサムが一致しません")
            return 0
        return response[2]
    else:
        print(f"エントリ件数の取得に失敗しました {port} (response: {response.hex()})")
        return 0

def read_entry(ser, entry_number, port):
    header = 0x9A
    cmd = 0x39
    check = header ^ cmd ^ entry_number
    command = bytearray([header, cmd, entry_number, check])
    ser.reset_input_buffer()
    ser.write(command)
    response = b''
    time.sleep(2.0)
    while ser.in_waiting > 0:
        response += ser.read(ser.in_waiting)
        time.sleep(0.05)

    accel_gyro_data, geomagnetic_data, extension_data = [], [], []
    i = 0
    while i < len(response):
        if response[i] == 0x9a and i + 1 < len(response):
            cmd_code = response[i + 1]
            if cmd_code == 0x80 and i + 24 <= len(response):
                timestamp, ax, ay, az, gx, gy, gz = struct.unpack('<I', response[i+2:i+6])[0], parse_3byte_signed(response[i+6:i+9]), parse_3byte_signed(response[i+9:i+12]), parse_3byte_signed(response[i+12:i+15]), parse_3byte_signed(response[i+15:i+18]), parse_3byte_signed(response[i+18:i+21]), parse_3byte_signed(response[i+21:i+24])
                accel_gyro_data.append([timestamp, ax, ay, az, gx, gy, gz])
                i += 24
            elif cmd_code == 0x81 and i + 15 <= len(response):
                timestamp, mx, my, mz = struct.unpack('<I', response[i+2:i+6])[0], parse_3byte_signed(response[i+6:i+9]), parse_3byte_signed(response[i+9:i+12]), parse_3byte_signed(response[i+12:i+15])
                geomagnetic_data.append([timestamp, mx, my, mz])
                i += 15
            elif cmd_code == 0x84 and i + 11 <= len(response):
                timestamp, port_status = struct.unpack('<I', response[i+2:i+6])[0], response[i+6]
                p0, p1, p2, p3 = (port_status >> 0) & 1, (port_status >> 1) & 1, (port_status >> 2) & 1, (port_status >> 3) & 1
                ad0, ad1 = struct.unpack('<H', response[i+7:i+9])[0], struct.unpack('<H', response[i+9:i+11])[0]
                extension_data.append([timestamp, p0, p1, p2, p3, ad0, ad1])
                i += 11
            else: i += 1
        else: i += 1
    return accel_gyro_data, geomagnetic_data, extension_data

def save_to_csv(accel_gyro_data, geomagnetic_data, extension_data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Timestamp_Acc", "Acc_X 0.1[mG]", "Acc_Y 0.1[mG]", "Acc_Z 0.1[mG]", "Gyro_X 0.01[dps]", "Gyro_Y 0.01[dps]", "Gyro_Z 0.01[dps]", "Type", "Timestamp_Mag", "Mag_X 0.1[μT]", "Mag_Y 0.1[μT]", "Mag_Z 0.1[μT]", "Type", "Timestamp_Ext", "Port 0", "Port1", "Port2", "Port3", "AD0", "AD1"])
        max_len = max(len(accel_gyro_data), len(geomagnetic_data), len(extension_data))
        for i in range(max_len):
            ag_data = ["ags"] + accel_gyro_data[i] if i < len(accel_gyro_data) else ["ags", "", "", "", "", "", "", ""]
            geo_data = ["geo"] + geomagnetic_data[i] if i < len(geomagnetic_data) else ["geo", "", "", "", ""]
            ext_data = ["ext data"] + extension_data[i] if i < len(extension_data) else ["ext data", "", "", "", "", "", "", ""]
            writer.writerow(ag_data + geo_data + ext_data)

def parse_3byte_signed(data):
    if len(data) != 3: raise ValueError("データ長が3バイトでありません")
    data4 = 0xFF if data[2] & 0x80 else 0x00
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
        if ser.read(1) == b'\x00': pass
        else: print("計測データの記録クリアに失敗しました。")
    else: print("レスポンスが正しくありません。")

def run_measurement_thread(port, ser, barrier, start_queue):
    try:
        barrier.wait(timeout=60)
        flag, start_time = start_measurement(ser, port)
        if not flag:
            print(f"計測の開始に失敗しました。計測を終了します({port})")
            start_queue.put((port, "ERROR"))
            return

        start_queue.put((port, start_time))

        set_voltage(ser, params=[8, 1, 10, 10])
        time.sleep(1)
        set_voltage(ser, params=[9, 1, 10, 10])

        stop_event.wait()

    except Exception as e:
        print(f"計測スレッドでエラーが発生しました ({port}): {e}")
    finally:
        set_voltage(ser, params=[9, 1, 10, 10])
        stop_measurement(ser, port)

def read_and_save_data(port, ser, port_dict, start_time_dict, save_dir):
    if port not in start_time_dict or start_time_dict[port] == "ERROR":
        print(f'[{port}] 計測開始に失敗したため、データを保存できませんでした')
        return

    save_path = save_dir / f'{port_dict[port]}_{start_time_dict[port]}.csv'
    entry_count = get_entry_count(ser, port)
    if entry_count > 0:
        print(f"[{port}] {entry_count}件の記録を読み出します...")
        accel_gyro_data, geomagnetic_data, extension_data = read_entry(ser, entry_count, port)
        save_to_csv(accel_gyro_data, geomagnetic_data, extension_data, save_path)
        print(f'計測データを "{save_path}" に保存しました')
    else:
        print(f"保存するデータがありませんでした {port}")


def main():
    root_dir = Path(r"C:\Users\zutom\Desktop\IMUtool\test_data")
    port_dict = {}
    ports = []
    
    reuse_port_flag = input("前回のポート番号を再利用しますか？(y/n): ").lower()
    if reuse_port_flag == 'y':
        try:
            with open(root_dir / "port_dict.json", "r") as f:
                port_dict = json.load(f)
            ports = list(port_dict.keys())
        except (FileNotFoundError, json.JSONDecodeError):
            print("前回のポート番号が読み込めませんでした。ポート番号を再入力してください。")
            reuse_port_flag = 'n'

    if reuse_port_flag == 'n':
        sync_port = "COM" + input("同期用IMU AP0421533 のポート番号を入力:COM")
        sub_port = "COM" + input("患者腰用IMU AP0421538 のポート番号を入力:COM")
        thera_port = "COM" + input("療法士腰用IMU AP04215339 のポート番号を入力:COM")
        thera_rhand_port = "COM" + input("療法士右手用IMU AP0421540 のポート番号を入力:COM")
        thera_lhand_port = "COM" + input("療法士左手用IMU AP0421541 のポート番号を入力:COM")
        ports = [sync_port, sub_port, thera_port, thera_rhand_port, thera_lhand_port]
        ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
        port_dict = dict(zip(ports, ports_name))

    print("\n--- ポート番号の確認 ---")
    for port, name in port_dict.items():
        print(f"    {name.ljust(15)} : {port}")
    
    with open(root_dir / "port_dict.json", "w") as f:
        json.dump(port_dict, f, indent=4)

    sers = {}
    try:
        print("\nシリアルポートを開き、IMUの初期設定を行います...")
        for port in ports:
            ser = serial.Serial(port, 115200, timeout=1.0)
            sers[port] = ser
            set_device_time(ser)
            if not configure_accelgyro(ser, port) or not configure_magnetic(ser, port):
                raise IOError(f"{port}のIMU設定に失敗しました。")
            set_expantion_terminal(ser)
            set_voltage(ser, params=[9, 1, 10, 10])
        print("全IMUの初期設定が完了しました。")

        while True:
            print("\n--- 計測条件の入力 ---")
            current_date = datetime.now().strftime('%Y%m%d')
            sub_num = "sub" + input("被験者番号を入力してください: sub")
            thera_num = "thera" + input("介助者番号を入力してください（介助なしの場合は0を入力）: thera")
            record_num = "-" + input("この条件での撮影回数を入力してください :")
            base_dir_name = f"{thera_num}{record_num}"
            save_dir = root_dir / current_date / sub_num

            i = 1
            imu_save_dir = save_dir / base_dir_name
            while imu_save_dir.exists():
                i += 1
                imu_save_dir = save_dir / f"{base_dir_name}_{i}"
            imu_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"データは {imu_save_dir} に保存されます")

            threads = []
            barrier = threading.Barrier(len(ports))
            start_queue = queue.Queue()
            start_time_dict = {}
            stop_event.clear()

            for port in ports:
                thread = threading.Thread(target=run_measurement_thread, args=(port, sers[port], barrier, start_queue))
                threads.append(thread)
            
            print("\n計測準備完了。計測を停止するには、このウィンドウでCTRL+Cを押してください。")
            try:
                for thread in threads:
                    thread.start()
                
                while any(t.is_alive() for t in threads):
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n停止信号を受信しました。計測を停止します...")
                stop_event.set()
            
            for thread in threads:
                thread.join()

            # 【修正点 2】スレッド終了後の待機時間を1秒に延長
            time.sleep(1.0)

            while not start_queue.empty():
                port, start_time = start_queue.get()
                start_time_dict[port] = start_time

            print("\nメモリの書き出しを行います...")
            for port in ports:
                read_and_save_data(port, sers[port], port_dict, start_time_dict, imu_save_dir)

            choice = input("\n連続で計測を続けますか？ (y/n): ").lower()
            if choice != 'y':
                break
            
            # 【修正点 3】次のループに移る前に、全ポートのバッファを強制的にクリアする
            print("\n次の計測のためにバッファをクリアしています...")
            for port, ser in sers.items():
                if ser.in_waiting > 0:
                    ser.read(ser.in_waiting)
                    print(f"[{port}] のバッファをクリアしました。")


    except (IOError, serial.SerialException) as e:
        print(f"\nエラーが発生しました: {e}")
    finally:
        print("\nスクリプトを終了します。ポートを閉じています...")
        for port, ser in sers.items():
            if ser and ser.is_open:
                ser.close()
                print(f"{port} のポートを閉じました。")

if __name__ == "__main__":
    main()

