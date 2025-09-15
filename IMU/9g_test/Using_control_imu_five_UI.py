import serial
import time
import struct
import ctypes
import csv
import multiprocessing
from datetime import datetime
from pathlib import Path
import json

#各プロセスで終了を共有するためのイベント
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
    check = header ^ cmd ^ data1 ^ data2 ^ data3
    command = bytearray([header, cmd, data1, data2, data3, check])

    ser.read(100)
    ser.write(command)
    response = ser.read(3)

    if len(response) == 3 and response[2] == 0:
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
    data1 = 0x0A  # 計測周期 10ms 100Hz
    data2 = 0x01  # 送信データの平均回数 1回
    data3 = 0x01  # 記録データの平均回数 1回
    check = header ^ cmd ^ data1 ^ data2 ^ data3
    command = bytearray([header, cmd, data1, data2, data3, check])

    ser.read(100)
    ser.write(command)
    response = ser.read(3)

    if len(response) == 3 and response[2] == 0:
        flag = True
    else:
        print(f"地磁気の設定に失敗しました {port}")
        flag = False
    return flag

def start_measurement(ser, port):
    flag = False
    header = 0x9A
    cmd = 0x13
    params = bytes([0x00] * 14) # 相対時間指定で即時開始
    check = header ^ cmd
    for p in params:
        check ^= p
    command = bytearray([header, cmd]) + params + bytearray([check])

    ser.read(100)
    ser.write(command)
    time.sleep(0.1)
    response = ser.read(15)
    start_time = ""

    if len(response) == 15:
        start_year = 2000 + response[3]
        start_month = response[4]
        start_day = response[5]
        start_hour = response[6]
        start_minute = response[7]
        start_second = response[8]
        start_time = f"{start_year}-{start_month:02d}-{start_day:02d}-{start_hour:02d}-{start_minute:02d}-{start_second:02d}"
        print(f"{port}の計測が開始されました : {start_time}")
        flag = True
    else:
        print(f"計測開始のレスポンスが正しくありません。 {(response)}")
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
    if response and response[1] == 0xB6:
        return response[2]
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
    time.sleep(1)

    response = b''
    while ser.in_waiting > 0:
        response += ser.read(ser.in_waiting)
        time.sleep(0.01)

    accel_gyro_data, geomagnetic_data, extension_data = [], [], []
    i = 0
    while i < len(response):
        if response[i] == 0x9a and i + 1 < len(response):
            cmd_code = response[i+1]
            if cmd_code == 0x80 and i + 24 <= len(response):
                data = struct.unpack('<I', response[i+2:i+6])[0]
                ag = [parse_3byte_signed(response[i+x:i+x+3]) for x in range(6, 24, 3)]
                accel_gyro_data.append([data] + ag)
                i += 24
            elif cmd_code == 0x81 and i + 15 <= len(response):
                data = struct.unpack('<I', response[i+2:i+6])[0]
                mag = [parse_3byte_signed(response[i+x:i+x+3]) for x in range(6, 15, 3)]
                geomagnetic_data.append([data] + mag)
                i += 15
            elif cmd_code == 0x84 and i + 11 <= len(response):
                timestamp, port_status = struct.unpack('<IB', response[i+2:i+7])
                ports = [(port_status >> j) & 1 for j in range(4)]
                ad_vals = struct.unpack('<HH', response[i+7:i+11])
                extension_data.append([timestamp] + ports + list(ad_vals))
                i += 11
            else:
                i += 1
        else:
            i += 1
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
            ag_data = ["ags"] + accel_gyro_data[i] if i < len(accel_gyro_data) else ["ags"] + [""] * 7
            geo_data = ["geo"] + geomagnetic_data[i] if i < len(geomagnetic_data) else ["geo"] + [""] * 4
            ext_data = ["ext data"] + extension_data[i] if i < len(extension_data) else ["ext data"] + [""] * 7
            writer.writerow(ag_data + geo_data + ext_data)

def parse_3byte_signed(data):
    if len(data) != 3:
        raise ValueError("データ長が3バイトでありません")
    val = int.from_bytes(data, byteorder='little', signed=False)
    if val & (1 << 23):
        val -= (1 << 24)
    return val

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
        if result != b'\x00':
            print("計測データの記録クリアに失敗しました。")
    else:
        print("クリアコマンドのレスポンスが正しくありません。")

# --- 変更点 ---
# run_imu_on_port関数は、ポートを開いてから閉じるまでの一連の処理を担当します。
# 連続計測のループの中で毎回呼び出されるため、ポートの開閉も毎回行われます。
def run_imu_process(port, barrier, result_queue):
    ser = None
    try:
        ser = serial.Serial(port, 115200, timeout=1.0)
        print(f"{port} のポートを開きました。")

        set_device_time(ser)
        if not configure_accelgyro(ser, port):
            raise Exception("加速度の設定に失敗")
        if not configure_magnetic(ser, port):
            raise Exception("地磁気の設定に失敗")
        
        set_expantion_terminal(ser)
        set_voltage(ser, params=[9, 1, 10, 10])
        print(f"{port} IMUの設定が完了しました。")

        barrier.wait(timeout=60)

        flag, start_time = start_measurement(ser, port)
        if not flag:
            raise Exception("計測の開始に失敗")
        
        result_queue.put({'port': port, 'start_time': start_time, 'status': 'started'})

        set_voltage(ser, params=[8, 1, 10, 10])
        time.sleep(1)
        set_voltage(ser, params=[9, 1, 10, 10])

        while not stop_event.is_set():
            time.sleep(0.01)

    except Exception as e:
        print(f"エラーが発生しました ({port}): {e}")
    finally:
        if ser and ser.is_open:
            set_voltage(ser, params=[9, 1, 10, 10])
            stop_measurement(ser, port)
            
            # --- 変更点 ---
            # メモリの読み出しと保存を各プロセス内で行うように変更
            entry_count = get_entry_count(ser, port)
            if entry_count > 0:
                accel_gyro, geo, ext = read_entry(ser, entry_count, port)
                result_queue.put({
                    'port': port,
                    'status': 'data_read',
                    'data': (accel_gyro, geo, ext)
                })
            
            # entry_countが40を超えたらクリア
            if entry_count > 40:
                clear_measurement_data(ser)

            ser.close()
            print(f"ポートを閉じました。 ({port})")

# --- 変更点 ---
# main関数を計測1回分の処理を担う`run_measurement_cycle`に名称変更・再構成しました。
def run_measurement_cycle(ports, port_dict, save_dir):
    stop_event.clear()  # 実行前にイベントをクリア
    processes = []
    result_queue = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(len(ports))

    for port in ports:
        process = multiprocessing.Process(target=run_imu_process, args=(port, barrier, result_queue))
        processes.append(process)
        process.start()
    
    start_time_dict = {}
    try:
        # すべてのプロセスが計測を開始するのを待つ
        for _ in ports:
            result = result_queue.get(timeout=60)
            if result['status'] == 'started':
                start_time_dict[result['port']] = result['start_time']

        print("\n計測中です... 停止するにはCTRL+Cを押してください。")
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n停止コマンドを送信します...")
        stop_event.set()
    
    # データの読み出しと保存
    read_data_dict = {}
    # キューが空になるまでデータを取得
    while not result_queue.empty():
         result = result_queue.get()
         if result['status'] == 'data_read':
             read_data_dict[result['port']] = result['data']

    for p in processes:
        p.join()

    print("\nメモリの書き出しと保存を行います。")
    for port in ports:
        if port not in start_time_dict:
            print(f'計測開始時刻が不明なため、データを保存できませんでした {port}')
            continue
        if port in read_data_dict:
            save_path = save_dir / f'{port_dict[port]}_{start_time_dict[port]}.csv'
            accel_gyro_data, geomagnetic_data, extension_data = read_data_dict[port]
            save_to_csv(accel_gyro_data, geomagnetic_data, extension_data, save_path)
            print(f'計測データを "{save_path}" に保存しました {port}')
        else:
            print(f'データの読み出しに失敗したため、保存できませんでした {port}')


if __name__ == "__main__":
    multiprocessing.freeze_support()

    root_dir = Path(r"C:\Users\zutom\Desktop\IMUtool\test_data")
    # root_dir = Path(r"C:\Users\BRLAB\Desktop\data")
    # root_dir = Path(r"C:\Users\tus\Desktop\data")

    # --- 変更点 ---
    # ポート番号の設定は最初の一度だけ行います。
    port_dict_file = root_dir / "port_dict.json"
    port_dict = {}
    
    reuse_port_flag = input("前回のポート番号を再利用しますか？(y/n): ").lower()
    if reuse_port_flag == 'y':
        try:
            with open(port_dict_file, "r") as f:
                port_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("ポート設定ファイルが見つからないか、不正です。手動で入力してください。")
            reuse_port_flag = 'n'
    
    if reuse_port_flag != 'y':
        ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
        imu_ids = ["AP0421533", "AP0421538", "AP0421539", "AP0421540", "AP0421541"]
        ports = []
        for name, imu_id in zip(ports_name, imu_ids):
            port_num = input(f"{name}用IMU {imu_id} のポート番号を入力: COM")
            ports.append("COM" + port_num)
        port_dict = dict(zip(ports, ports_name))

    while True:
        print("\nポート番号の確認:")
        for port, name in port_dict.items():
            print(f"    {name}: {port}")
        check_port = input("上記のポート番号で正しいですか？(y/n): ").lower()
        if check_port == 'y':
            break
        else:
            print("ポート番号を再入力してください。")
            ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
            imu_ids = ["AP0421533", "AP0421538", "AP0421539", "AP0421540", "AP0421541"]
            ports = []
            for name, imu_id in zip(ports_name, imu_ids):
                port_num = input(f"{name}用IMU {imu_id} のポート番号を入力: COM")
                ports.append("COM" + port_num)
            port_dict = dict(zip(ports, ports_name))

    ports = list(port_dict.keys())
    with open(port_dict_file, "w") as f:
        json.dump(port_dict, f, indent=4)

    # --- 変更点 ---
    # 計測全体をループで囲み、連続計測を可能にします。
    while True:
        # 計測条件の入力、保存先のディレクトリを作成
        current_date = datetime.now().strftime('%Y%m%d')
        sub_num = "sub" + input("被験者番号を入力してください: sub")
        thera_num = "thera" + input("介助者番号を入力してください（介助なしの場合は0を入力）: thera")
        record_num = "-" + input("この条件での撮影回数を入力してください :")
        save_dir_base = root_dir / current_date / sub_num / (thera_num + record_num)

        # 同名のディレクトリが存在する場合、連番を付与
        save_dir = save_dir_base
        i = 2
        while save_dir.exists():
            save_dir = save_dir_base.with_name(f"{save_dir_base.name}_{i}")
            i += 1
        save_dir.mkdir(parents=True)
        
        print(f"データは{save_dir}に保存されます")
        
        imu_save_folder = save_dir / "IMU"
        imu_save_folder.mkdir()

        port_dict_check_file = imu_save_folder / "port_dict_check.json"
        with open(port_dict_check_file, "w") as f:
            json.dump(port_dict, f, indent=4)
        
        # 計測サイクルの実行
        run_measurement_cycle(ports, port_dict, imu_save_folder)

        # 連続計測の確認
        continue_measurement = input("\n連続で計測を続けますか？(y/n): ").lower()
        if continue_measurement != 'y':
            break
            
    print("プログラムを終了します。")
