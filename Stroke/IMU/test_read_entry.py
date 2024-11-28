import time
import struct
import ctypes
import serial
import csv
from pathlib import Path
from datetime import datetime

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
    print(f"受信バッファの初期データ数: {port}, {ser.in_waiting}")
    # ser.in_waitingは受信データのバイト数を返す
    while ser.in_waiting > 0:
        strat_wait = ser.in_waiting
        start_responce = len(response)
        # print(f"受信バッファのデータ数: {port}, {ser.in_waiting}")
        read_data = ser.read(min(ser.in_waiting, 100))
        # print(f"受信データ数: {port}, {len(read_data), read_data}")
        response += read_data
        end_wait = ser.in_waiting
        end_responce = len(response)
        # print(f"{port} レスポンス数:{end_responce - start_responce} 取得データ数:{strat_wait - end_wait}")
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

    print(f"{port}の合計データ数（バイト）: {len(response)}")

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

def read_save_memory(port, port_dict, start_time_dict, save_dir):
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
        save_path = save_dir / f'sensor_data_{port_dict[port]}_{start_time_dict[port]}.csv'
        save_to_csv(accel_gyro_data, geomagnetic_data, save_path)
        print(f"計測データをCSVに保存しました。 ({port})")

    # シリアルポートを閉じる
    ser.close()


def main():
    root_dir = Path(r"C:\Users\zutom\OneDrive\デスクトップ\IMU\data\20241128\data")

    sub_port = "COM28"
    ports = [sub_port]
    ports_name = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
    port_dict = dict(zip(ports, ports_name))

    start_time_dict = {
        f"{sub_port}": "2024-11-28-10-19-16"
    }

    # 計測条件の入力、保存先のディレクトリを作成
    current_date = datetime.now().strftime('%Y%m%d')
    condition = "a"
    # condition = input("計測条件を入力してください: ")
    save_dir = root_dir / current_date / condition

    new_save_dir = save_dir
    i = 1
    while new_save_dir.exists():
        new_save_dir = save_dir.with_name(f"{condition}_{i}")
        i += 1
    new_save_dir.mkdir(parents=True, exist_ok=False)

    save_dir = Path(r"C:\Users\zutom\OneDrive\デスクトップ\IMU\data\20241128\a")

    read_save_memory(sub_port, port_dict, start_time_dict, save_dir)




if __name__ == "__main__":
    main()