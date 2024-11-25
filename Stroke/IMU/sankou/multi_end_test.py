import serial
import time
import struct
import ctypes
import csv
import threading

#各スレッドで終了をするためのイベント
stop_event = threading.Event()

def configure_accelgyro(ser):
    print(f"\n加速度レスポンス: b'fegbkjafnjnakjnr'")
    print("加速度の設定は正常に完了しました。")

def configure_magnetic(ser):
    print(f"地磁気レスポンス: b'472967842y98ty4'")
    print("地磁気の設定は正常に完了しました。")


def send_sync_signal(ser, level):
    pass

def start_measurement(ser):
    # 計測開始コマンド
    start_time = time.time()
    print(f"計測開始時刻 {start_time}")

def stop_measurement(ser):
    pass
    print(f"\n計測を停止しました。")

def get_entry_count(ser):
    print(f"\nエントリ件数レスポンス: b'93jfriwnscidh")


def read_entry(ser, entry_number):
    print(f"内部メモリから全てのデータを取得しました。")

    accel_gyro_data = []
    geomagnetic_data = []
    return accel_gyro_data, geomagnetic_data

def save_to_csv(accel_gyro_data, geomagnetic_data, filename):
    pass


def read_sensor_data(ser, port):
    try:
        i = 1
        while not stop_event.is_set():
            print(f"ポート{port} {i}回目のデータ取得")
            i += 1

    except KeyboardInterrupt:
        pass

def clear_measurement_data(ser):
    print("計測データの記録クリアが正常に完了しました。")

def send_sync_pulse(ser):
    send_sync_signal(ser, level=8)  # Low出力 (8)
    time.sleep(0.0001)  # 少しの間Lowに保持
    send_sync_signal(ser, level=9)  # High出力 (9)

def run_imu_on_port(port, barrier):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200
    # シリアルポートを開く

    print(f"{port} is open")
    configure_accelgyro(ser)
    configure_magnetic(ser)

    barrier.wait()  # 他のスレッドと同期

    # 計測を開始する
    start_measurement(ser)
    print(f"計測設定が完了し、計測を開始しました。 ({port})\n")

    # 加速度、角速度、地磁気のデータを読み取るループを開始
    try:
        read_sensor_data(ser, port)
    finally:
        stop_measurement(ser)

def read_memory(port):
    ser = serial.Serial()
    ser.port = port
    ser.timeout = 1.0
    ser.baudrate = 115200

    # 最新の計測記録を取得し、CSVに保存
    print(f"計測データをCSVに保存しました。 ({port})")

    # 計測データの記録をクリア
    clear_measurement_data(ser)

    send_sync_signal(ser, level=9)  # 外部端子1をHigh出力

    # シリアルポートを閉じる
    ser.close()
    print(f"計測を終了し、シリアルポートを閉じました。 ({port})")

def main():
    sync_port = "COM" + input("同期用IMUのポート番号を入力:COM")
    subject_port = "COM" + input("患者用IMUのポート番号を入力:COM")
    therapist_port = "COM" + input("療法士用IMUのポート番号を入力:COM")
    ports = [sync_port, subject_port, therapist_port]
    # ports = ["COM11", "COM6", "COM8"]
    threads = []
    threads_post = []
    barrier = threading.Barrier(len(ports))  # スレッド間の同期のためのバリアを作成

    # 各シリアルポートに対してスレッドを作成し、実行
    for port in ports:
        thread = threading.Thread(target=run_imu_on_port, args=(port,barrier))
        threads.append(thread)

    try:
        for thread in threads:
            thread.start()
       # スレッドが実行中はメインスレッドはそのまま継続する
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.00001)  # 少し待機しながらスレッドの終了を待つ
    except KeyboardInterrupt:
        stop_event.set()  # Signal threads to stop

        # Wait for all threads to finish safely
        for thread in threads:
            thread.join()

        print("計測を終了しました。IMUメモリの書き出しを行います")


    ## 内部メモリの書き出し
    for port in ports:
        thread_post = threading.Thread(target=read_memory, args=(port,))
        threads_post.append(thread_post)

    for thread_post in threads_post:
        thread_post.start()

    print("メモリの書き出しを行っています 少々お待ちください")

    for thread_post in threads_post:
        thread_post.join()

    print("\n全てのIMUで書き出しを終了しました\n ")

if __name__ == '__main__':
    main()
