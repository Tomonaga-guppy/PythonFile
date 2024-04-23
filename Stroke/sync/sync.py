import serial
import time

# Arduinoのシリアルポートを開く（適切なポート名に置き換えてください）
ser = serial.Serial('COM3', 9600)  # Windowsの場合

def send_data(data):
    """
    Arduinoにデータを送信する関数
    :param data: 送信するデータ（文字列）
    """
    ser.write(data.encode())  # データをエンコードして送信 string -> bytes

try:
    # '1'をArduinoに送信
    send_data('1')
    while True:
        # シリアル通信を停止したい場合、ループを終了させる
        #enterキーを押すと終了
        if input() == '':
            break


finally:
    # 終了時にシリアルポートを閉じる
    send_data('1')
    ser.close()
