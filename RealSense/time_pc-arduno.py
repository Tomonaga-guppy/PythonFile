import serial
import time

# シリアルポートとボーレートを設定
ser = serial.Serial('COM3', 115200)  
time.sleep(2)  # Arduinoのリセット後の待機時間

# Arduinoにコマンドを送信してからのラグを計測
ser.write(b'\x01')
start_time = time.perf_counter_ns()

# Arduinoからの返信を待機
response = ser.readline().decode().strip()
end_time = time.perf_counter_ns()

# ラグの計算
lag = (end_time - start_time) * 1e-6  # ns -> ms
print(f"ラグ: {lag:.2f} ms")

ser.close()
