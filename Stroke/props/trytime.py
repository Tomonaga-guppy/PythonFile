import datetime
import time

#日本時間で時刻をマイクロ秒単位で取得
now = datetime.datetime.now()
print(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')}")

time.sleep(10)
now2 = datetime.datetime.now()
print(f"Start time: {now2.strftime('%Y-%m-%d %H:%M:%S.%f')}")

defference = now2 - now
print(f"defference: {defference}")