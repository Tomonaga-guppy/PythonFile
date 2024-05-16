import concurrent.futures
import time
import threading

def task(n, start_times):
    start_time = time.perf_counter()  #秒単位の時間を取得
    start_times[n] = start_time
    print(f'Task {n} started at {start_time}')
    time.sleep(1)
    print(f'Task {n} finished')

start_times = {}

# ロックオブジェクトを使用して start_times へのアクセスを保護します
lock = threading.Lock()

def synchronized_task(n):
    with lock:
        task(n, start_times)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(synchronized_task, i) for i in range(3)]

# Wait for all futures to complete
concurrent.futures.wait(futures)

# タスクの開始時間の差を計算して表示します
start_times_list = [start_times[i] for i in range(len(start_times))]
time_differences = [start_times_list[i] - start_times_list[0] for i in range(1, len(start_times_list))]

print(f"Start times: {start_times_list}")
print(f"Time differences: {time_differences}")
