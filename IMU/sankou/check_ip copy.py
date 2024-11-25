import bluetooth

# Bluetoothデバイスをスキャン
print("Bluetoothデバイスをスキャン中...")
devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=False)

# 見つかったデバイスを表示
for addr, name in devices:
    print(f"見つかったデバイス: {name} - MACアドレス: {addr}")
