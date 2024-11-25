import asyncio
from bleak import BleakScanner

async def scan_ble_devices():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"デバイス名: {device.name}, MACアドレス: {device.address}")

# イベントループでスキャンを実行
loop = asyncio.get_event_loop()
loop.run_until_complete(scan_ble_devices())
