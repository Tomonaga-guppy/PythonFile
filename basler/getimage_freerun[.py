import time
from pypylon import pylon

# トランスポートレイヤーインスタンスを取得
tl_factory = pylon.TlFactory.GetInstance()

# カメラのインスタンスを作成
camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())

# カメラを開く
camera.Open()

# フリーランモードでの画像取得を開始
camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

camera.AcquisitionFrameRateEnable = True

# 取得したフレーム数のカウンタ
frame_count = 0

# フレームの取得開始
print('画像取得を開始します')
start_time = time.time()

# カメラがグラビング中かどうかを確認
while camera.IsGrabbing():
    # 結果を取得
    grab = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)

    # 取得が成功した場合
    if grab.GrabSucceeded():
        frame_count += 1

    # 100フレーム取得したら停止
    if frame_count == 100:
        break

print(f'{frame_count}フレームを{time.time() - start_time:.0f}秒で取得しました')

# カメラを閉じる
camera.Close()
