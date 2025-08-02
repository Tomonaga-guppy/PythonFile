import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PoseLandmarkerのセットアップ (変更なし) ---

# モデルファイルのパス
model_path = r"C:\Users\Tomson\.vscode\PythonFile\Stroke\MediaPipe\model\pose_landmarker_heavy.task"

# オプションを設定
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

# PoseLandmarkerインスタンスを作成
landmarker = vision.PoseLandmarker.create_from_options(options)


# --- 2. 描画関連の準備 (変更なし) ---

# 描画ユーティリティと接続情報
mp_drawing = mp.solutions.drawing_utils
pose_connections = mp.solutions.pose.POSE_CONNECTIONS

# 3Dプロットの準備
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.ion()


# --- 3. 動画処理 ---

# 動画ファイルの読み込み
video_path = r"G:\gait_pattern\20250228_ota\data\20250221\sub0\thera0-3\sagi\Undistort.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
    exit()

print("動画の処理を開始します...'q'キーで終了します。")

# 軸の範囲を固定して見やすくする
AXIS_MIN = -0.75
AXIS_MAX = 0.75

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("動画の最後まで到達しました。")
        break

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    annotated_image = frame.copy()

    # --- 2Dランドマークの描画 (変更なし) ---
    if detection_result.pose_landmarks:
        pose_landmarks_list = detection_result.pose_landmarks[0]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            pose_connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # ================================================================= #
    # --- 3Dランドマークの描画 (★★★ここを修正★★★) ---
    # ================================================================= #
    ax.clear() # 毎回プロットをクリア
    ax.set_title("3D Pose Estimation")
    ax.set_xlabel("X")
    ax.set_ylabel("Z") # Z軸（奥行き）
    ax.set_zlabel("-Y") # Y軸（高さ）。MediaPipeはYが下向きなので-Yで表示

    # 軸の範囲を設定
    ax.set_xlim([AXIS_MIN, AXIS_MAX])
    ax.set_ylim([AXIS_MIN, AXIS_MAX])
    ax.set_zlim([AXIS_MIN, AXIS_MAX])

    if detection_result.pose_world_landmarks:
        # ランドマークの座標を取得
        landmarks = detection_result.pose_world_landmarks[0]

        # 点を描画 (scatter)
        for landmark in landmarks:
            ax.scatter(landmark.x, landmark.z, -landmark.y, c='lime', marker='o')

        # 線を描画 (plot)
        for connection in pose_connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                ax.plot([landmarks[start_idx].x, landmarks[end_idx].x],
                        [landmarks[start_idx].z, landmarks[end_idx].z],
                        [-landmarks[start_idx].y, -landmarks[end_idx].y], 'c-') # c-はシアンの線
    else:
        # 3Dランドマークが検出されなかった場合
        print(f"フレーム {timestamp_ms} [ms]: 3Dランドマークが検出されませんでした。")


    # プロットを更新
    ax.view_init(elev=10., azim=80)
    plt.pause(0.001)

    # --- 結果の表示 (変更なし) ---
    mini_img = cv2.resize(annotated_image, (640, 480))
    cv2.imshow('MediaPipe Pose Landmarker (2D)', mini_img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 後処理 ---
landmarker.close()
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show() # 最後にプロットウィンドウを保持