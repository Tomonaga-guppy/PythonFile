import numpy as np
import cv2
import torch
from PIL import Image

from transformers.models.vitpose import VitPoseImageProcessor, VitPoseForPoseEstimation

MODEL_ID = "usyd-community/vitpose-plus-large"
IMAGE_PATH = r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fl\undistorted_facemasked\frame_00000.png"
DATASET_INDEX = 5  # WholeBody

device = "cuda" if torch.cuda.is_available() else "cpu"

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)
h, w = img_bgr.shape[:2]
pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

processor = VitPoseImageProcessor.from_pretrained(MODEL_ID)
model = VitPoseForPoseEstimation.from_pretrained(MODEL_ID).to(device).eval()

person_boxes = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
inputs = processor(pil, boxes=[person_boxes], return_tensors="pt").to(device)

dataset_index = torch.tensor([DATASET_INDEX], device=device)

with torch.no_grad():
    outputs = model(**inputs, dataset_index=dataset_index)

# ===== ここが重要：モデル出力を直接見る =====
print("outputs keys:", outputs.keys())
# 代表的に logits / heatmaps を探す
for k in ["logits", "heatmaps", "preds", "keypoint_logits"]:
    if k in outputs:
        t = outputs[k]
        print(k, "shape:", tuple(t.shape), "dtype:", t.dtype)
# =========================================

pose_results = processor.post_process_pose_estimation(
    outputs,
    boxes=[person_boxes],
    threshold=0.0,
)

people = pose_results[0]
print("num people:", len(people))

p0 = people[0]
kpts = p0["keypoints"].cpu().numpy()
scores = p0["scores"].cpu().numpy()
print("postprocess K =", kpts.shape[0])
