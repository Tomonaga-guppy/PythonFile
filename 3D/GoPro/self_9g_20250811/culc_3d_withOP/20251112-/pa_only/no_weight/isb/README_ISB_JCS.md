# ISB推奨 Joint Coordinate System (JCS) への修正ガイド

## 概要

このドキュメントでは、関節角度計算プログラムをISB（International Society of Biomechanics）推奨の
Joint Coordinate System (JCS) に基づく方法に修正した内容を説明します。

## 参考文献

1. **Wu, G., et al. (2002)**  
   "ISB recommendation on definitions of joint coordinate system of various joints 
   for the reporting of human joint motion—part I: ankle, hip, and spine."  
   *Journal of Biomechanics*, 35(4), 543-548.

2. **Grood, E.S., & Suntay, W.J. (1983)**  
   "A joint coordinate system for the clinical description of three-dimensional motions: 
   application to the knee."  
   *Journal of Biomechanical Engineering*, 105(2), 136-144.

---

## 主な変更点

### 1. 座標系の定義（ISB推奨）

#### 元のコード
```python
# 骨盤座標系
e_x0_pelvis_0 = (hip_0 - sacrum)/np.linalg.norm(hip_0 - sacrum)  # 前方
e_y_pelvis_0 = (lasi - rasi)/np.linalg.norm(lasi - rasi)         # 左方
e_z_pelvis_0 = np.cross(e_x0_pelvis_0, e_y_pelvis_0)             # 上方
```

#### ISB推奨（修正後）
```python
# 骨盤座標系 (XYZ)
# Z軸: 右向き（左右ASIS間に平行）
z_axis = normalize(rasi - lasi)
# X軸: 前向き（骨盤平面内、Z軸に直交）
x_axis = normalize(np.cross(y_axis, z_axis))
# Y軸: 頭側向き（XとZに垂直）
y_axis = normalize(np.cross(z_axis, x_axis))
```

### 2. 角度計算方法

#### 元のコード（オイラー角分解）
```python
# 相対回転行列を計算
r_hip_relative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_pelvis)

# オイラー角 'YZX' で分解
r_hip_angle_rot = R.from_matrix(r_hip_relative_rotation)
r_hip_angle_flex = r_hip_angle_rot.as_euler('YZX', degrees=True)[0]
r_hip_angle_inex = r_hip_angle_rot.as_euler('YZX', degrees=True)[1]
r_hip_angle_adab = r_hip_angle_rot.as_euler('YZX', degrees=True)[2]
```

#### ISB推奨（Grood & Suntay JCS）
```python
# JCS軸の定義
e1 = pelvis_axes[:, 2]   # 骨盤のZ軸（固定軸）
e3 = femur_axes[:, 1]    # 大腿のy軸（固定軸）
e2 = normalize(np.cross(e3, e1))  # 浮動軸

# 各軸周りの角度を直接計算
# α (屈曲/伸展): e1軸周りの回転
flexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))

# β (内転/外転): e2軸（浮動軸）周りの回転
adduction = np.degrees(np.arcsin(-np.dot(e1, e3)))

# γ (内旋/外旋): e3軸周りの回転
internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
```

---

## ISB JCSの概念

### Joint Coordinate System (JCS) の構成

```
      近位セグメント
           │
           │ e1 (固定軸)
           ▼
    ──────────────────
    │                │
    │   e2 (浮動軸)  │  ← e2 = e3 × e1
    │       ↓        │
    ──────────────────
           │
           │ e3 (固定軸)
           ▼
      遠位セグメント
```

### 各関節のJCS定義

#### 股関節
| 軸 | 固定先 | 方向 | 運動 |
|---|-------|-----|-----|
| e1 | 骨盤Z軸 | 右向き | 屈曲/伸展 (α) |
| e2 | 浮動軸 | e3×e1 | 内転/外転 (β) |
| e3 | 大腿y軸 | 頭側向き | 内旋/外旋 (γ) |

#### 膝関節
| 軸 | 固定先 | 方向 | 運動 |
|---|-------|-----|-----|
| e1 | 大腿z軸 | 内外側 | 屈曲/伸展 |
| e2 | 浮動軸 | e3×e1 | 内反/外反 |
| e3 | 脛骨Y軸 | 頭側向き | 内旋/外旋 |

#### 足関節複合体
| 軸 | 固定先 | 方向 | 運動 |
|---|-------|-----|-----|
| e1 | 脛骨Z軸 | 内外側 | 背屈/底屈 |
| e2 | 浮動軸 | e3×e1 | 内反/外反 |
| e3 | 踵骨y軸 | 頭側向き | 内旋/外旋 |

---

## 符号規約（正の値の意味）

| 関節 | 屈曲/伸展 | 内転/外転 | 内旋/外旋 |
|-----|---------|---------|---------|
| 股関節 | 屈曲(+) / 伸展(-) | 内転(+) / 外転(-) | 内旋(+) / 外旋(-) |
| 膝関節 | 屈曲(+) / 伸展(-) | 内反(+) / 外反(-) | 内旋(+) / 外旋(-) |
| 足関節 | 背屈(+) / 底屈(-) | 内反(+) / 外反(-) | 内旋(+) / 外旋(-) |

---

## ファイル構成

```
├── isb_joint_angles.py      # ISB JCS関節角度計算モジュール（再利用可能）
├── joint_angles_isb.py      # 使用例・統合プログラム
├── m_opti.py                # 元のモーションキャプチャデータ読み込み
└── README_ISB_JCS.md        # このドキュメント
```

---

## 使用方法

### 基本的な使用

```python
from joint_angles_isb import process_mocap_data_isb
import m_opti as opti

# データ読み込み
keypoints_mocap, full_range, start_frame, end_frame = opti.read_3d_optitrack(
    csv_path, start_frame, end_frame
)

# ISB JCS関節角度計算
angles_df = process_mocap_data_isb(
    keypoints_mocap, 
    full_range,
    sampling_freq=100,
    filter_order=4,
    cutoff_freq=6,
    marker_radius=0.0127,
    height=1.76
)

# 結果の確認
print(angles_df.head())
```

### モジュールとして使用

```python
from isb_joint_angles import ISBJointAngles

# インスタンス作成
calculator = ISBJointAngles(marker_radius=0.0127, height=1.76)

# 1フレームの角度計算
angles = calculator.calculate_all_angles(
    rasi, lasi, rpsi, lpsi,
    rknee, rknee2, lknee, lknee2,
    rank, rank2, lank, lank2,
    rtoe, rhee, ltoe, lhee
)

print(f"右股関節屈曲角度: {angles['R_Hip_FlEx']:.1f}°")
```

---

## マーカー配置の注意点

### 必要なマーカー

| マーカー名 | 位置 | 役割 |
|----------|-----|-----|
| RASI/LASI | 上前腸骨棘 | 骨盤座標系 |
| RPSI/LPSI | 上後腸骨棘 | 骨盤座標系 |
| RKNE/LKNE | 膝関節**外側** | 大腿・下腿座標系 |
| RKNE2/LKNE2 | 膝関節**内側** | 大腿・下腿座標系 |
| RANK/LANK | **外踝** | 下腿・足部座標系 |
| RANK2/LANK2 | **内踝** | 下腿・足部座標系 |
| RTOE/LTOE | つま先 | 足部座標系 |
| RHEE/LHEE | 踵 | 足部座標系 |

### マーカー配置の確認

コード内のマーカーインデックスを確認してください：

```python
# インデックス配置（m_opti.pyに基づく）
# 0:LANK(外踝), 1:LANK2(内踝), 2:LASI, 3:LHEE, 4:LKNE(外側), 5:LKNE2(内側), 6:LPSI, 7:LTOE
# 8:RANK(外踝), 9:RANK2(内踝), 10:RASI, 11:RHEE, 12:RKNE(外側), 13:RKNE2(内側), 14:RPSI, 15:RTOE
```

---

## オイラー角方式との違い

### オイラー角方式の問題点

1. **ジンバルロック**: 特定の角度付近で不安定になる
2. **順序依存性**: 回転順序によって結果が変わる
3. **臨床的解釈の困難さ**: 臨床用語と対応しにくい

### JCS方式の利点

1. **浮動軸の使用**: ジンバルロックを回避
2. **臨床的解釈**: 屈曲/伸展、内転/外転、内旋/外旋が直接得られる
3. **標準化**: ISBの国際標準に準拠

---

## 注意事項

1. **座標系の向き**: OptiTrackの座標系とISB座標系の対応を確認してください
2. **マーカー名**: 内側/外側の区別を正しく行ってください
3. **左右の対称性**: 左脚の角度は符号が反転する場合があります
4. **中立位の定義**: 被験者の解剖学的中立位を基準としています

---

## トラブルシューティング

### 角度が大きくずれる場合

1. マーカーの内側/外側が逆になっていないか確認
2. 座標系の向き（X, Y, Z）が正しいか確認
3. OptiTrackの出力座標系を確認

### NaNが発生する場合

1. マーカーデータの欠損を確認
2. ベクトルの正規化でゼロ除算が発生していないか確認

---

## 更新履歴

- 2024年: ISB JCS方式への修正版を作成
