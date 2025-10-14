import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def debug_triangulation_step_by_step():
    """三角測量の各ステップを詳細にデバッグする"""

    print("=== 三角測量デバッグ開始 ===")

    # データパス
    base_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera0-2")
    stereo_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")

    cam1_op_dir = base_dir / "fl" / "openpose.json"
    cam2_op_dir = base_dir / "fr" / "openpose.json"
    cam1_params = stereo_dir / "fl" / "camera_params_with_ext_OC.json"
    cam2_params = stereo_dir / "fr" / "camera_params_with_ext_OC.json"

    # 1. ファイル存在確認
    print("\n1. ファイル存在確認:")
    print(f"Cam1 OpenPose: {cam1_op_dir.exists()}")
    print(f"Cam2 OpenPose: {cam2_op_dir.exists()}")
    print(f"Cam1 Params: {cam1_params.exists()}")
    print(f"Cam2 Params: {cam2_params.exists()}")

    if not all([cam1_op_dir.exists(), cam2_op_dir.exists(), cam1_params.exists(), cam2_params.exists()]):
        print("❌ 必要なファイルが見つかりません")
        return

    # 2. カメラパラメータ読み込み
    print("\n2. カメラパラメータ確認:")
    try:
        with open(cam1_params, 'r') as f:
            params1 = json.load(f)
        with open(cam2_params, 'r') as f:
            params2 = json.load(f)

        print("Cam1 パラメータキー:", list(params1.keys()))
        print("Cam2 パラメータキー:", list(params2.keys()))

        # 内部パラメータ確認
        K1 = np.array(params1['intrinsics'])
        K2 = np.array(params2['intrinsics'])
        print(f"Cam1 内部パラメータ形状: {K1.shape}")
        print(f"Cam2 内部パラメータ形状: {K2.shape}")
        print("Cam1 K行列:")
        print(K1)
        print("Cam2 K行列:")
        print(K2)

        # 外部パラメータ確認
        R1 = np.array(params1['extrinsics']['rotation_matrix'])
        t1 = np.array(params1['extrinsics']['translation_vector'])
        R2 = np.array(params2['extrinsics']['rotation_matrix'])
        t2 = np.array(params2['extrinsics']['translation_vector'])

        print(f"Cam1 回転行列形状: {R1.shape}, 平行移動ベクトル形状: {t1.shape}")
        print(f"Cam2 回転行列形状: {R2.shape}, 平行移動ベクトル形状: {t2.shape}")

        print("Cam1 回転行列:")
        print(R1)
        print("Cam1 平行移動:")
        print(t1)
        print("Cam2 回転行列:")
        print(R2)
        print("Cam2 平行移動:")
        print(t2)

        # プロジェクション行列作成
        P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])

        print(f"P1形状: {P1.shape}")
        print(f"P2形状: {P2.shape}")
        print("P1:")
        print(P1)
        print("P2:")
        print(P2)

    except Exception as e:
        print(f"❌ カメラパラメータ読み込みエラー: {e}")
        return

    # 3. OpenPoseデータ確認
    print("\n3. OpenPoseデータ確認:")

    # 最初のファイルを詳細チェック
    cam1_files = sorted(list(cam1_op_dir.glob("*_keypoints.json")))
    cam2_files = sorted(list(cam2_op_dir.glob("*_keypoints.json")))

    print(f"Cam1ファイル数: {len(cam1_files)}")
    print(f"Cam2ファイル数: {len(cam2_files)}")

    if len(cam1_files) == 0 or len(cam2_files) == 0:
        print("❌ OpenPoseファイルが見つかりません")
        return

    # 最初のファイルを詳細チェック
    test_file1 = cam1_files[0]
    test_file2 = cam2_files[0]

    print(f"\nテストファイル: {test_file1.name}")

    try:
        with open(test_file1, 'r') as f:
            data1 = json.load(f)
        with open(test_file2, 'r') as f:
            data2 = json.load(f)

        print(f"Cam1 people数: {len(data1.get('people', []))}")
        print(f"Cam2 people数: {len(data2.get('people', []))}")

        if not data1.get('people') or not data2.get('people'):
            print("❌ 人物が検出されていません")
            return

        # キーポイント取得
        kp1_raw = np.array(data1['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        kp2_raw = np.array(data2['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

        print(f"Cam1 キーポイント形状: {kp1_raw.shape}")
        print(f"Cam2 キーポイント形状: {kp2_raw.shape}")

        # RAnkle (index 11) とLAnkle (index 14)を詳細チェック
        rankle1 = kp1_raw[11]  # [x, y, confidence]
        lankle1 = kp1_raw[14]
        rankle2 = kp2_raw[11]
        lankle2 = kp2_raw[14]

        print("\nキーポイント詳細:")
        print(f"Cam1 RAnkle: x={rankle1[0]:.1f}, y={rankle1[1]:.1f}, conf={rankle1[2]:.3f}")
        print(f"Cam1 LAnkle: x={lankle1[0]:.1f}, y={lankle1[1]:.1f}, conf={lankle1[2]:.3f}")
        print(f"Cam2 RAnkle: x={rankle2[0]:.1f}, y={rankle2[1]:.1f}, conf={rankle2[2]:.3f}")
        print(f"Cam2 LAnkle: x={lankle2[0]:.1f}, y={lankle2[1]:.1f}, conf={lankle2[2]:.3f}")

        # 有効性チェック
        rankle_valid = rankle1[2] > 0 and rankle2[2] > 0
        lankle_valid = lankle1[2] > 0 and lankle2[2] > 0

        print(f"RAnkle有効: {rankle_valid}")
        print(f"LAnkle有効: {lankle_valid}")

        if not (rankle_valid or lankle_valid):
            print("❌ 有効な足首キーポイントがありません")
            return

        # 4. 手動三角測量テスト
        print("\n4. 手動三角測量テスト:")

        if rankle_valid:
            print("RAnkleで三角測量テスト...")

            # 2D点
            pt1 = rankle1[:2]  # Cam1のRAnkle
            pt2 = rankle2[:2]  # Cam2のRAnkle

            print(f"Cam1点: ({pt1[0]:.1f}, {pt1[1]:.1f})")
            print(f"Cam2点: ({pt2[0]:.1f}, {pt2[1]:.1f})")

            # OpenCVの三角測量を使用
            try:
                # ホモジニアス座標に変換
                pt1_homo = np.array([[pt1[0]], [pt1[1]]], dtype=np.float32)
                pt2_homo = np.array([[pt2[0]], [pt2[1]]], dtype=np.float32)

                # 三角測量実行
                points_4d = cv2.triangulatePoints(P1, P2, pt1_homo, pt2_homo)

                # 3D座標に変換（正規化）
                if points_4d[3, 0] != 0:
                    point_3d = points_4d[:3, 0] / points_4d[3, 0]
                    print(f"三角測量結果: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f}")

                    # 座標変換後
                    angle_rad = np.radians(180)
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, np.cos(angle_rad), -np.sin(angle_rad)],
                        [0, np.sin(angle_rad), np.cos(angle_rad)]
                    ])
                    rotated_point = rotation_matrix @ point_3d
                    final_point = rotated_point + np.array([-35, 189, 0])
                    print(f"座標変換後: X={final_point[0]:.3f}, Y={final_point[1]:.3f}, Z={final_point[2]:.3f}")

                    # 再投影エラー確認
                    point_3d_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1])

                    # Cam1への再投影
                    proj1 = P1 @ point_3d_homo
                    if proj1[2] != 0:
                        reproj1 = proj1[:2] / proj1[2]
                        error1 = np.linalg.norm(reproj1 - pt1)
                        print(f"Cam1再投影エラー: {error1:.3f} pixels")

                    # Cam2への再投影
                    proj2 = P2 @ point_3d_homo
                    if proj2[2] != 0:
                        reproj2 = proj2[:2] / proj2[2]
                        error2 = np.linalg.norm(reproj2 - pt2)
                        print(f"Cam2再投影エラー: {error2:.3f} pixels")

                else:
                    print("❌ 三角測量失敗: 同次座標の正規化不可")

            except Exception as e:
                print(f"❌ 三角測量エラー: {e}")

        # 5. 複数フレームでのテスト
        print("\n5. 複数フレームでのテスト:")
        test_count = min(5, len(cam1_files))

        for i in range(test_count):
            try:
                with open(cam1_files[i], 'r') as f:
                    data1 = json.load(f)
                with open(cam2_files[i], 'r') as f:
                    data2 = json.load(f)

                if data1.get('people') and data2.get('people'):
                    kp1 = np.array(data1['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
                    kp2 = np.array(data2['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

                    rankle1, rankle2 = kp1[11], kp2[11]
                    valid = rankle1[2] > 0 and rankle2[2] > 0

                    print(f"Frame {i:2d}: RAnkle有効={valid}, "
                          f"Cam1=({rankle1[0]:.1f},{rankle1[1]:.1f}), "
                          f"Cam2=({rankle2[0]:.1f},{rankle2[1]:.1f})")

                    if valid:
                        # 簡易三角測量テスト
                        pt1_homo = np.array([[rankle1[0]], [rankle1[1]]], dtype=np.float32)
                        pt2_homo = np.array([[rankle2[0]], [rankle2[1]]], dtype=np.float32)

                        points_4d = cv2.triangulatePoints(P1, P2, pt1_homo, pt2_homo)
                        if points_4d[3, 0] != 0:
                            point_3d = points_4d[:3, 0] / points_4d[3, 0]
                            print(f"         → 3D: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f})")
                        else:
                            print("         → 三角測量失敗")
                else:
                    print(f"Frame {i:2d}: 人物未検出")
            except Exception as e:
                print(f"Frame {i:2d}: エラー - {e}")

    except Exception as e:
        print(f"❌ OpenPoseデータ処理エラー: {e}")
        import traceback
        traceback.print_exc()

def debug_projection_matrices():
    """プロジェクション行列の妥当性をチェック"""
    print("\n=== プロジェクション行列妥当性チェック ===")

    stereo_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    cam1_params = stereo_dir / "fl" / "camera_params_with_ext_OC.json"
    cam2_params = stereo_dir / "fr" / "camera_params_with_ext_OC.json"

    try:
        with open(cam1_params, 'r') as f:
            params1 = json.load(f)
        with open(cam2_params, 'r') as f:
            params2 = json.load(f)

        # 内部パラメータ
        K1 = np.array(params1['intrinsics'])
        K2 = np.array(params2['intrinsics'])

        # 外部パラメータ
        R1 = np.array(params1['extrinsics']['rotation_matrix'])
        t1 = np.array(params1['extrinsics']['translation_vector'])
        R2 = np.array(params2['extrinsics']['rotation_matrix'])
        t2 = np.array(params2['extrinsics']['translation_vector'])

        print("内部パラメータチェック:")
        print(f"K1の行列式: {np.linalg.det(K1):.6f}")
        print(f"K2の行列式: {np.linalg.det(K2):.6f}")
        print(f"K1の焦点距離: fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}")
        print(f"K2の焦点距離: fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}")
        print(f"K1の主点: cx={K1[0,2]:.1f}, cy={K1[1,2]:.1f}")
        print(f"K2の主点: cx={K2[0,2]:.1f}, cy={K2[1,2]:.1f}")

        print("\n外部パラメータチェック:")
        print(f"R1の行列式: {np.linalg.det(R1):.6f} (1に近いべき)")
        print(f"R2の行列式: {np.linalg.det(R2):.6f} (1に近いべき)")
        print(f"R1の直交性チェック: {np.allclose(R1 @ R1.T, np.eye(3))}")
        print(f"R2の直交性チェック: {np.allclose(R2 @ R2.T, np.eye(3))}")

        # カメラ間の相対位置
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        print(f"\nカメラ間相対位置: {np.linalg.norm(t_rel):.3f} mm")
        print(f"相対回転角度: {np.rad2deg(np.arccos((np.trace(R_rel) - 1) / 2)):.1f} 度")

        # プロジェクション行列の条件数
        P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])

        print(f"\nP1の条件数: {np.linalg.cond(P1):.2e}")
        print(f"P2の条件数: {np.linalg.cond(P2):.2e}")

        # 三角測量のジオメトリチェック
        # カメラの光軸方向
        optical_axis1 = R1[:, 2]  # 3列目がZ軸（光軸）
        optical_axis2 = R2[:, 2]
        angle_between_axes = np.rad2deg(np.arccos(np.dot(optical_axis1, optical_axis2)))
        print(f"カメラ光軸間角度: {angle_between_axes:.1f} 度")

        return True

    except Exception as e:
        print(f"❌ プロジェクション行列チェックエラー: {e}")
        return False

if __name__ == "__main__":
    debug_triangulation_step_by_step()
    debug_projection_matrices()