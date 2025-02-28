import pickle
import json
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
import copy
import cv2

keypoints_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow",
                "LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe",
                    "RSmallToe","RHeel"]

def loadCameraParameters(filename):
    with open(filename
              , "rb") as f:
        CameraParams_dict = pickle.load(f)
    return CameraParams_dict

def mkCSVOpenposeData(openpose_dir, overwrite=True):
    # print(f"openpose_dir:{openpose_dir}")
    condition = (openpose_dir.stem).split("_op")[0]
    # すでにcsvファイルがあれば処理を終了する
    output_csvs = [openpose_dir.with_name("keypoints2d_"  + condition  + f"_{i}.csv") for i in range(2)]
    if overwrite and (output_csvs[0].exists() or output_csvs[1].exists()):
            print(f"csvファイル{output_csvs}を上書きします。")
            for output_csv in output_csvs:
                output_csv.unlink()
    elif overwrite == False and output_csvs[0].exists() and output_csvs[1].exists():
        print(f"以前の{output_csvs}を再利用します。")
        return output_csvs

    json_files = list(openpose_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"jsonファイルが見つかりませんでした。")
        return
    # print(f"jsonファイル:{json_files}")

    csv_header = []

    for keypoint in keypoints_name:
        for n in ("x","y","p"):
            csv_header.append(f"{keypoint}_{n}")
    csv_header.insert(0, "frame_num")

    header_write_list = [False, False]
    for ijson, json_file in tqdm(enumerate(json_files)):
        # print(f"{ijson+1}/{len(json_files)} {json_file}を処理中")

        with open(json_file, "r") as f_json:
            json_data = json.load(f_json)
        all_people_data = json_data["people"]

        for ipeople, data in enumerate(all_people_data):
            # person_id = people["person_id"]  #これを使いたいがなぜかすべて-1になる
            person_id = str(ipeople)
            # print(f"person_id:{person_id}")
            output_csv = output_csvs[ipeople]
            # print(f"output_csv_dir:{output_csv_dir}")
            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if not header_write_list[ipeople]:
                    writer.writerow(csv_header)
                    header_write_list[ipeople] = True
                pose_keypoints_2d = data["pose_keypoints_2d"]
                pose_keypoints_2d_str = [str(value) for value in pose_keypoints_2d]
                pose_keypoints_2d_str.insert(0, str(ijson))
                # print(f"pose_keypoints_2d:{pose_keypoints_2d_str}")
                writer.writerow(pose_keypoints_2d_str)
    print(f"csvファイルの作成が完了しました。")
    return output_csvs

def mkFrameCheckCSV(frame_ch_csv):
    with open(frame_ch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RiseHandFrame", "StartWalkFrame", "0m"])
    print(f"{frame_ch_csv}を作成しました。")

def undistordOpenposeData(openpose_df, CamPrams_dict):
    print(f"openpose_df_dict:{openpose_df}")
    print(f"CamPrams_dict:{CamPrams_dict}")

    for keypoint_name in keypoints_name:
        points = np.array([openpose_df[keypoint_name+"_x"], openpose_df[keypoint_name+"_y"]]).T
        undistort_points =cv2.undistortPoints(points, CamPrams_dict["intrinsicMat"], CamPrams_dict["distortion"], P=CamPrams_dict["intrinsicMat"])
        print(f"openpose_df[keypoint?name+'_x].shape:{openpose_df[keypoint_name+'_x'].shape}")
    return openpose_df

def adjustOpenposeDF(openpose_df_dict, walk_start_frame):
    # print(f"openpose_df_dict.items():{openpose_df_dict.items()}")
    stop_frame = 0
    for key, df_list in openpose_df_dict.items():
        for i, df in enumerate(df_list):
            end_frame = df.index[-1]
            if end_frame > stop_frame:
                stop_frame = end_frame
    # print(f"stop_frame:{stop_frame}")
    for key, df_list in openpose_df_dict.items():
        for i, df in enumerate(df_list):
            df = df.reindex(range(walk_start_frame, stop_frame+1), fill_value=0)
            openpose_df_dict[key][i] = df
    # print(f"openpose_df_dict:{openpose_df_dict}")
    return openpose_df_dict

class Camera:
    def __init__(self):
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        self.update_P()

    def update_P(self):
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def set_K(self, K):
        self.K = K
        self.update_P()

    def set_R(self, R):
        self.R = R
        self.update_P()

    def set_t(self, t):
        self.t = t
        self.update_P()


def setCameraList(CamPrams_dict):
    # print(f"camParams_dict:{CamPrams_dict}")
    camera_list = []
    for key, camParams in CamPrams_dict.items():
        # print(f"camParams:{camParams}")
        c = Camera()
        c.set_K(camParams["intrinsicMat"])
        c.set_R(camParams["rotation"])
        c.set_t(camParams["translation"])
        camera_list.append(c)
    return camera_list

def p2e(projective):
    return (projective / projective[-1, :])[0:-1, :]

def cul_3dkeypoints(keypoints2d, possibilites, camera_list):
    # print(f"keypoints2d:{keypoints2d}")
    # print(f"possibilites:{possibilites}")
    # print(f"camera_list:{camera_list}")
    def _construct_D_block(P, uv,w=1):
        # print(f"P:{P}")
        # print(f"uv:{uv}")
        # print(f"w:{w}")
        return w*np.vstack((uv[0] * P[2, :] - P[0, :],
                            uv[1] * P[2, :] - P[1, :]))
    D = np.zeros((len(camera_list) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(camera_list)), camera_list, keypoints2d.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv,w=possibilites[cam_idx])
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    pt3d = p2e(u[:, -1, np.newaxis])
    return pt3d

def cul_3DKeyPoints(keypoints2d_df_dict, CamPrams_dict):
    # 3次元再構成で使用しやすい形にカメラパラメータを変換
    camera_list = setCameraList(CamPrams_dict)
    # 3次元座標を格納する辞書を作成
    keypoint3d_dict = {"person0":[],"person1":[]}
    keypoints_3dname = [keypoints_name[i]+f"_{name}" for i in range(len(keypoints_name)) for name in ["X","Y","Z"]]
    keypoints_3ddic = {key:[] for key in keypoints_3dname}
    keypoint3d_dict = {"person0":copy.deepcopy(keypoints_3ddic),"person1":copy.deepcopy(keypoints_3ddic)}

    # 各フレームごとに3次元座標を求める
    frame_range = keypoints2d_df_dict["fl"][0].index
    for iPeople in range(2):
        print(f"person{iPeople}の3次元座標を求めます。")
        for frame_num in tqdm(frame_range):
            for keypoint in keypoints_name:
                keypoints_2d = np.array([[keypoints2d_df_dict["fl"][iPeople].loc[frame_num, keypoint+"_x"],
                                        keypoints2d_df_dict["fl"][iPeople].loc[frame_num, keypoint+"_y"]],
                                        [keypoints2d_df_dict["fr"][iPeople].loc[frame_num, keypoint+"_x"],
                                        keypoints2d_df_dict["fr"][iPeople].loc[frame_num, keypoint+"_y"]]])
                possibilites_2d = np.array([keypoints2d_df_dict["fl"][iPeople].loc[frame_num, keypoint+"_p"],
                                        keypoints2d_df_dict["fr"][iPeople].loc[frame_num, keypoint+"_p"]])
                keypoints_3d = cul_3dkeypoints(keypoints_2d, possibilites_2d, camera_list)
                keypoint3d_dict[f"person{iPeople}"][keypoint+"_X"].append(keypoints_3d[0].item())
                keypoint3d_dict[f"person{iPeople}"][keypoint+"_Y"].append(keypoints_3d[1].item())
                keypoint3d_dict[f"person{iPeople}"][keypoint+"_Z"].append(keypoints_3d[2].item())
    return keypoint3d_dict



