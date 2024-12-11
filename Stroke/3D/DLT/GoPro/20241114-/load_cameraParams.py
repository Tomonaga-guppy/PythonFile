import pickle
import os
import glob
import numpy as np
import pandas as pd
from camera import Camera
import cv2

root_dir = r"G:\gait_pattern\20241114_ota_test\gopro"
# root_dir = r"G:\gait_pattern\20241126_br9g\gopro"
pickle_files = glob.glob(os.path.join(root_dir, "*", "*soln0.pickle"))

CamParamDict = {}
keyPointsDict_3d = {}
keyPointsDict_2d = {}
keyPointsDict_2d_undistort = {}

for pickle_file in pickle_files:
    print(f"pickle_file: {pickle_file}")
    def loadCameraParameters(filename):
        open_file = open(filename, "rb")
        cameraParams = pickle.load(open_file)

        open_file.close()
        return cameraParams

    CamParams = loadCameraParameters(pickle_file)
    """
    # Camera parameters is a dictionary with intrinsics 以下は、カメラパラメータの例
    cameraParams: {
        'distortion': array([[-0.24124805,  0.0489745 ,  0.00072663, -0.02385565,  0.08879946]]),
        'intrinsicMat': array([[2.05357130e+03, 0.00000000e+00, 1.91446909e+03],[0.00000000e+00, 2.01645394e+03, 1.08781011e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'imageSize': array([[2160.],[3840.]]),
        'rotation': array([[-0.00377515,  0.88325695, -0.46887409],[-0.99998129, -0.00107781,  0.00602101],[ 0.00481274,  0.46888805,  0.8832445 ]]),
        'translation': array([[ 546.92724657],[ 766.93473331],[2186.23172209]]),
        'rotation_EulerAngles': array([[ 0.37831574],[-0.38715909],[-1.53922965]])
        }
    """
    CamName = os.path.basename(os.path.dirname(pickle_file))
    CamParamDict[CamName] = CamParams
    print(f"camparandidct: {CamParamDict}")

    keypoints_cheker_path = os.path.join(os.path.dirname(pickle_file), "object3d_twodpoints.csv")
    keypoints_df = pd.read_csv(keypoints_cheker_path, header=0)
    keypointsArray = keypoints_df.values  #[[3x,3y,3z,2x,2y],[]...]  20*5
    keyPointsDict_3d[CamName] = keypointsArray[:, :3]
    # keyPointsDict_2d[CamName] = keypointsArray[:, 3:]

    keypoints_2d_undistort = []
    keypoints_2d = keypointsArray[:, 3:]
    for i, points2d in enumerate(keypoints_2d):
        points2d_undistort = cv2.undistortPoints(points2d, CamParams['intrinsicMat'], CamParams['distortion'], P=CamParams['intrinsicMat'])
        keypoints_2d_undistort = np.append(keypoints_2d_undistort, points2d_undistort.copy()).reshape(-1, 2)
    print(f"keypoints_2d_undistort: {keypoints_2d_undistort}")
    keyPointsDict_2d[CamName] = keypoints_2d_undistort
# print(f"CamParamDict: {CamParamDict}")
print(f"keyPoints2dDict: {keyPointsDict_2d}")


cameraList = []
for camParam in CamParamDict.values():
    # print(f"camParam: {camParam}")
    c = Camera()
    c.set_K(camParam['intrinsicMat'])
    c.set_R(camParam['rotation'])
    c.set_t(np.reshape(camParam['translation'], (3,1)))
    cameraList.append(c)

# triangulate
def triangulatePoints(cameras, twodpoints):
    # print(f"twodpoints: {twodpoints}")

    def _construct_D_block(P, uv,w=1):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """
        # print(f"uv shape: {uv.shape}, P shape: {P.shape}")
        # print(f"uv: {uv}, P: {P}")

        return w*np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))

    def p2e(projective):
        """
        Convert 2d or 3d projective to euclidean coordinates.
        :param projective: projective coordinate(s)
        :type projective: numpy.ndarray, shape=(3 or 4, n)
        :return: euclidean coordinate(s)
        :rtype: numpy.ndarray, shape=(2 or 3, n)
        """
        assert(type(projective) == np.ndarray)
        assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
        return (projective / projective[-1, :])[0:-1, :]

    D = np.zeros((2*len(cameraList), 4))
    # print(f"len(cameras): {len(cameras)}")
    # print(f"cameras: {cameras}")
    # print(f"twodpoints: {twodpoints}")
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, twodpoints):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv)
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    pt3d = p2e(u[:, -1, np.newaxis])

    # print(f"pt3d: {pt3d}")
    return pt3d

# チェッカーボードの3D座標を再構成
point3d_check = []
for [u1, v1], [u2, v2] in zip(keyPointsDict_2d["fl"], keyPointsDict_2d["fr"]):
    twodpoints = np.array([[u1, v1], [u2, v2]])
    points3d_cheker = triangulatePoints(cameraList, twodpoints)
    point3d_check = np.append(point3d_check, points3d_cheker)
    point3d_check = point3d_check.reshape(-1, 3)

print(f"point3d_check: {point3d_check}")

object3d_check = np.array(keyPointsDict_3d["fl"])

error = point3d_check - object3d_check
MAE = np.mean(np.abs(error), axis=0)
print(f"誤差：{error}")
print(f"MAE:{MAE}")

