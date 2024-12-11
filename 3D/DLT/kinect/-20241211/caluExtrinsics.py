import cv2
import numpy as np
import glob
import os
import re
import pickle
import copy

root_dir = r"G:\gait_pattern\20241106"
cali_extrinsic_dirs = glob.glob(os.path.join(root_dir, "*cali_1080*"))
cali_extrinsic_dirs = [cali_extrinsic_dir for cali_extrinsic_dir in cali_extrinsic_dirs if os.path.isdir(cali_extrinsic_dir)]
print(cali_extrinsic_dirs)
visualize = True

def generate3Dgrid(checker_pattern, squareSize):
    #  3D points real world coordinates. Assuming z=0
    objectp3d = np.zeros((1, checker_pattern[0]
                          * checker_pattern[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:checker_pattern[0],
                                    0:checker_pattern[1]].T.reshape(-1, 2)
    objectp3d = objectp3d * squareSize
    return objectp3d

def saveCameraParameters(filename,CameraParams):
    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()
    return True


def main():
    # Camera parameters is a dictionary with intrinsics

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []
    # Vector for 2D points
    twodpoints = []

    checker_pattern = (4, 5)
    squareSize = 35  # mm

    #  3D points real world coordinates. Assuming z=0
    objectp3d = generate3Dgrid(checker_pattern, squareSize)

    useSecondExtrinsicsSolution = False

    for cali_extrinsic_dir in cali_extrinsic_dirs:
        # Load and resize image - remember calibration image res needs to be same as all processing
        imageFileName = os.path.join(cali_extrinsic_dir, "original", "120.png")
        image = cv2.imread(imageFileName)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true

        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Note I tried findChessboardCornersSB here, but it didn't find chessboard as reliably
        ret, corners = cv2.findChessboardCorners(
                    grayColor, checker_pattern, cv2.CALIB_CB_ADAPTIVE_THRESH)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            # 3D points real world coordinates
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # For testing: Draw and display the corners
            # image = cv2.drawChessboardCorners(image,
            #                                  CheckerBoardParams['dimensions'],
            #                                   corners2, ret)
            # Draw small dots instead
            # Choose dot size based on size of squares in pixels
            circleSize = 1
            squareSize = np.linalg.norm((corners2[1,0,:] - corners2[0,0,:]).squeeze())
            if squareSize >12:
                circleSize = 2

            for iPoint in range(corners2.shape[0]):
                thisPt = corners2[iPoint,:,:].squeeze()
                cv2.circle(image, tuple(thisPt.astype(int)), circleSize, (255,255,0), 2)

        if ret == False:
            print('No checkerboard detected. Will skip cam in triangulation.')
            return None

        # This function gives two possible solutions.
        # It helps with the ambiguous cases with small checkerboards (appears like
        # left handed coord system). Unfortunately, there isn't a clear way to
        # choose the correct solution. It is the nature of the solvePnP problem
        # with a bit of 2D point noise.

        id = os.path.basename(cali_extrinsic_dir).split("cali_")[-1]
        CameraParams_path = os.path.join(root_dir, f"Intrinsics_{id}.pickle")
        with open(CameraParams_path, "rb") as f:
            CameraParams = pickle.load(f)

        rets, rvecs, tvecs, reprojError = cv2.solvePnPGeneric(
            objectp3d, corners2, CameraParams['intrinsicMat'],
            CameraParams['distortion'], flags=cv2.SOLVEPNP_IPPE)
        rvec = rvecs[1]
        tvec = tvecs[1]

        if rets < 1 or np.max(rvec) == 0 or np.max(tvec) == 0:
            print('solvePnPGeneric failed. Use SolvePnPRansac')
            # Note: can input extrinsics guess if we generally know where they are.
            # Add to lists to look like solvePnPRansac results
            rvecs = []
            tvecs = []
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectp3d, corners2, CameraParams['intrinsicMat'],
                CameraParams['distortion'])
            if ret is True:
                rets = 1
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                print('Extrinsic calculation failed. Will skip cam in triangulation.')
                return None

        # Select which extrinsics solution to use
        extrinsicsSolutionToUse = 0
        if useSecondExtrinsicsSolution:
            extrinsicsSolutionToUse = 1

        topLevelExtrinsicImageFolder = os.path.join(cali_extrinsic_dir, 'extrinsicCalib')
        if not os.path.exists(topLevelExtrinsicImageFolder):
            os.makedirs(topLevelExtrinsicImageFolder,exist_ok=True)

        for iRet,rvec,tvec in zip(range(rets),rvecs,tvecs):
            theseCameraParams = copy.deepcopy(CameraParams)
            # Show reprojections
            img_points, _ = cv2.projectPoints(objectp3d, rvec, tvec,
                                            CameraParams['intrinsicMat'],
                                            CameraParams['distortion'])

            # Plot reprojected points
            # for c in img_points.squeeze():
            #     cv2.circle(image, tuple(c.astype(int)), 2, (0, 255, 0), 2)

            # Show object coordinate system
            imageCopy = copy.deepcopy(image)
            imageWithFrame = cv2.drawFrameAxes(
                imageCopy, CameraParams['intrinsicMat'],
                CameraParams['distortion'], rvec, tvec, 200, 4)

            # Create zoomed version.
            ht = image.shape[0]
            wd = image.shape[1]
            bufferVal = 0.05 * np.mean([ht,wd])
            topEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[1]-bufferVal,0]))
            leftEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[0]-bufferVal,0]))
            bottomEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[1]+bufferVal,ht]))
            rightEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[0]+bufferVal,wd]))

            # imageCopy2 = copy.deepcopy(imageWithFrame)
            imageCropped = imageCopy[topEdge:bottomEdge,leftEdge:rightEdge,:]


            # Save extrinsics picture with axis
            imageSize = np.shape(image)[0:2]
            #findAspectRatio
            ar = imageSize[1]/imageSize[0]
            # cv2.namedWindow("axis", cv2.WINDOW_NORMAL)
            cv2.resize(imageWithFrame,(600,int(np.round(600*ar))))

            # save crop image to local camera file
            savePath2 = os.path.join(os.path.dirname(imageFileName),
                                    'extrinsicCalib_soln{}.jpg'.format(iRet))
            cv2.imwrite(savePath2,imageCropped)

            if visualize:
                print('Close image window to continue')
                cv2.imshow('axis', image)
                cv2.waitKey()

                cv2.destroyAllWindows()

            R_worldFromCamera = cv2.Rodrigues(rvec)[0]

            theseCameraParams['rotation'] = R_worldFromCamera
            theseCameraParams['translation'] = tvec
            theseCameraParams['rotation_EulerAngles'] = rvec

            # save extrinsics parameters to video folder
            # will save the selected parameters in Camera folder in main
            saveExtPath = os.path.join(
                os.path.dirname(imageFileName),
                'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(iRet))
            saveCameraParameters(saveExtPath,theseCameraParams)

            # save images to top level folder and return correct extrinsics
            camName = os.path.split(os.path.abspath(
                    os.path.join(os.path.dirname(imageFileName), '../../')))[1]

            if iRet == extrinsicsSolutionToUse:
                fullCamName = camName
                CameraParamsToUse = copy.deepcopy(theseCameraParams)
            else:
                fullCamName = 'altSoln_{}'.format(camName)
            savePath = os.path.join(topLevelExtrinsicImageFolder, 'extrinsicCalib_'
                                    + fullCamName + '.jpg')
            cv2.imwrite(savePath,imageCropped)


if __name__ == '__main__':
    main()