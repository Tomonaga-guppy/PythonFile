from pathlib import Path
import pickle

# pickle_file1 = r"G:\gait_pattern\int_cali\tkrzk_9g\Intrinsic_fl75.pickle"
# pickle_file2 = r"G:\gait_pattern\int_cali\tkrzk_9g\Intrinsic_fr.pickle"
# pickle_files = [pickle_file1, pickle_file2]

root_dir = Path(r"G:\gait_pattern\20241112(1)\gopro")
pickle_files = list(root_dir.glob("*Intrinsic*.pickle"))

for pickle_file in pickle_files:
    print(f"pickle_file: {pickle_file}")
    def loadCameraParameters(filename):
        open_file = open(filename, "rb")
        cameraParams = pickle.load(open_file)

        open_file.close()
        return cameraParams

    CamParams = loadCameraParameters(pickle_file)

    print(f"camparandidct: {CamParams}")