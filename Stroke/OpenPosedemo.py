import subprocess

input_path = r"c:\Users\Tomson\openpose\examples\media\video.avi"
out_dir = r"c:\Users\Tomson\BRLAB\Stroke\pretest"

gtcwd = r'c:\Users\Tomson\openpose'
import os
os.chdir(gtcwd)

# OpenFaceで顔の推定
command = r'c:\Users\Tomson\openpose '
inputpath = '--video ' + input_path + ' '
outpath = '-out_dir ' + out_dir
print(f"running_command = {command + inputpath + outpath}")

subprocess.run (command + inputpath + outpath)
print(f"running_command = {command + inputpath + outpath}")