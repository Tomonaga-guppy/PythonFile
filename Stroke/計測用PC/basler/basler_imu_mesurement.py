import subprocess

root_dir = r"C:\Users\tus\.vscode\python_scripts\basler"

# スクリプトのパス
control_imu_script = root_dir + r"\control_imu.py"
test6_save_imgs_script = root_dir + r"\test6_save_imgs.py"

commands = [
    "python " + control_imu_script,
    "python " + test6_save_imgs_script
]

for command in commands:
    subprocess.Popen(f'start cmd /k "{command}"', shell=True)