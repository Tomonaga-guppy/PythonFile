import os
import subprocess
from pathlib import Path
import sys

root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
ori_imgs_folder_paths = list(root_dir.glob(fr"**\Undistort"))

# フォルダ名が 'fl' または 'fr' の動画ファイルのみを抽出
ori_imgs_folder_paths = [p for p in ori_imgs_folder_paths if p.parent.name == 'fl' or p.parent.name == 'fr']


print(f"ori_imgs_folder_paths: {ori_imgs_folder_paths}" )

# sys.exit()

os.chdir(r"C:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更

for i, ori_imgs_folder_path in enumerate(ori_imgs_folder_paths):
    dir_name = ori_imgs_folder_path.parent
    print(f"dir_name: {dir_name}")
    print(f"{i+1}/{len(ori_imgs_folder_paths)}: {ori_imgs_folder_path}")
    stem_name = str(ori_imgs_folder_path.stem)

    # #動画を画像にして保存（切り出す場合はフレーム数が変化するので一旦消した）
    # original_imgs_dir = dir_name / f"{stem_name}_img"
    # if not original_imgs_dir.exists():
    #     os.makedirs(original_imgs_dir)

    #     cmd = f"ffmpeg -i {ori_mov_path} {original_imgs_dir / f'%04d.jpg'}"
    #     subprocess.run(cmd)

    if Path(dir_name / f"{stem_name}_op.avi").exists():
        print(f"{stem_name}_op.avi はすでに存在します") #すでに推定済みの場合はスキップ
        continue

    condition = dir_name.parent.name  #条件名を取得
    print(f"condition: {condition}")

    if condition == "thera0-3":
        max_people = 1
    else:
        max_people = 2

    file_name = stem_name + "_op"

    ######OpenPoseへの命令作成 windowsの場合
    program= r".\build\x64\Release\OpenPoseDemo.exe"
    pre_img_dir= f" --image_dir " + str(ori_imgs_folder_path)  #動画の場所
    fps = f" --write_video_fps 60"
    after_video_place= f" --write_video {dir_name / f'{file_name}.avi'}"  #動画で出力
    after_images_place= f" --write_images {dir_name / f'{file_name}'}"  #画像で出力
    images_format = f" --write_images_format jpg"  #画像出力のフォーマットを指定
    after_json_place= f" --write_json {dir_name / f'{file_name}.json'}"  #各キーポイントの座標をjsonで出力
    other_order= f" --number_people_max {max_people} --scale_number 2 --scale_gap 0.2"
    cmd =program + pre_img_dir + fps + after_video_place + after_images_place + images_format + after_json_place + other_order
    ######OpenPoseへの命令作成終了


    subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行