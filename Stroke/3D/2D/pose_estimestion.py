import os
import subprocess
from pathlib import Path

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro")
ori_mov_paths = list(root_dir.glob(f"*sagi\*int_cali.MP4"))
print(f"ori_mov_paths: {ori_mov_paths}")


os.chdir(r"C:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更

for i, ori_mov_path in enumerate(ori_mov_paths):
    dir_name = ori_mov_path.parent
    print(f"dir_name: {dir_name}")
    print(f"{i+1}/{len(ori_mov_paths)}: {ori_mov_path}")
    stem_name = str(ori_mov_path.stem)

    #動画を画像にして保存
    original_imgs_dir = dir_name / f"{stem_name}_img"
    if not original_imgs_dir.exists():
        os.makedirs(original_imgs_dir)
        cmd = f"ffmpeg -i {ori_mov_path} {original_imgs_dir / f'%04d.jpg'}"
        subprocess.run(cmd)

    if Path(dir_name / f"{stem_name}_op.avi").exists():
        print(f"{stem_name}_op.avi はすでに存在します") #すでに推定済みの場合はスキップ
        continue

    # ######OpenPoseへの命令作成 windowsの場合
    # program= r".\build\x64\Release\OpenPoseDemo.exe"
    # pre_video_place= f" --video " + str(ori_mov_path)  #動画の場所
    # after_video_place= f" --write_video {dir_name / f'{stem_name}_op.avi'}"  #動画で出力
    # after_images_place= f" --write_images {dir_name / f'{stem_name}_op'}"  #画像で出力
    # images_format = f" --write_images_format jpg"  #画像出力のフォーマットを指定
    # after_json_place= f" --write_json {dir_name / f'{stem_name}_op.json'}"  #各キーポイントの座標をjsonで出力
    # other_order= f" --number_people_max 4 --scale_number 2 --scale_gap 0.2"
    # cmd =program + pre_video_place + after_video_place + after_images_place + images_format + after_json_place + other_order
    # ######OpenPoseへの命令作成終了


    # subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行