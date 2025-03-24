from pathlib import Path
import subprocess

sub_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
condition_list = ["thera0-3", "thera1-1", "thera2-1"]

for condition in condition_list:
    mp4_dir = sub_dir / condition / "sagi"
    mp4_file = list(Path(mp4_dir).glob("GX*.MP4"))[0]

    output_dir_name = "Ori_imgs"
    output_dir = Path(mp4_file.parent, output_dir_name)
    output_dir.mkdir(exist_ok=True)  # ディレクトリが存在しなければ作成

    cmd = [
        "ffmpeg",
        "-i", str(mp4_file),           # 入力動画ファイル
        "-r", "60",                  # フレームレート (例: 30fps)
        str(output_dir / "frame_%04d.png") # 出力画像ファイル名
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"動画 {mp4_file} を連続画像に変換しました。出力先: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpegの実行中にエラーが発生しました: {e}")
        print(e)