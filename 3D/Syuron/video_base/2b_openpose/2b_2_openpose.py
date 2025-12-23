"""
OpenPose処理用スクリプト（出力を direction/<結果フォルダ>/ 以下にまとめる版）
時間の都合でまだ実際に動かしていないので確認必要

出力構成（例）
gopro/<direction>/<結果フォルダ>/
    images/   (jpg)
    json/     (json)
    video/    (avi)

- 介助歩行 thera{i}-0 (i=1..6) は
    1) undistorted_seg        -> openpose_seg                 (最大検出1人)
    2) undistorted_facemasked -> openpose_facemasked_mp1      (最大検出1人)
    3) undistorted_facemasked -> openpose_facemasked_mp2      (最大検出2人)

- それ以外は
    undistorted_facemasked -> openpose_facemasked             (最大検出1人)

- OpenPose実行ごとの処理時間と累計を表示
"""









"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






まだ動かしてないので慎重に！ subの方が確実に動くけど出力がdirction直下になっちゃう!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
import os
import subprocess
from pathlib import Path
import time

# =========================
# OpenPoseの場所
# =========================
# ※OpenPoseDemo.exe があるフォルダに移動
os.chdir(r"C:\Users\Tomson\openpose")
PROGRAM = r".\build\x64\Release\OpenPoseDemo.exe"

# =========================
# データ設定
# =========================
root_dir = Path(r"G:\gait_pattern\BR9G_shuron")
directions = ["fl", "fr"]

# thera{i}-0 のパターン（i=1..6）
THERA_I_0_PATTERNS = [f"thera{i}-0" for i in range(1, 7)]

# 入力ディレクトリ名
DIR_FROM_FACEMASK = "undistorted_facemasked"
DIR_FROM_PTSEG = "undistorted_seg"

# OpenPose設定
FPS = 60
SCALE_NUMBER = 2
SCALE_GAP = 0.2
NET_RESOLUTION = "-1x368"

# 既に出力があれば飛ばす（video があれば skip）
SKIP_IF_OUTPUT_EXISTS = True

# thera0-1 などを除外する
EXCLUDE_THERA0_1 = True


# =========================
# 時間表示用
# =========================
def sec_to_hms(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# ユーティリティ
# =========================
def list_subject_dirs(root: Path):
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub")]


def list_thera_dirs(subject_dir: Path):
    thera = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]

    if EXCLUDE_THERA0_1:
        # thera0-1*, thera0-2_1* を除外
        thera = [d for d in thera if not d.name.startswith("thera0-1")]
        thera = [d for d in thera if not d.name.startswith("thera0-2_1")]
        # thera1-1* ～ thera6-1* を除外
        exclude_patterns = [f"thera{i}-1" for i in range(1, 7)]
        for pattern in exclude_patterns:
            thera = [d for d in thera if not d.name.startswith(pattern)]

    return thera


def is_thera_i_0(thera_name: str) -> bool:
    return any(thera_name.startswith(p) for p in THERA_I_0_PATTERNS)


def has_pngs(img_dir: Path) -> bool:
    return img_dir.exists() and any(img_dir.glob("*.png"))


def build_openpose_cmd(image_dir: Path, out_root: Path, max_people: int):
    """
    出力先を out_root 配下にまとめる
      out_root/
        images/
        json/
        openpose.avi
    """
    out_images = out_root / "images"
    out_json = out_root / "json"
    out_video = out_root / "openpose.avi"

    # 親を作っておく（OpenPoseはディレクトリは作るが、念のため）
    out_images.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'{PROGRAM}'
        f' --image_dir "{image_dir}"'
        f' --write_video_fps {FPS}'
        f' --write_video "{out_video}"'
        f' --write_images "{out_images}"'
        f' --write_images_format jpg'
        f' --write_json "{out_json}"'
        f' --number_people_max {max_people}'
        f' --scale_number {SCALE_NUMBER} --scale_gap {SCALE_GAP}'
        f' --net_resolution {NET_RESOLUTION}'
        f' --display 0'
    )
    return cmd, out_video


def iter_runs_for_thera_direction(thera_dir: Path, direction: str):
    """
    実行すべき (img_dir, out_folder_name, max_people) を列挙

    - 介助歩行 thera{i}-0:
        1) undistorted_seg        -> openpose_seg                 (max_people=1)
        2) undistorted_facemasked -> openpose_facemasked_mp1      (max_people=1)
        3) undistorted_facemasked -> openpose_facemasked_mp2      (max_people=2)

    - それ以外:
        undistorted_facemasked -> openpose_facemasked             (max_people=1)
    """
    gopro_dir = thera_dir / "gopro" / direction
    runs = []

    if is_thera_i_0(thera_dir.name):
        img_seg = gopro_dir / DIR_FROM_PTSEG
        runs.append((img_seg, "openpose_seg", 1))

        img_face = gopro_dir / DIR_FROM_FACEMASK
        runs.append((img_face, "openpose_facemasked_mp1", 1))
        runs.append((img_face, "openpose_facemasked_mp2", 2))
    else:
        img_face = gopro_dir / DIR_FROM_FACEMASK
        runs.append((img_face, "openpose_facemasked", 1))

    return runs


# =========================
# メイン
# =========================
def main():
    subject_dirs = list_subject_dirs(root_dir)
    print(f"対象のsubディレクトリ: {[d.name for d in subject_dirs]}")

    # 実行予定数（SKIPを除いたOpenPose実行数）を先に数える
    total_runs = 0

    for subject_dir in subject_dirs:
        for thera_dir in list_thera_dirs(subject_dir):
            for direction in directions:
                gopro_dir = thera_dir / "gopro" / direction
                for img_dir, out_folder, mp in iter_runs_for_thera_direction(thera_dir, direction):
                    if not has_pngs(img_dir):
                        continue
                    out_root = gopro_dir / out_folder
                    cmd, out_video = build_openpose_cmd(img_dir, out_root, mp)
                    if SKIP_IF_OUTPUT_EXISTS and out_video.exists():
                        continue
                    total_runs += 1

    print(f"OpenPose 実行予定数: {total_runs}")

    run_count = 0
    total_elapsed = 0.0
    t_all_start = time.perf_counter()

    for subject_dir in subject_dirs:
        thera_dirs = list_thera_dirs(subject_dir)
        print(f"\n対象のtheraディレクトリ({subject_dir.name}): {[d.name for d in thera_dirs]}")

        for thera_dir in thera_dirs:
            for direction in directions:
                gopro_dir = thera_dir / "gopro" / direction

                for img_dir, out_folder, mp in iter_runs_for_thera_direction(thera_dir, direction):
                    if not img_dir.exists():
                        print(f"[SKIP] 入力が存在しません: {img_dir}")
                        continue

                    pngs = sorted(img_dir.glob("*.png"))
                    if not pngs:
                        print(f"[SKIP] pngが見つかりません: {img_dir}")
                        continue

                    out_root = gopro_dir / out_folder
                    cmd, out_video = build_openpose_cmd(img_dir, out_root, mp)

                    if SKIP_IF_OUTPUT_EXISTS and out_video.exists():
                        print(f"[SKIP] すでに存在: {out_video}")
                        continue

                    run_count += 1
                    print(f"\n[RUN {run_count}/{total_runs}] {img_dir}")
                    print(f"  thera: {thera_dir.name} / dir: {direction} / input: {img_dir.name} / max_people={mp}")
                    print(f"  out_root: {out_root}")
                    print(f"  out_video: {out_video}")

                    t0 = time.perf_counter()
                    # Windowsで文字列コマンドを確実に動かすため shell=True
                    subprocess.run(cmd, shell=True, check=False)
                    dt = time.perf_counter() - t0

                    total_elapsed += dt
                    avg = total_elapsed / max(run_count, 1)
                    remain = max(total_runs - run_count, 0)
                    est_remain = avg * remain

                    print(
                        f"[DONE {run_count}/{total_runs}] "
                        f"今回: {sec_to_hms(dt)} / "
                        f"累計: {sec_to_hms(total_elapsed)} / "
                        f"平均: {sec_to_hms(avg)} / "
                        f"残り予測: {sec_to_hms(est_remain)}"
                    )

    t_all = time.perf_counter() - t_all_start
    print("\n" + "=" * 70)
    print("すべて完了")
    print(f"OpenPose実行回数: {run_count}/{total_runs}")
    print(f"総経過時間(全体): {sec_to_hms(t_all)}  (OpenPose実行累計: {sec_to_hms(total_elapsed)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
