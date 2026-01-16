"""
動画から外部パラメータを計算するためのキャリブレーション画像をフレーム画像を手動抽出するツール（スライダーでフレーム移動）

- matplotlib のスライダーでフレーム番号を動かしてプレビュー
- 「保存」ボタンで現在フレームをPNG保存（何枚でも可）
- 「終了」ボタン（またはウィンドウを閉じる）で終了
- 保存ファイル名にフレーム番号を付与: <動画名>_frame000123.png
"""

from __future__ import annotations

import cv2
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def extract_frames_with_slider(video_path: str | Path, out_dir: str | Path | None = None) -> None:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    # 出力先
    if out_dir is None:
        out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    if total <= 0:
        cap.release()
        raise RuntimeError("フレーム数を取得できませんでした（壊れた動画/コーデック問題の可能性）")

    # 先頭フレームを読み込んで初期表示
    def read_frame(frame_idx: int):
        frame_idx = max(0, min(total - 1, int(frame_idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return None, frame_idx
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb, frame_idx

    frame_rgb, cur_idx = read_frame(0)
    if frame_rgb is None:
        cap.release()
        raise RuntimeError("最初のフレームを読み込めませんでした")

    # ===== GUI（matplotlib） =====
    fig = plt.figure(figsize=(12, 7))
    ax_img = fig.add_axes([0.05, 0.20, 0.90, 0.75])  # 画像表示
    ax_img.set_title(f"{video_path.name}  |  total={total} frames  fps={fps:.2f}")
    ax_img.axis("off")

    im = ax_img.imshow(frame_rgb)

    # スライダー（フレーム番号）
    ax_slider = fig.add_axes([0.08, 0.10, 0.84, 0.05])
    slider = Slider(
        ax=ax_slider,
        label="Frame",
        valmin=0,
        valmax=total - 1,
        valinit=cur_idx,
        valstep=1,
    )

    # ボタン：保存 / 終了
    ax_btn_save = fig.add_axes([0.70, 0.02, 0.12, 0.06])
    btn_save = Button(ax_btn_save, "save")

    ax_btn_quit = fig.add_axes([0.84, 0.02, 0.12, 0.06])
    btn_quit = Button(ax_btn_quit, "quit")

    # 状態表示（保存枚数など）
    saved_indices: set[int] = set()
    status_text = fig.text(0.02, 0.03, "", fontsize=11)

    def update_status(idx: int):
        t = f"Current frame: {idx}  |  Saved: {len(saved_indices)} images  |  Output directory: {out_dir}"
        status_text.set_text(t)

    update_status(cur_idx)

    # スライダーが動いたときにフレーム更新
    def on_slider_change(val):
        nonlocal cur_idx
        idx = int(val)
        frame, idx2 = read_frame(idx)
        if frame is None:
            return
        cur_idx = idx2
        im.set_data(frame)
        update_status(cur_idx)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    # 保存ボタン押下
    def on_save(_event):
        idx = int(slider.val)
        frame, idx2 = read_frame(idx)
        if frame is None:
            return

        # 保存ファイル名（フレーム番号付き）
        out_name = f"{video_path.stem}_frame{idx2:06d}.png"
        out_path = out_dir / out_name

        # 重複保存を避けたい場合は set で管理（上書きしたいならこのifを消す）
        if idx2 in saved_indices and out_path.exists():
            print(f"[SKIP] すでに保存済み: {out_path}")
        else:
            # cv2.imwrite はBGRが必要なので戻す
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out_path), frame_bgr)
            if ok:
                saved_indices.add(idx2)
                print(f"[SAVE] {out_path}")
            else:
                print(f"[ERROR] 保存に失敗: {out_path}")

        update_status(cur_idx)
        fig.canvas.draw_idle()

    btn_save.on_clicked(on_save)

    # 終了ボタン押下
    def on_quit(_event):
        plt.close(fig)

    btn_quit.on_clicked(on_quit)

    # キーボード操作（おまけ）
    # s: 保存 / q or Esc: 終了 / ←→: 1フレーム / Shift+←→: 10フレーム
    def on_key(event):
        step = 10 if event.key in ("shift+left", "shift+right") else 1

        if event.key == "s":
            on_save(None)
        elif event.key in ("q", "escape"):
            on_quit(None)
        elif event.key in ("left", "shift+left"):
            slider.set_val(max(0, int(slider.val) - step))
        elif event.key in ("right", "shift+right"):
            slider.set_val(min(total - 1, int(slider.val) + step))

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    cap.release()

    print(f"\n完了: 保存 {len(saved_indices)} 枚 -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python extract_frames_slider.py <video_path> [out_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else None
    extract_frames_with_slider(video_path, out_dir)
