import matplotlib.pyplot as plt
# =========================
# フォント・見た目（全体設定）
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,          # 全体
    "axes.titlesize": 20,    # タイトル
    "axes.labelsize": 20,     # 軸ラベル
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 19,
    "legend.title_fontsize": 19,
})

# =========================
# データ（meanのみ）
# =========================
labels = [
    "Hip max ext",
    "Knee max flex",
    "Ankle max do",
    "Hip max ab"
]

mae_mean = [7.93, 3.07, 7.20, 3.19]

# =========================
# プロット
# =========================
plt.figure(figsize=(10, 5))

plt.bar(labels, mae_mean)


# 12.5 deg：MoCap誤差目安
plt.axhline(
    12.5,
    color="red",
    linestyle="-",
    linewidth=2,
    label="Mocap Error Upper Bound (12.5°)"
)

# 5 deg：臨床目安
plt.axhline(
    5.0,
    color="red",
    linestyle="-.",
    linewidth=2,
    label="Clinically Acceptable Error (5°)"
)


# 軸・タイトル
plt.ylabel("MAE [deg]")
plt.title("Mean MAE of Joint Angles")

# y軸レンジ
# plt.ylim(0, max(mae_mean + [12.5]) * 1.25)
plt.ylim(0, 18)

# 凡例
plt.legend()

plt.tight_layout()
save_path = r"C:\Users\Tomson\StrokeProject\3D\Shuron_9g_ViT\4_compareMocap\item\joint_angle_mae_barplot.png"
plt.savefig(save_path, dpi=300)
plt.close()