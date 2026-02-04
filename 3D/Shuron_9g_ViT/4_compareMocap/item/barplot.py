import matplotlib.pyplot as plt

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
plt.figure(figsize=(7, 5))

plt.bar(labels, mae_mean)

# 5 deg：臨床目安
plt.axhline(
    5.0,
    color="red",
    linestyle="-",
    linewidth=2,
    label="Clinical threshold (5°)"
)

# 12.5 deg：MoCap誤差目安
plt.axhline(
    12.5,
    color="black",
    linestyle="-.",
    linewidth=2,
    label="MoCap error tolerance (12.5°)"
)

# 軸・タイトル
plt.ylabel("MAE [deg]")
plt.title("Mean MAE of Joint Angles")

# y軸レンジ
plt.ylim(0, max(mae_mean + [12.5]) * 1.25)

# 凡例
plt.legend()

plt.tight_layout()
save_path = r"C:\Users\Tomson\StrokeProject\3D\Shuron_9g_ViT\4_compareMocap\item\joint_angle_mae_barplot.png"
plt.savefig(save_path, dpi=300)
plt.close()