import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# root_dir = r"F:\Tomson\gait_pattern\20240712"
root_dir = r"F:\Tomson\gait_pattern\20240807test"

condition_num = "test"
# condition_num = "0"
npz_file_path = glob.glob(os.path.join(root_dir, f"{condition_num}*frame.npz"))[0]

def animate(keypoint_sets):
    diagonal_right = keypoint_sets['diagonal_right']
    diagonal_left = keypoint_sets['diagonal_left']
    frontal = keypoint_sets['frontal']
    # mocap = keypoint_sets['mocap']
    frame_range = keypoint_sets['common_frame']
    sagittal_3d = keypoint_sets['sagittal_3d']

    print(f"diagonal_right.shape = {diagonal_right.shape}")
    print(f"diagonal_left.shape = {diagonal_left.shape}")
    print(f"frontal.shape = {frontal.shape}")
    # print(f"mocap.shape = {mocap.shape}")
    print(f"frame_range = {frame_range}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        print(f"frame = {frame}")
        ax.clear()
        # mocap_x = mocap[frame, :, 0]
        # mocap_y = mocap[frame, :, 1]
        # mocap_z = mocap[frame, :, 2]

        frontal_x = frontal[frame, :, 0]
        frontal_y = frontal[frame, :, 1]
        frontal_z = frontal[frame, :, 2]

        diagonal_right_x = diagonal_right[frame, :, 0]
        diagonal_right_y = diagonal_right[frame, :, 1]
        diagonal_right_z = diagonal_right[frame, :, 2]

        diagonal_left_x = diagonal_left[frame, :, 0]
        diagonal_left_y = diagonal_left[frame, :, 1]
        diagonal_left_z = diagonal_left[frame, :, 2]

        saggital_x = sagittal_3d[frame, :, 0]
        saggital_y = sagittal_3d[frame, :, 1]
        saggital_z = sagittal_3d[frame, :, 2]

        # ax.scatter(mocap_x, mocap_y, mocap_z, label='mocap')
        ax.scatter(frontal_x, frontal_y, frontal_z, label='frontal')
        ax.scatter(diagonal_right_x, diagonal_right_y, diagonal_right_z, label='diagonal_right')
        ax.scatter(diagonal_left_x, diagonal_left_y, diagonal_left_z, label='diagonal_left')
        ax.scatter(saggital_x, saggital_y, saggital_z, label='saggital')
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f"Frame {frame}")
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 4)
        ax.set_zlim(-4, 0)

    ani = animation.FuncAnimation(fig, update, frames=frame_range, repeat=False)
    anime_save_path = os.path.join(root_dir, f"{condition_num}_animation.mp4")
    writer = FFMpegWriter(fps=30)
    ani.save(anime_save_path, writer=writer)
    plt.show()

    for frame in frame_range:
        # plt.cla()
        # ax.clear()
        print(f"frame = {frame}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # mocap_x = mocap[frame, :, 0]
        # mocap_y = mocap[frame, :, 1]
        # mocap_z = mocap[frame, :, 2]

        frontal_x = frontal[frame, :, 0]
        frontal_y = frontal[frame, :, 1]
        frontal_z = frontal[frame, :, 2]

        diagonal_right_x = diagonal_right[frame, :, 0]
        diagonal_right_y = diagonal_right[frame, :, 1]
        diagonal_right_z = diagonal_right[frame, :, 2]

        diagonal_left_x = diagonal_left[frame, :, 0]
        diagonal_left_y = diagonal_left[frame, :, 1]
        diagonal_left_z = diagonal_left[frame, :, 2]

        saggital_x = sagittal_3d[frame, :, 0]
        saggital_y = sagittal_3d[frame, :, 1]
        saggital_z = sagittal_3d[frame, :, 2]

        # ax.scatter(mocap_x, mocap_y, mocap_z, label='mocap')
        ax.scatter(frontal_x, frontal_y, frontal_z, label='frontal')
        ax.scatter(diagonal_right_x, diagonal_right_y, diagonal_right_z, label='diagonal_right')
        ax.scatter(diagonal_left_x, diagonal_left_y, diagonal_left_z, label='diagonal_left')
        ax.scatter(saggital_x, saggital_y, saggital_z, label='saggital')
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f"Frame {frame}")
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 4)
        ax.set_zlim(-1, 3)
        plt.show()

def main():
    keypoint_sets = np.load(npz_file_path)

    animate(keypoint_sets)

if __name__ == "__main__":
    main()


