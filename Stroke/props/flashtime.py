import tkinter as tk
import time

class FrameCounterApp:
    def __init__(self, root):
        self.root = root
        self.frame_count = 0
        self.last_time = time.time()

        # フレームカウントを表示するラベルの設定
        self.label = tk.Label(root, font=('Helvetica', 50), fg='blue')
        self.label.pack(pady=20)

        # フレーム更新メソッドの呼び出し
        self.update_frame()

    def update_frame(self):
        current_time = time.time()
        if (current_time - self.last_time) >= 0.001:  # 0.001秒 = 1ミリ秒
            self.frame_count += 1
            self.label.config(text=f"Frame: {self.frame_count}")
            self.last_time = current_time

        # 1ミリ秒後に再びこのメソッドを呼び出す
        self.root.after(1, self.update_frame)

def main():
    root = tk.Tk()
    root.title("Frame Counter")
    app = FrameCounterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
