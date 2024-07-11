import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe' #ffmpegを


df = pd.read_excel('C:\\Users\\zutom\\PythonDataFile\\data_task4.xlsx', usecols=[0,1], header=None, names=["Time", "Displacement"]).values
#エクセルデータから1、2列目を指定し、一番上の行にラベルを設定してから.valueで配列として読み取り
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, 60)
ax.set_ylim(-30, 40)
plt.grid()
plt.xlabel("time [sec]")
plt.ylabel("displacement [mm]")
line, = ax.plot([], [], c='b', markersize=5)        #line, で1つの線を表すLine2Dオブジェクトを返す。
                                                    #Line2Dオブジェクトを直接参照するため、line.set.data(x,y)のように線のデータの更新が可能

interval_time=1000               #図の切り替わる時間[ms]を1000に設定                           

#frameは0からframes-1までで処理される。今回だと0-59999の6万回excelplotを処理
def excelplot(frame):
    i = int((frame)*interval_time)
    x = df[:i, 0]       # 0列目の0からi行目までを取り出す
    y = df[:i, 1]       # 1列目の0からi行目までを取り出す
    line.set_data(x, y)     # 線のデータを更新する
    return [line]

ani = animation.FuncAnimation(fig, excelplot, frames=int(len(df)/interval_time), interval=interval_time, blit=True)
#blitはblittingという手法による描画の高速化らしい
# ani.save('anim.gif', writer='Pillow')       #gifで保存する場合
ani.save('anim.mp4', writer='ffmpeg')             #mp4で保存する場合
