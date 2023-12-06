#ライブラリのインポート
from PIL import Image
import glob
import os

#画像を入れる箱を準備
pictures=[]
root_dir = r"C:\Users\zutom\.vscode\PythonFile\test\GIF"
pics = glob.glob(os.path.join(root_dir,"*.png"))

#画像を箱に入れていく
for pic in(pics):
    img = Image.open(pic)
    pictures.append(img)

#gifアニメを出力する
pictures[0].save('anime.gif',save_all=True, append_images=pictures[1:],
optimize=True, duration=1000/30, loop=0)
