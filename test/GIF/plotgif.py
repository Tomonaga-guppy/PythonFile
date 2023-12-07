#ライブラリのインポート
from PIL import Image
import glob
import os

root_dir = r"C:\Users\zutom\.vscode\PythonFile\test\GIF"
start_frame,end_frame=150,360

# def combine_pics(root_dir):
#     pics=[]
#     for i in range (start_frame,end_frame):
#         # 画像を開く
#         pic1 = root_dir+"/RGB/"+str(i).zfill(4)+".png"
#         pic2 = root_dir+"/plot/"+str(i).zfill(4)+".png"
#         img1 = Image.open(pic1)
#         img2 = Image.open(pic2)

#         # 新しい画像のサイズを計算する
#         new_width = img1.width + img2.width
#         new_height = max(img1.height, img2.height)

#         # 新しい画像を作成する
#         new_img = Image.new('RGB', (new_width, new_height))

#         # 新しい画像に2つの画像を貼り付ける
#         new_img.paste(img1, (0, 0))
#         new_img.paste(img2, (img1.width, 0))
#         pics.append(new_img)
#     return pics

# pics = combine_pics(root_dir)
pics = []
for i in range (start_frame,end_frame):
    pic = root_dir+"/RGB/"+str(i).zfill(4)+".png"
    # pic = root_dir+"/plot/"+str(i).zfill(4)+".png"
    img = Image.open(pic)
    pics.append(img)

def create_gif(root_dir,pics):
    #画像を入れる箱を準備
    pictures=[]

    #画像を箱に入れていく
    for pic in(pics):
        # pic.Image.open(pic)
        pictures.append(pic)

    #gifアニメを出力する
    pictures[0].save(root_dir+'/RGB.gif',save_all=True, append_images=pictures[1:],optimize=True, duration=1000/30, loop=0)
    # pictures[0].save(root_dir+'/plot.gif',save_all=True, append_images=pictures[1:],optimize=True, duration=1000/30, loop=0)

create_gif(root_dir,pics)
print("gif is saved!")