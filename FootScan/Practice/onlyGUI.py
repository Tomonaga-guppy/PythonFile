# いろんなモジュールを import
# 逆にここにあるモジュールを cmd で pip install すればPC変えてもこのプログラムをつかって装置動かせる。あと保存先のファイルとEPSONアプリ必要だけど、、
import datetime
import tkinter as tk #GUIアプリケーションで使う
import tkinter.ttk as ttk
from tkinter import filedialog as fd  # 参照用
import cv2  # 画像解析
import openpyxl as px  # excel 書き込み用、pip install openpyxl 必要
import win32api
import numpy as np # pip install numpy 必
from matplotlib import pyplot as plt # pip install matplotlib 必要
import serial,time # serialは、PythonとArduinoなどの機器との間でシリアル通信を実現させられるモジュール、pip install pyserial 必要
# シリアル通信は接続されたコンピュータ同士で、1ビットずつデータを送る通信の方式,パラレル通信はデータを複数のビットにまとめて送る通信の方式
from multiprocessing import Process # マルチプロセスによる並列処理をサポートする
import threading # マルチスレッドプログラミングを行うためのモジュール
import csv # CSVファイル(Comma Separated Valueの略で、カンマ（,）で区切られた値が含まれているテキストファイル)の読み書きをするためのモジュール
import glob # 引数に指定されたパターンにマッチするファイルパス名を取得、処理するときに使うモジュール
import os # Pythonのコード上でOS（オペレーティング・システム）に関する操作を実現するためのモジュール,pip install pywin32 必要？
from scipy import signal
import pandas as pd
from PIL import Image, ImageTk ,ImageWin # Pillow の pip install が必要
import win32print # 印刷に必要、pip install pywin32 で入る。~~~~~~~~ のままで大丈夫。

#解析用自作ファイル
import kaiseki_press
import kaiseki_area
import kaiseki_jushin_angl
import kaiseki_edge
import kaiseki_HandW


wb = px.load_workbook('/Users/BRLAB/hikensya.xlsx') # 被験者情報の保存先のエクセルファイル
ws = wb.active # ファイルの書き換え可能にするactive

def start_jouken():

    frame_start = tk.Frame(baseGround, width=500, height=700)
    frame_start.pack()
    def click_btnStart1(): # 下のスタートボタン1を押した後のイベント
        global jnum
        jnum=1
        frame_start.destroy()
        id()
    def click_btnStart2(): # 下のスタートボタン2を押した後のイベント
        global jnum
        jnum=2
        frame_start.destroy()
        id()
    def click_btnStart3(): # 下のスタートボタン3を押した後のイベント
        global jnum
        jnum=3
        frame_start.destroy()
        id()
    def click_btnStart4(): # 下のスタートボタン4を押した後のイベント
        global jnum
        jnum=4
        frame_start.destroy()
        id()
    def file_dialog(): # 下のスタートボタン5を押した後のイベント
        frame_start.destroy()
        kaiseki_gui()

    # ラベルやボタンの作成と配置
    label2 = tk.Label(frame_start, text="まだ台に乗らないでください",font=("",30))
    label2.place(x=20, y=10)
    label2['fg']="red"

    buttonStart1 = tk.Button(frame_start,text = '脱いですぐSTART', command=click_btnStart1,font=("",20))
    buttonStart1.place(x=50, y=145, relwidth=0.75, relheight= 0.125)

    buttonStart2 = tk.Button(frame_start,text = '脱いでしばらくしてSTART', command=click_btnStart2,font=("",20))
    buttonStart2.place(x=50, y=245, relwidth=0.75, relheight= 0.125)

    buttonStart3 = tk.Button(frame_start,text = '+20kgでSTART', command=click_btnStart3,font=("",20))
    buttonStart3.place(x=50, y=345, relwidth=0.75, relheight= 0.125)

    buttonStart4 = tk.Button(frame_start,text = 'ウエットティッシュでSTART', command=click_btnStart4,font=("",20))
    buttonStart4.place(x=50, y=445, relwidth=0.75, relheight= 0.125)

    buttonStart5 = tk.Button(frame_start,text = '画像処理/解析', command=file_dialog,font=("",20))
    buttonStart5.place(x=50, y=545, relwidth=0.75, relheight= 0.125)
