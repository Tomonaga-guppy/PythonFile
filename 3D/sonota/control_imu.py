# -*- coding: utf-8 -*-
#
# TSND/AMWS sample program for Python3
#
# required:
#     PySerial
#
import os
import sys
import serial  # シリアル通信を扱うためのライブラリ
import time
import struct
import binascii  # バイナリデータを16進数で扱うためのライブラリ
import ctypes  # C言語スタイルのデータ型を扱うためのライブラリ

if __name__ == '__main__':

    # シリアルポートの設定
    ser = serial.Serial()
    ser.port = "/dev/tty.TSND151-AP09181080-Blue"  # TSNDデバイスのシリアルポート名
    ser.port = "COM6"  # TSNDデバイスのシリアルポート名
    ser.timeout = 1.0  # シリアル通信のタイムアウトを1秒に設定
    ser.baudrate = 115200  # 通信速度（ボーレート）を115200bpsに設定

    # シリアルポートを開く
    ser.open()

    # コマンドの送信準備
    header = 0x9A  # コマンドのヘッダー
    cmd = 0x16  # 加速/各速度データ計測設定
    data = 0x01  # コマンドのデータ
    data1 = 0x0A  # 追加のデータ
    data2 = 0x00  # 追加のデータ

    # チェックサムの計算（XORを使ってヘッダーやデータを全てチェック）
    check = header ^ cmd
    check ^= data
    check ^= data1
    check ^= data2

    # コマンドを表示
    print(ser)

    # 送信するコマンドをバイト列に変換
    list = bytearray([header, cmd, data, data1, data2, check])

    # シリアルポートにデータを書き込む（コマンド送信）
    ser.read(1000)  # 応答を待つために一度読み取る
    ser.write(list)  # コマンドを送信

    # デバイスからの応答を1行読み取る
    str = ser.readline()

    # 応答を表示
    print('CmdRes:' + repr(str))

    # 別のコマンドを設定（日時設定などのコマンド）
    header = 0x9A
    cmd = 0x13  # 計測開始
    smode = 0x00  # 開始モード
    syear = 0x00  # 開始年
    smonth = 0x01  # 開始月
    sday = 0x01  # 開始日
    shour = 0x00  # 開始時
    smin = 0x00  # 開始分
    ssec = 0x00  # 開始秒
    emode = 0x00  # 終了モード
    eyear = 0x00  # 終了年
    emonth = 0x01  # 終了月
    eday = 0x01  # 終了日
    ehour = 0x00  # 終了時
    emin = 0x00  # 終了分
    esec = 0x00  # 終了秒

    # チェックサムを再度計算
    check = header ^ cmd
    check ^= smode ^ syear ^ smonth ^ sday ^ shour ^ smin ^ ssec
    check ^= emode ^ eyear ^ emonth ^ eday ^ ehour ^ emin ^ esec

    # 新しいコマンドをバイト列に変換
    list = bytearray([header, cmd, smode, syear, smonth, sday, shour, smin, ssec, emode, eyear, emonth, eday, ehour, emin, esec, check])

    # 再度コマンドを送信
    ser.read(100)  # 一度読み取る（空読み）
    ser.write(list)  # コマンドを送信

    # 応答を再度読み取る
    str = ser.readline()

    # 受信したデータを1バイト読み取る
    str = ser.read(1)

    # ヘッダー（0x9A）を見つけるまで読み取る
    while ord(str) != 0x9A:
        str = ser.read(1)

    # データを1バイト読み取る
    str = ser.read(1)

    # 応答が加速度データの場合（0x80は加速度各速度計測データ通知）
    if ord(str) == 0x80:

        # 応答のデータ（加速度センサーデータを含む）
        str = ser.read(4)  # 4バイト読み取る

        # 3バイトの加速度データを読み取る
        data1 = ser.read(1)
        data2 = ser.read(1)
        data3 = ser.read(1)

        # 3バイトのデータを4バイトの整数に変換（符号拡張）
        if ord(data3) & 0x80:  # data3の最上位ビットが1なら負の値
            data4 = b'\xFF'  # 符号拡張（負の値用にFFを追加）
        else:
            data4 = b'\x00'  # 符号拡張（正の値用に00を追加）

        # データを16進数で表示
        print(binascii.b2a_hex(data1))
        print(binascii.b2a_hex(data2))
        print(binascii.b2a_hex(data3))
        print(binascii.b2a_hex(data4))

        # 4バイトのデータを32ビット整数として結合
        accx = ord(data1)
        accx += ord(data2) << 8
        accx += ord(data3) << 16
        accx += ord(data4) << 24

        # 符号付き32ビット整数として表示
        print("accx = %d" % (ctypes.c_int(accx).value))

    # シリアルポートを閉じる
    ser.close()
