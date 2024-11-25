# -*- coding: utf-8 -*-
#
# TSND/AMWS sample program for Python3
#
# required:
#     PySerial
#
import os
import sys
import serial
import time
import struct
import binascii
import ctypes

if __name__ == '__main__':

        # Serial port 設定
        ser = serial.Serial()
        ser.port = "COM6"  # 使用するシリアルポート（デバイスに応じて変更）
        ser.timeout = 1.0  # タイムアウト設定（1秒）
        ser.baudrate = 115200  # 通信速度（ボーレート）

        # Serial port Open
        ser.open()  # シリアルポートを開く

        # 計測開始コマンドを送信するための設定
        header = 0x9A  # ヘッダー（コマンドの先頭）
        cmd = 0x16  # 加速度計測開始のコマンド
        data = 0x01  # 計測周期（1ms）
        data1 = 0x0A  # 送信データの平均回数
        data2 = 0x00  # 記録設定なし

        # チェックサムの計算（各バイトのXORを取る）
        check = header ^ cmd
        check = check ^ data
        check = check ^ data1
        check = check ^ data2

        # シリアルオブジェクトの状態を表示
        print(ser)

        # コマンドリストを作成
        list = bytearray([header, cmd, data, data1, data2, check])

        # 受信バッファをクリアして、コマンドを送信
        ser.read(1000)
        ser.write(list)

        # コマンドレスポンスを読み取る
        str = ser.readline()
        print('CmdRes:' + repr(str))

        # 計測の開始・終了時間を設定するためのコマンド
        header = 0x9A
        cmd = 0x13  # 計測開始/予約コマンド
        smode = 0x00  # 相対時間指定
        syear = 0x00  # 開始年（2000年からの経過年数）
        smonth = 0x01  # 開始月
        sday = 0x01  # 開始日
        shour = 0x00  # 開始時
        smin = 0x00  # 開始分
        ssec = 0x00  # 開始秒
        emode = 0x00  # 相対時間指定
        eyear = 0x00  # 終了年（2000年からの経過年数）
        emonth = 0x01  # 終了月
        eday = 0x01  # 終了日
        ehour = 0x00  # 終了時
        emin = 0x00  # 終了分
        esec = 0x00  # 終了秒

        # チェックサムの計算
        check = header ^ cmd
        check = check ^ smode
        check = check ^ syear
        check = check ^ smonth
        check = check ^ sday
        check = check ^ shour
        check = check ^ smin
        check = check ^ ssec
        check = check ^ emode
        check = check ^ eyear
        check = check ^ emonth
        check = check ^ eday
        check = check ^ ehour
        check = check ^ emin
        check = check ^ esec

        # コマンドリスト作成
        list = bytearray([header, cmd, smode, syear, smonth, sday, shour, smin, ssec, emode, eyear, emonth, eday, ehour, emin, esec, check])

        # コマンド送信
        ser.read(100)
        ser.write(list)

        # レスポンスを読み取る
        str = ser.readline()

        # 1バイトずつデータを読み取る
        str = ser.read(1)

        # レスポンスが0x9A（データの開始）まで待つ
        while ord(str) != 0x9A:
                str = ser.read(1)

        # コマンドの種類を読み取る
        str = ser.read(1)

        # 取得したコマンドが加速度データかどうか確認（コード 0x80）
        if ord(str) == 0x80:

                # 残りの4バイトを読み取る（ヘッダなど）
                str = ser.read(4)

                # X軸の加速度データを読み取る（3バイト）
                data1 = ser.read(1)
                data2 = ser.read(1)
                data3 = ser.read(1)

                # 3バイトを4バイトの符号付き整数に変換
                if ord(data3) & 0x80:  # もし負の値ならば符号を補正
                        data4 = b'\xFF'
                else:
                        data4 = b'\x00'

                # 受信したデータを16進数で表示
                print(binascii.b2a_hex(data1))
                print(binascii.b2a_hex(data2))
                print(binascii.b2a_hex(data3))
                print(binascii.b2a_hex(data4))

                # X軸の加速度値を符号付き32ビット整数に変換
                accx = ord(data1)
                accx += ord(data2) << 8
                accx += ord(data3) << 16
                accx += ord(data4) << 24

                # 加速度データ（X軸）を表示
                print("accx = %d" % (ctypes.c_int(accx).value))

        # シリアルポートを閉じる
        ser.close()
