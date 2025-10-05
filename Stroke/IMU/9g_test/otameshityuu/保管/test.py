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

        # Serial port 險ｭ螳�
        ser = serial.Serial()
        ser.port = "COM4"  # 繝昴�繝�
        ser.timeout=1.0                                # 繧ｿ繧､繝�繧｢繧ｦ繝�
        ser.baudrate = 115200                          # 繝懊�繝ｬ繝ｼ繝�

        # Serial port Open
        ser.open() 
        
        # 蜉�騾溷ｺｦ繝ｻ隗帝溷ｺｦ繝代Λ繝｡繝ｼ繧ｿ險ｭ螳�
        header = 0x9A
        cmd = 0x16
        data = 0x01
        data1 = 0x0A
        data2 = 0x00
        
        check = header ^ cmd
        check = check ^ data
        check = check ^ data1
        check = check ^ data2

        print(ser)
	
        list = bytearray([header,  cmd,  data,  data1, data2 , check])

        #繝舌ャ繝輔ぃ繧ｯ繝ｪ繧｢
        ser.read(1000)
        ser.write(list)

        str = ser.readline()

        print('CmdRes:' + repr(str))
	
        # 險域ｸｬ髢句ｧ�
        header = 0x9A
        cmd    = 0x13
        smode   = 0x00
        syear  = 0x00
        smonth = 0x01
        sday   = 0x01
        shour  = 0x00
        smin   = 0x00
        ssec   = 0x00
        emode  = 0x00
        eyear  = 0x00
        emonth = 0x01
        eday   = 0x01
        ehour  = 0x00
        emin   = 0x00
        esec   = 0x00
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
	
        list = bytearray([header,  cmd, smode, syear, smonth, sday, shour, smin, ssec, emode, eyear, emonth, eday, ehour, emin, esec, check])
  
        # 繝舌ャ繝輔ぃ繧ｯ繝ｪ繧｢
        ser.read(100)
        ser.write(list)
	
        str = ser.readline()
	
        # 險域ｸｬ髢句ｧ矩夂衍
        str =ser.read(1)
	
        # 繝倥ャ繝讀懃ｴ｢
        while ord(str) != 0x9A:
                str = ser.read(1)

        # 繧ｳ繝槭Φ繝牙叙蠕�
        str = ser.read(1)
	
        # 蜉�騾溷ｺｦ隗帝溷ｺｦ險域ｸｬ繝��繧ｿ騾夂衍縺ｮ縺ｿ蜃ｦ逅�☆繧�
        if ord(str) == 0x80:
                
                # 繧ｿ繧､繝�繧ｹ繧ｿ繝ｳ繝�
                str = ser.read(4)
                
                # 蜉�騾溷ｺｦX
                data1 = ser.read(1)
                data2 = ser.read(1)
                data3 = ser.read(1)

                # 3byte縺ｮ蛟､繧�4byte縺ｮint蝙九→縺励※繝槭う繝翫せ縺ｮ繝上Φ繝峨Μ繝ｳ繧ｰ
                if ord(data3) & 0x80:
                        data4 = b'\xFF'
                else:
                        data4 = b'\x00'
		
                print(binascii.b2a_hex(data1))
                print(binascii.b2a_hex(data2))
                print(binascii.b2a_hex(data3))
                print(binascii.b2a_hex(data4))
		
                # 繧ｨ繝ｳ繝�ぅ繧｢繝ｳ螟画鋤
                accx = ord(data1)
                accx += ord(data2)<<8
                accx += ord(data3)<<16
                accx += ord(data4)<<24

                print("accx = %d" % (ctypes.c_int(accx).value))
	
        ser.close();
