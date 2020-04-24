# 从windows每一分钟取出一次正常日志，并清空log文件
import time

windows_logcat = open("C:\\Users\\11469\\AppData\\Local\\Genymobile\\Genymotion\\deployed\\Samsung Galaxy S6\\logcat-m38620.txt"
                   , mode="r+",encoding='UTF-8')
for i in range(1,10):
    windows_logcat.seek(0)
    windows_logcat.truncate()
    time.sleep(5)
    temp_windows_logcat=open("F:\\pycharmproject\\GraduationProject\\data\\normal_log_data\\logcat_"
                             +str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
                             +".txt"
                   , mode="w",encoding='UTF-8')
    temp_windows_logcat.writelines(windows_logcat.readlines())

