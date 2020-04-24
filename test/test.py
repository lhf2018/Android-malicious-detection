import time

temp_windows_logcat=open("F:\\pycharmproject\\GraduationProject\\data\\normal_log_data\\logcat_"
                             +str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
                             +".txt"
                   , mode="w",encoding='UTF-8')