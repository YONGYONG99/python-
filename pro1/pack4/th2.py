import time

now  = time.localtime()
print(now)
print('{}년 {}월 {}일 {}시 {}분 {}초'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

print('-------------------')
import threading

def cal_show():
    now = time.localtime()
    print('{}년 {}월 {}일 {}시 {}분 {}초'.format( \
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

cal_show()