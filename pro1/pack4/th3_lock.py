# 여러 스레드 간 공유자원 충돌 방지
# 동기화(줄서기) - 하나의 스레드가 자원을 사용하는 동안 다른 스레드는 공유자원사용을 대기

import threading , time

g_count =  0 # 전역변수는 자동으로 스레드의 공유자원이 됨
lock = threading.Lock()

def threadCount(id , count):
    global g_count # 전역
    for i in range(count): # count -> 지역
        lock.acquire() # 스레드 간 충돌 방지용. 현재 스레드가 공유자원을 점유하고 있는 동안 다른 스레드를 대기상태
        print('id %s ===> count:%s, g_count:%s'%(id, i, g_count))
        time.sleep(0.1)
        g_count += 1
        lock.release() # 공유자원을 점유 해제

for i in range(1, 6):
    threading.Thread(target=threadCount , args=(i, 5)).start()
    
time.sleep(5)
print('처리 후 최종 g_count : ', g_count)
print('프로그램 종료')

