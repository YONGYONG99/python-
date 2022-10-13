# 스레드간 자원공유 + 스레드 활성화/비활성화

import threading , time 

bread_plate = 0 # 빵접시 - 공유자원
lock = threading.Condition() # lock을 위한 조건변수

# 생산자 
class Maker(threading.Thread): # 스레드 상속받기
    def run(self):
        global bread_plate
        for i in range(30):
            lock.acquire() # 공유자원 점유 / 공유 자원 충돌 목적으로 lock 걸기
            while bread_plate >= 10:
                print('빵 생산 초과로 대기')
                lock.wait() # 스레드 비활성화
            bread_plate += 1
            print('빵생산 : ', bread_plate)
            lock.notify()   # 스레드 활성화
            lock.release() # 공유자원 점유 해제 
            time.sleep(0.05)
# acquire , release 가 한쌍 / wait , notify 가 한쌍이다.

# 소비자
class Consumer(threading.Thread): # 스레드 상속받기
    def run(self):
        global bread_plate
        for i in range(30):
            lock.acquire()
            while bread_plate < 1:
                print('빵 소비 초과로 대기')
                lock.wait() 
            bread_plate -= 1
            print('빵소비 : ', bread_plate)
            lock.notify()   # 스레드 활성화
            lock.release() # 공유자원 점유 해제 
            time.sleep(0.06)

mak = []; con = []

for i in range(5): # 생산자수 , 빵만드는 사람이 5명
    mak.append(Maker())

for i in range(5): # 소비자수도 5명
    con.append(Consumer())

for th1 in mak: # 스레드 시작
    th1.start()

for th2 in con:
    th2.start()

for th1 in mak:
    th1.join()

for th2 in con:
    th2.join()

print('오늘 영업 끝~~~')
    