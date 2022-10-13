# 채팅 클라이언트

import socket
import threading
import sys

def handle(socket):
    while True:
        data = socket.recv(1024)
        if not data:continue
        print(data.decode('utf-8'))

sys.stdout.flush() # 파이썬의 표준출력은 버퍼링이 되는데 이때 버퍼를 비우기

name = input('채팅 아이디 입력:')
cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 클라이언트 소캣 만들기
cs.connect(('192.168.0.72' , 5000))
cs.send(name.encode('utf-8'))

th = threading.Thread(target=handle, args=(cs, ))
th.start()

while True:
    msg = input()
    sys.stdout.flush()
    if not msg:continue
    cs.send(msg.encode('utf-8'))

cs.close()
