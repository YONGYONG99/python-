# 멀티 채팅 서버 프로그램 - socket + thread

import socket 
import threading

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind(('127.0.0.1', 5000))
ss.listen(5)
print('채팅 서버 서비스 시작...')

users = []

def chatUser(conn): # 스레드 처리 함수
    name = conn.recv(1024)
    data = '^^ ' + name.decode('utf-8') + '님 입장 ^^'
    print(data)
    
    try:
        for p in users:
            p.send(data.encode('utf-8'))
        
        while True:
            msg = conn.recv(1024)    
            if not msg:continue # 메시지가 없으면 올라감
            data = name.decode('utf-8') + '님 메시지:' + msg.decode('utf-8')
            print('수신 내용 : ', data)
            for p in users:
                p.send(data.encode('utf-8'))           
    except:
        users.remove(conn) # 채팅사용자가 다 빠져나갔을 경우
        data = '~~ ' + name.decode('utf-8') + '님 퇴장 ~~'
        print(data)
        if users:
            for p in users:
                p.send(data.encode('utf-8'))
        else:
            print('사용자가 없어요')
    
while True:
    conn, addr = ss.accept()
    users.append(conn) # 클라이언트를 저장
    th = threading.Thread(target= chatUser , args=(conn,))
    th.start()

# netstat -ano | findstr :3000 -> 프롬프트 명령어 / 3000번 포트를 사용 중인 프로그램있는지 확인하기