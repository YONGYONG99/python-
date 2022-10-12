# client / server(echo) 프로그래밍
# server

from socket import *

# socket으로 서버 구성
serversock = socket(AF_INET , SOCK_STREAM) # socket(소켓종류, 소켓유형)
serversock.bind(('127.0.0.1' , 8888))
serversock.listen(1) # 동시 접속 최대수 설정 (1 ~ 5)
print('server start...')

conn, addr = serversock.accept() # 연결 대기
print('addr : ', addr)
print('conn : ', conn)
print('from client message : ', conn.recv(1024).decode())
conn.close() # 단발성
serversock.close()