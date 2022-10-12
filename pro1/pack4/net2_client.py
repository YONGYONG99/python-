# 단순 클라이언트
from socket import *

clientsock =socket(AF_INET, SOCK_STREAM)
clientsock.connect(('127.0.0.1' , 7878)) # 능동적으로 server에 접속
clientsock.send('삐리빠리뽀옹'.encode(encoding='utf_8')) # 송신
re_msg = clientsock.recv(1024).decode() # 수신
print('수신자료 : ', re_msg)
clientsock.close()
