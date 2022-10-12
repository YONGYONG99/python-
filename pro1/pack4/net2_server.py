# 서버 무한 루핑
import socket
import sys
HOST = '127.0.0.1'
PORT = 7878

serversock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    serversock.bind((HOST , PORT))
    serversock.listen(5) # 동시 접속 최대수 설정 (1 ~ 5)
    print('server start...')
    
    while True:
        conn, addr = serversock.accept() # 연결 대기
        print('client info : ', addr[0], addr[1]) # ip address , port number
        print('from client message:' , conn.recv(1024).decode()) # 수신
        
        # 메시지 송신
        conn.send(('from server : ' +str(addr[0]) + ' 너도 잘 지내라').encode('utf_8'))
        
        
        
except socket.error as err:
    print('err : ', err)
    sys.exit()
finally:
    serversock.close()
    conn.close()



