# 웹 서버 구축


from http.server import SimpleHTTPRequestHandler , HTTPServer
# HTTPServer : 기본적인 socket 연결을 관리
# SimpleHTTPRequestHandler : 요청을 처리 (get , post)

port = 7777

handler = SimpleHTTPRequestHandler
serv = HTTPServer(('127.0.0.1' , port), handler)
print('웹 서비스 시작 ...')
serv.serve_forever() # 웹주소창에 127.0.0.1:7777 입력해보기


