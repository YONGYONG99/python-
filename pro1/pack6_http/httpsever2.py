# 웹 서버 구축

from http.server import CGIHTTPRequestHandler, HTTPServer
# CGIHTTPRequestHandler : 동적으로 웹서버를 운영 가능
# CGI : 웹서버와 웹프로그램 사이에서 정보를 주고받는 방법 또는 규약

port = 8888

class Handler(CGIHTTPRequestHandler):
    cg_directories = ['/cgi-bin']
    
serv = HTTPServer(('127.0.0.1' , port), Handler)
print('웹 서비스 시작 ...')
serv.serve_forever() 