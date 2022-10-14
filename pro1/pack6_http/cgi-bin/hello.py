# 웹 서비스 대상 파일
kor = 50
eng = 60
tot = kor + eng

print('Content-Type:text/html;charset=utf-8\n') # -> MIME type / 브라우저한테 문서의 타입을 알려주는것 준비하라고
print('<html><body>')
print('<b>안녕하세요</b> 파이썬 모듈로 작성했어요<br>') 
print('총점은 %s'%(tot))
print('</body></html>')  # 약간 servlet 개념이라고 보면됨
