s1 = '자료1 '
s2 = '두번째 자료'

print('Content-Type:text/html;charset=utf-8\n')
# 문자열로 html주기
print('''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
<h1>반가워요</h1>
자료 출력 : {0}, {1}
<br>
<img src='../images/logo.png' width='60%' />
<br/>
<a href='../index.html'> 메인으로 </a>
</body></html>
'''.format(s1,s2)) # 매핑하기