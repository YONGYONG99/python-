#연산자 , 출력 서식

v1 = 2 # 치환
v1 = v2 = v3 = v4 = 5
print(v1, v2, v3 , v4)
v1 = 1, 2, 3
print(v1)
v1 , v2 = 10 ,20
print(v1 , v2)
v2 , v1 = v1 , v2
print(v1 , v2)

print('값 할당 packing')
v1 , *v2 = 1,2,3,4,5
# *v1 , v2 = 1,2,3,4,5
print(v1)
print(v2)

v1 , *v2, v3 =1,2,3,4,5
print(v1 ,v2, v3)

print()
print('------------------------')
print('\n\n연산자(산술, 관계 , 논리)')
print(5+3 , 5-3 , 5*3 , 5/3)
print(5 // 3 , 5%3 , divmod(5,3))

print('연산자 우선순위' , 3+4*5,(3+4)*5)
# 소괄호 () > 산술연산자(** , * , / > + , -) 관리연산자 > 논리연산자 > 치환(=)

print('관계연산자')
print(5>3 , 5==3 , 5 !=3 )
print('논리연산자')
print(5>3 and 4<3 , 5>3 or 4<3 , not(5>=3))

print('문자열 더하기 연산자')
print('파이썬' + '만' + "세")
print('파이썬' * 20)

print('누적')
a=10
a = a+1
a +=1
# a++ 
# ++a 증감 연산자 안된다. a--
print('a:' , a)

print()
print(a, a * -1 , -a , --a)

print('bool : ', True, False, type(True))
print(bool(True))
print(bool(True) , bool(1) , bool(-23.4) , bool('kbs'))
print(bool(False) , bool(0) , bool(0.0) , bool('') , bool(None) , bool([]) , bool(()) , bool({}))

















