# 배열 연산

import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
print(x, x.dtype) # 소수점 들어간거 확인
y = np.arange(5,9).reshape(2,2) # 2행 2열짜리로 바꾸기
y = y.astype(np.float32) # 형변환
print(y , y.dtype)

print()
print(x+y)
print(np.add(x,y)) # numpy 더하기 함수
imsi = np.random.rand(1000000)
print(sum(imsi)) # 파이썬의 함수
print(np.sum(imsi)) # numpy의 함수 , 결과는 같지만 파이썬 연산함수보다 속도가 빠름

print()
print(x -y)
print(np.subtract(x,y))

print()
print(x * y)
print(np.multiply(x,y))

print()
print(x / y)
print(np.divide(x,y))

print()
# 벡터간 내적 연산 ~ (행렬곱) : dot함수 , 참고로 R에서는 a %*% b
v = np.array([9, 10])
w = np.array([11,12])
print(v * w)
print(v.dot(w)) # v[0]*w[0] + v[1]*w[1]
print(np.dot(v, w)) # 더 빠름

print()
print(x) # 2차원
print(v) # 1차원
print(np.dot(x,v)) 

print()
print(x) # 2차원
print(y) # 1차원
print(np.dot(x,y)) 

print('------------')
print(np.sum(x))
print(np.mean(x))
print(np.cumsum(x)) # 누적합
print(np.cumprod(x)) # 누적곱
# ...

print()
name1 = np.array(['tom','james','tom','oscar'])
name2 = np.array(['tom','page','john'])
print(np.unique(name1)) # 중복 배제
print(np.intersect1d(name1,name2, assume_unique = True)) # 교집합 , 뒤표현은 중복 허용
print(np.union1d(name1,name2)) # 합집합

print('\nTranspose : 전치')
print(x)
print(x.T)
print(x.transpose())
print(x.swapaxes(0,1))

print('\nBroadcast 연산 : 크기가 다른 배열 간의 연산을 하면 작은 배열이 큰 배열의 크기를 자동으로 따라감')
x = np.arange(1, 10).reshape(3,3)
print(x)
y = np.array([1,0,1])
print(y)
print(x+y)

print('\n띠용')
print(x)
np.savetxt('my.txt',x)
imsi = np.loadtxt('my.txt')
print(imsi)
print()
imsi2 = np.loadtxt('my2.txt', delimiter=',')
print(imsi2)






