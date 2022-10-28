# numpy
import numpy as np

ss = ['tom' , 'james', 'oscar' , 1]
print(ss, type(ss)) # <class 'list'>

ss2 = np.array(ss)
print(ss2, type(ss2)) # <class 'numpy.ndarray'>

# 메모리 비교 (list vs numpy)
li = list(range(1,11))
print(li)
print(li[0] ,li[1], id(li[0]), id(li[1]), id(li[2])) # 별도의 객체 주소를 기억
print(li * 10) # 참조하고 있는 버퍼값을 열번을 돌림

print()
num_arr = np.array(li)
print(num_arr[0] ,num_arr[1], id(num_arr[0]), id(num_arr[1]), id(num_arr[2])) # 배열 요소들이 같은 주소 기억
# numpy는 유연성이 떨어지지만 , 데이터 저장에 효율적이다. 별도의 버퍼를 준비하지 않아서 메모리를 적게쓴다.
print(num_arr * 10) # 버퍼가 없어 요소값에 각각 10을 곱한 결과가 나온다.
print(type(num_arr), num_arr.dtype, num_arr.shape, num_arr.ndim, num_arr.size)
print(num_arr[1], ' ', num_arr[1:5]) # 인덱싱 , 슬라이싱

print()
b= np.array([[1,2,3],[4,5,6]])
print(b.ndim) # 차원의 수 , 2차원
print(b[0], b[0][0], b[[0]])

c = np.zeros((2,2))
print(c)

d = np.ones((2,2))
print(d)

e = np.full((2,2), fill_value = 7)
print(e)

f= np.eye(3)
print(f) # 단위행렬

print()
np.random.seed(0)
print(np.random.rand(5)) # 균등분포
print(np.mean(np.random.rand(5)))

print(np.random.randn(5)) # 정규분포
print(np.mean(np.random.randn(5)))

print() # 수열
print(list(range(1,10)))
print(np.arange(10))

print()
a=np.array([1,2,3,4,5])
print(a[1:4])
print(a[1:4:2])
print(a[1:])
print(a[-4])

b = a
print(a)
print(b)
b[0] = 33
print(a)
print(b)
c = np.copy(a)
c[0] = 77
print(a)
print(c)

print('---------')
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a[:])
print(a.ndim)
print(a[0],a[0][0], a[[0]])
print(a[[0][0]], a[0,0])
print(a[1, 0:3])
print()
print(a)
b = a[:2, 1:3]
print(b)
b[0,0] = 88
print(b)
print(a)
print()
a = np.array([[1,2,3,],[4,5,6],[7,8,9]])
print(a)
print(a.shape)
r1 = a[1, :]

r2 = a[1:2, ]
print(r1, r1.shape)
print(r1.ndim) # 인덱싱 -> 1차원 , 차원 낮춤

print(r2, r2.shape)
print(r2.ndim) # 슬라이싱 -> 2차원 , 차원 유지시킴

print()
c1 = a[:, 1]
c2= a[:, 1:2]
print(c1, c1.shape , c1.size) # 1차원이라 3열만 리턴해주는데 , shape는 리턴값이 tuple이라서 tuple를 나타내기위해 (3,) 이렇게 나옴
print(c2, c2.shape , c2.size)

print()
bool_idx = (a>5)
print(bool_idx)
print(a[bool_idx])
print(a[a > 5])