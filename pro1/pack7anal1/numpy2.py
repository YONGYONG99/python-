# numpy
import numpy as np

ss = ['tom' , 'james', 'oscar' , 1]
print(ss, type(ss)) # <class 'list'>

ss2 = np.array(ss)
print(ss2, type(ss2)) # <class 'numpy.ndarray'>

# 메모리 비교 (list vs numpy)
li = list(range(1,10))
print(li)
print(li[0] ,li[1], id(li[0]), id(li[1]), id(li[2])) # 별도의 객체 주소를 기억

print()
num_arr = np.array(li)
print(num_arr[0] ,num_arr[1], id(num_arr[0]), id(num_arr[1]), id(num_arr[2])) # 배열 요소들이 같은 주소 기억