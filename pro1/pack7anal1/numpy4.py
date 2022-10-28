# 배열에 행/열 추가

import numpy as np

aa = np.eye(3)
print(aa, aa.shape)

# 열 추가
bb = np.c_[aa, aa[2]] # 2열과 동일한 열을 추가
print(bb)

# 행 추가
cc = np.r_[aa, [aa[2]]] # 왜 대괄호???
print(cc)

print()
a = np.array([1,2,3])
print('a = ',a)
print(np.c_[a]) # 행을 열로 변환
print(a.reshape(3,1))
print(np.c_[a].shape)
print(a.shape)

print('---append, insert, delete---')
print(a)
#b = np.append(a,[4,5])
b = np.append(a,[4,5], axis = 0) # axis = 0 -> 열방향,행기준으로 append , 1 -> 지금 행하나밖에 없어서 오류
print(b)

c = np.insert(a,1,[6,7],axis=0)
print(c)

# d = np.delete(a,1)
# d = np.delete(a,[1])
d = np.delete(a,[1,2]) # 1번쨰와 2번쨰 지우기
print(d)

print('---2차원----')
aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa,1,99)) # aa에다가 , index , 99 를 넣기 / # aa 배열을 차원 축소 후 1번째 지점에 99을 추가 
print(np.insert(aa,1,99, axis=0)) # 행기준
print(np.insert(aa,1,99, axis=1)) # 열기준

print()
print(aa)
bb = np.arange(10, 19).reshape(3,3)
print(bb)

cc = np.append(aa,bb)
print(cc) # 역시 차원 떨어짐

cc = np.append(aa,bb,axis=0)
print(cc) 

cc = np.append(aa,bb,axis=1)
print(cc) 

print(np.delete(aa,1))
print(np.delete(aa,1,axis=0))
print(np.delete(aa,1,axis=1))











