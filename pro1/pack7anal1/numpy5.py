# 배열에서 조건 연산 where(조건, 참, 거짓) ... 

import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])

condionData  = np.array([True, False, True])
result = np.where(condionData, x, y)
print(result)

print()
aa = np.where(x >=2)
print(aa)
print(x[aa])
print(np.where(x >=2 , 'T', 'F'))
print(np.where(x >=2 , x+10, x*5))

print('배열 결합')
kbs = np.concatenate([x,y])
print(kbs)
print('배열 분할')
x1,x2 = np.split(kbs,2)
print(x1)
print(x2)

print()
a = np.arange(1, 17).reshape(4,4)
print(a)
x1, x2 = np.hsplit(a,2)
print(x1)
print(x2)

x1, x2 = np.vsplit(a,2)
print(x1)
print(x2)

print('---sampling : 복원 / 비복원----') # 통계에선 비복원이 기본이다.
li = np.array([1,2,3,4,5,6,7])

# 복원 추출
for _ in range(5):
    print(li[np.random.randint(0, len(li) -1)], end = ' ')

print()
import random
# 비복원 추출
print(random.sample(list(li), k=5))
print(random.sample(range(1,46), k=6))

print()
# choice()
print(list(np.random.choice(range(1,46), 6))) # 복원 추출
print(list(np.random.choice(range(1,46), 6 , replace=True))) # 복원 추출
print(list(np.random.choice(range(1,46), 6 , replace=False))) # 비복원 추출

print()
datas = 'air book cat d e f god'
ar = datas.split(sep=' ')
print(ar)
print(np.random.choice(ar,3, p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4])) # god가 많이 나오지

# https://cafe.daum.net/flowlife/SBU0/10  -> 넘파이 연습문제

