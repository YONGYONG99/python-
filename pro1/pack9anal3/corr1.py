# 공분산 / 상관계수

import numpy as np
import matplotlib.pyplot as plt

# 공분산 예
print(np.arange(1,6), np.arange(2,7)) # [1 2 3 4 5] [2 3 4 5 6]
print(np.cov(np.arange(1,6), np.arange(2,7)))
# [[2.5 2.5]
#  [2.5 2.5]]
print()
print(np.arange(1,6), (3,3,3,3,3))
print(np.cov(np.arange(1,6), (3,3,3,3,3))) # 0
print()
print(np.arange(1,6), np.arange(6,1,-1))
print(np.cov(np.arange(1,6), np.arange(6,1,-1))) # -2.5
print()
print(np.arange(10,60,10), np.arange(20,70,10))
print(np.cov(np.arange(10,60,10), np.arange(20,70,10))) # 250 
# 맨위랑 같은 패턴의 선형인데 단순 수치로는 비교가 어렵다. 따라서 표준화 필요
# 공분산을 표준화 -> 상관계수
# x1 = [1,2,3,4,5]
# y1 = [2,3,4,5,6]
# x2 = [10,20,30,40,50]
# y2 = [20,30,40,50,60]
# plt.scatter(x1,y1)
# plt.show()
# plt.scatter(x2,y2)
# plt.show()

print('*')
x = [8,3,6,6,9,4,3,9,3,4]
print('x 평균 : ',np.mean(x))
print('x 분산 : ',np.var(x))

# y = [6,2,4,6,9,5,1,8,4,5]
y = [600,200,400,600,900,500,100,800,400,500]
print('y 평균 : ',np.mean(y))
print('y 분산 : ',np.var(y))

# plt.scatter(x,y)
# plt.show()

print('x, y 공분산 : ', np.cov(x,y)[0, 1])
print('x, y 상관계수 : ', np.corrcoef(x,y)[0, 1]) # x, y 상관계수 :  0.8663686463212855
# y = [6,2,4,6,9,5,1,8,4,5]
# y = [600,200,400,600,900,500,100,800,400,500]
# 위 두 y배열은 상관계수가 같다.
print('**')
# 상관계수 보는 다른 방법도 있다.
from scipy import stats
print(stats.pearsonr(x,y))
print(stats.spearmanr(x,y))
print('***')
# 주의 : 공분산이나 상관계수는 선형 데이터인 경우에만 활용 가능하다.
# 비선형인 예
m = [-3,-2,-1,0,1,2,3]
n= [9,4,1,0,1,4,9]
# plt.scatter(m,n)
# plt.show()
print('m, n 공분산 : ', np.cov(m,n)[0, 1])
print('m, n 상관계수 : ', np.corrcoef(m,n)[0, 1])







