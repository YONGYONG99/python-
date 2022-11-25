# KNN : k최근접 이웃 알고리즘
# 레이블이 있는 데이터를 사용하여 분류 작업을 하는 알고리즘이다. 데이터로부터 거리가 가까운 k개의
# 다른 데이터의 레이블을 참조하여 분류한다. 대개의 경우에 유클리디안 거리 계산법을 사용하여 거리를
# 측정하는데, 벡터의 크기가 커지면 계산이 복잡해진다.

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

kmodel = KNeighborsClassifier(n_neighbors = 3, weights='distance' , p=2)

train=[
    [5,3,2],
    [1,3,5],
    [4,5,7]
]
label = [0,1,1] # 줄어들기 패턴 -> 0 / 늘어나기 패턴 -> 1

# plt.plot(train, 'o')
# plt.xlim([-1, 5])
# plt.ylim([0,10])
# plt.show()

kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)
# n_neighbors로 k를 정해줘야 한다
# k개의 이웃 중 거리가 가까운 이웃의 영향을 더 많이 받도록 가중치를 설정하려면 weights = "distance"를 지정해줄 수 있다.
# ‘distance’일 때는 거리를 고려한 가중치 평균(average)을 계산합니다.
# 거듭제곱의 크기를 정하는 매개변수인 p가 기본값 2일 때 유클리디안 거리와 같습니다.

kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred : ', pred)
print('정확도 : {:.2f}'.format(kmodel.score(train, label)))

new_data = [[1,2,8], [6,3,1]] # 새로운 늘어나기 패턴과 줄어들기 패턴
new_pred = kmodel.predict(new_data)
print('new_pred : ', new_pred)




































































































