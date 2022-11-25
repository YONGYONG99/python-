# KNN

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=66) # stratify : 데이터가 한쪽으로 쏠리는걸 방지

# stratify 파라미터는 분류 문제를 다룰 때 매우 중요하게 활용되는 파라미터 값 입니다. stratify 값으로는 target 값을 지정해주면 됩니다.
# stratify값을 target 값으로 지정해주면 target의 class 비율을 유지 한 채로 데이터 셋을 split 하게 됩니다. 만약 이 옵션을 지정해주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (426, 30) (143, 30) (426,) (143,)

train_acc = []
test_acc = []

neighbors_setting = range(1, 11)

for n_neigh in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors = n_neigh)
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train)) # KNeighborsClassifier의 score메소드 -> 정확도 구하기
    test_acc.append(clf.score(x_test, y_test))

import numpy as np
print('train 분류 평균 정확도 : ', np.mean(train_acc))
print('test 분류 평균 정확도 : ', np.mean(test_acc))

# k 값 결정?
plt.plot(neighbors_setting, train_acc, label='train acc')
plt.plot(neighbors_setting, test_acc, label='train acc')
plt.ylabel('accuracy')
plt.xlabel('k')
plt.legend()
plt.show() # k값이 늘어나면서 train은 줄어들고 test 늘어나고 있는데
           # 가장 이상적인 k값은 train, test 간격이 좁은 6,7,8 




























