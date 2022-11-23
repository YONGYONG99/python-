# Support Vector Machine(SVM)
# 분류와 회귀분석을 위해 주로 사용한다. 두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때,
# SVM 알고리즘은 주어진 데이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는
# 비확률적 이진 선형분류 모델을 만든다. 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로
# 표현되는데 SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘이다.
# SVM은 선형 분류와 더불어 비선형 분류에서도 사용될 수 있다.

# 커널 트릭을 사용함으로 인해 저차원의 데이터를 고차원으로 변형한 후 분류 및 회귀분석을 할 수도 있다.

# SVM으로 XOR 연산 처리
x_data = [      # AND 임
    [0,0,0], # 0 0 일때 0
    [0,1,1], # 0 1 일때 1
    [1,0,1], # 1 0 일때 1
    [1,1,0]  # 1 1 일때 0
]

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

x_df = pd.DataFrame(x_data)
print(x_df)
feature = np.array(x_df.iloc[:, 0:2]) # 슬라이싱 차원축소 x
label = np.array(x_df.iloc[:, 2]) # 인덱싱 차원축소 o
print(feature)
print(label)

model1 = LogisticRegression().fit(feature, label)
pred = model1.predict(feature)
print('Logistic 예측값 : ', pred) # Logistic 예측값 :  [0 0 0 0]

print('Logistic acc : ', metrics.accuracy_score(label,pred)) # Logistic acc :  0.5

print('------------------------------------------------')
model2 = svm.SVC().fit(feature, label)
# model2 = svm.LinearSVC().fit(feature, label)
# model2 = svm.SVC(C=1).fit(feature, label)              이 둘도 가능
pred = model2.predict(feature)
print('svm 예측값 : ', pred) # svm 예측값 :  [0 1 1 0]

print('svm acc : ', metrics.accuracy_score(label,pred)) # svm acc :  1.0

































