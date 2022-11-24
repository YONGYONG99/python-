# 나이브 베이즈는 분류기를 만들 수 있는 간단한 기술로써 단일 알고리즘을 통한 훈련이 아닌 일반적인 원칙에 근거한
# 여러 알고리즘들을 이용하여 훈련된다. 모든 나이브 베이즈 분류기는 공통적으로 모든 특성 값은 서로 독립임을 가정한다.
# 예를 들어, 특정 과일을 사과로 분류 가능하게 하는 특성들 (둥글다, 빨갛다, 지름 10cm)은 나이브 베이즈 분류기에서
# 특성들 사이에서 발생할 수 있는 연관성이 없음을 가정하고 각각의 특성들이 특정 과일이 사과일 확률에 독립적으로 기여 하는 것으로 간주한다.

# https://glanceyes.tistory.com/entry/%EB%82%98%EC%9D%B4%EB%B8%8C-%EB%B2%A0%EC%9D%B4%EC%A6%88-%EB%B6%84%EB%A5%98Naive-Bayes-Classification

# 사이킷런의 naive_bayes 서브패키지에서는 다음과 같은 세가지 나이브베이즈 모형 클래스를 제공한다.
# GaussianNB: 정규분포 나이브베이즈
# BernoulliNB: 베르누이분포 나이브베이즈
# MultinomialNB: 다항분포 나이브베이즈

# 조건부 확류 P(Label|Feature)사후확률 = P(Feature|Label)가능도 * P(Label)사전확률 / P(Feature)
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]
print(x)
y = np.array([1,3,5,7,9])

model = GaussianNB().fit(x,y)
print(model)
pred = model.predict(x)
print('분류 정확도 : ', metrics.accuracy_score(y, pred)) # 분류 정확도 :  1.0

# 새로운 값으로 예측
new_x = np.array([[0.1],[0.5],[5],[12]])
new_pred = model.predict(new_x)
print('새로운 예측 결과 : ', new_pred) # 새로운 예측 결과 :  [1 1 9 9]


















































