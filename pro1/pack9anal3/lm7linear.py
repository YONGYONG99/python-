# 선형회귀분석모델 작성시 LinearRegression을 사용 -> 그런데 이건 summary() 함수를 지원X
# 분석모델을 평가할 수 있는 score 알아보기

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # 정규화 지원
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error # 이거 보여주고싶어~
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# (참고) sklearn은 독립변수가 matrix여야만 한다. (종속변수는 vector)

# 편차가 있는 표본 데이터 작성 
sample_size = 100
np.random.seed(1)

# 정규분포를 따르는 데이터를 뽑아보기
print('표준편차가 같은 두개의 변수를 생성 , 분산이 작음')
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수:', np.corrcoef(x, y)) # 0.99939357

# 스케일링해보자 0에서 1사이로
# 독립변수 x에 대한 정규화
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:5])

# 시각화
# plt.scatter(x_scaled, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y)
y_pred = model.predict(x_scaled)
print('예측값 : ',y_pred[:5])
print('실제값 : ',y[:5])
# print(model.summary()) # AttributeError: 'LinearRegression' object has no attribute 'summary' / summary 사용불가

print()
# 모델 성능 파악용 함수 작성
def RegScore_func(y_true, y_pred):
    print('r2_score(결정계수,설명력):{}'.format(r2_score(y_true, y_pred)))
    print('explained_variance_score(설명분산점수):{}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(MSE, 평균제곱오차):{}'.format(mean_squared_error(y_true, y_pred)))
    # 평균제곱오차 : 예측값에서 실제값(관찰값)을 뺀 값의 제곱의 합을 표본수로 나눈 것
    # RMSE : 평균오차제곱근
    
RegScore_func(y, y_pred)
# r2_score(결정계수,설명력):0.9987875127274646
# explained_variance_score(설명분산점수):0.9987875127274646
# mean_squared_error(MSE, 평균제곱오차):86.14795101998743

###########################################################################################################################

print('\n표준편차가 다른 두개의 변수를 생성, 분산이 큼')
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수:', np.corrcoef(x, y)) # 0.00401167

# 스케일링해보자 0에서 1사이로
# 독립변수 x에 대한 정규화
scaler2 = MinMaxScaler()
x_scaled2 = scaler2.fit_transform(x.reshape(-1,1))
print(x_scaled2[:5])

# 시각화
plt.scatter(x_scaled2, y)
plt.show()

model2 = LinearRegression().fit(x_scaled2, y)
y_pred2 = model2.predict(x_scaled2)
print('예측값 : ',y_pred2[:5])
print('실제값 : ',y[:5])
# print(model.summary()) # AttributeError: 'LinearRegression' object has no attribute 'summary' / summary 사용불가

print()
# 모델 성능 파악용 함수 작성
def RegScore_func2(y_true, y_pred):
    print('r2_score(결정계수,설명력):{}'.format(r2_score(y_true, y_pred)))
    print('explained_variance_score(설명분산점수):{}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(MSE, 평균제곱오차):{}'.format(mean_squared_error(y_true, y_pred)))
    # 평균제곱오차 : 예측값에서 실제값(관찰값)을 뺀 값의 제곱의 합을 표본수로 나눈 것
    # RMSE : 평균오차제곱근
    
RegScore_func2(y, y_pred2)
# r2_score(결정계수,설명력):1.6093526521765433e-05
# explained_variance_score(설명분산점수):1.6093526521765433e-05
# mean_squared_error(MSE, 평균제곱오차):282457.9703485092

