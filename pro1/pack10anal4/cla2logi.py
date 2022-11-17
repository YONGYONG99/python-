# 날씨정보 데이터로 이항분류 : 비가 올지?
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

data = pd.read_csv("../testdata/weather.csv")
print(data.head(2), data.shape) # (366, 12)
#          Date  MinTemp  MaxTemp  Rainfall  ...  Cloud  Temp  RainToday  RainTomorrow
# 0  2016-11-01      8.0     24.3       0.0  ...      7  23.6         No           Yes
# 1  2016-11-02     14.0     26.9       3.6  ...      3  25.7        Yes           Yes

# 데이터 가공좀 하자
data2 = pd.DataFrame()
data2 = data.drop(['Date','RainToday'], axis=1)
print(data2.head(2), data2.shape)
# RainTomorrow에 Yes는1 No는0으로 
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0}) # Dummy
print(data2.head(2))
print(data2['RainTomorrow'].unique()) # [1 0]
# RaiTomorrow : 종속변수, 그 외는 독립변수

# train : test split == 7 : 3 
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape) # (256, 10) (110, 10)
col_select = "+".join(train.columns.difference(['RainTomorrow'])) # RainTomorrow를 제외한 나머지들 엮는다.
my_formula = 'RainTomorrow ~ ' + col_select
print(my_formula)
# model = smf.glm(formula = my_formula, data=train, family=sm.families.Binomial()).fit()
model = smf.logit(formula = my_formula, data=train, family=sm.families.Binomial()).fit()
print('------------------------------------------')
print(model.summary())
# print(model.params) -> 계수값보기
print('예측값 : ', np.around(model.predict(test)[:10].values)) # 예측값 :  [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
print('예측값 : ', np.rint(model.predict(test)[:10].values)) # 예측값 :  [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
print('실제값 : ', test['RainTomorrow'][:10].values) # 실제값 :  [0 0 0 0 0 0 1 1 0 0]

# 정확도
conf_mat = model.pred_table() # AttributeError: 'GLMResults' object has no attribute 'pred_table'
print('conf_mat : \n', conf_mat)
# 위에서 glm쓰면
# model = smf.glm(formula = my_formula, data=train, family=sm.families.Binomial()).fit()
# 안된다. -> logit()으로 바꾸고 다시 하기
# conf_mat : 
#  [[197.   9.]
#  [ 21.  26.]]
print('분류 정확도 : ', (conf_mat[0][0] + conf_mat[1][1]) / len(train)) # 분류 정확도 :  0.87109375
from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('분류 정확도 : ', accuracy_score(test['RainTomorrow'], np.around(pred))) # 분류 정확도 :  0.8727272727272727

# 머신러닝의 포용성(inclusion, tolerance)
# 통계 및 추론 모델로 새로운 값을 예측(정량, 정성)
# y = w * 2 + 0 수학에서는 100%의 답을 원함
# 통계에서는 4의 주변값이 나올 수 있도록 학습을 함.
# 예를 들어 개 이미지 분류를 하는 경우 꼬리가 없는 개도 정확하게 분류되도록 하는 것이 머신러닝의 목적
# 포용성이 있는 모델이라 함은 데이터 분류 인식률이 80%, 90% 등인 것이 100% 인 경우 보다 더 효과적이다.
















































































