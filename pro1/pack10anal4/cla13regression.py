# DecisionTreeRegressor, RandomForestRegressor로 정량적인 예측 모델 생성

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score


boston = load_boston()
# print(boston.keys())
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])

dfx = pd.DataFrame(boston.data, columns = boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([dfx, dfy], axis=1)
print(df.head(3))
print(df.corr())

# 시각화
cols = ['MEDV', 'RM', 'LSTAT']
# sns.pairplot(df[cols])
# plt.show() 상관관계 눈으로 파악 완료

# 단순선형회귀
x = df[['LSTAT']]
y = df['MEDV'].values
print(x[:3])
print(y[:3])

print('DecisionTreeRegressor ------')
model = DecisionTreeRegressor(criterion = 'mse', random_state=123).fit(x,y)
print('예측값 : ', model.predict(x)[:5]) # 예측값 :  [24.  21.6 34.7 33.4 32.8]
print('실제값 : ', y[:5]) # 실제값 :  [24.  21.6 34.7 33.4 36.2]
print('결정계수 : ', r2_score(y, model.predict(x))) # 결정계수 :  0.9590088126871839

print('RandomForestRegressor ------')
model2 = RandomForestRegressor(criterion = 'mse', n_estimators=100, random_state=123).fit(x,y)
print('예측값 : ', model2.predict(x)[:5]) # 예측값 :  [24.469      21.975      35.48173333 38.808      32.11911667]
print('실제값 : ', y[:5]) # 실제값 :  [24.  21.6 34.7 33.4 36.2]
print('결정계수 : ', r2_score(y, model2.predict(x))) # 결정계수 :  0.9081654854048482



































































