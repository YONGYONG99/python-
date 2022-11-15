# 단순선형회귀 모델
# 기본적인 결정론적 선형회귀 방법 : 독립변수에 대해 대응하는 종속변수와 유사한 예측값을 출력하는 함수 f(x)를 찾는 작업이다.
# 즉 결정된 데이터(독립변수,종속변수)를 기반으로 f(x)를 찾는 과정

import pandas  as pd

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3))
#    친밀도  적절성  만족도
# 0    3    4    3
# 1    3    3    2
# 2    4    4    4
print(df.corr())
#           친밀도       적절성       만족도
# 친밀도  1.000000  0.499209  0.467145
# 적절성  0.499209  1.000000  0.766853
# 만족도  0.467145  0.766853  1.000000
print()
import statsmodels.formula.api as smf

# 적절성이 만족도에 영향을 준다라는 가정하에 모델 생성
model = smf.ols(formula = '만족도 ~ 적절성', data = df).fit() # fit() -> 훈련시키는 메소드 / (참고) R에서는 내부적으로 처리됨
# print(model)
print(model.summary()) # 생성된 모델의 요약결과를 반환. 능력치를 확인다고 볼수있다. (나중에 보고서 작성시에도 유용)

# 표를 살펴보면
# Prob (F-statistic):   2.24e-52    -> p-value를 나타낸다.(F 검정을통해 구한) / 0.05 보다 현저히 작으므로 적합한 회귀모델이라고 판단 가능
# R-squared:         0.588 -> 결정계수(설명력) / x가 종속변수의 분산을 설명하는 비율 , 즉 이 모델의 성능을 나타낸다.

#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.7789       0.124      6.273      0.000       0.534       1.023
# 적절성          0.7393        0.038      19.340     0.000       0.664       0.815
#                ㄴ기울기        ㄴ표준오차     ㄴ기울기/표준오차
#                                          ㄴt값으로 F값을 얻을수있고 F값을통해 p값 얻을수있음

# std err가 작아지면 분산의 설명력은 커지고(결정계수 커짐) F나 t값이 커진다.
# F나 t가 커지면 p값은 작아진다. -> 모델이 유의하다고 판단될 가능성이 높아짐 

# SST , SSE , SSR , R² 살펴보기 https://docs.google.com/document/d/1UfCDD_pqiurna0o8wsI2YK3j8XbNY3E3ly_gcSH636Q/edit

print('회귀계수 : ', model.params)
print('결정계수 : ', model.rsquared)
print('유의확률 : ', model.pvalues)
print('예측값 : ', model.predict()[:5])
print('실제값 : ', df.만족도[:5].values)

print()
new_df = pd.DataFrame({'적절성':[4,3,2,1]})
new_pred = model.predict(new_df)
print('예측 결과 : ', new_pred)
# 예측 결과 :  0    3.735963    적절성 4일때
# 1    2.996687               적절성 3일때
# 2    2.257411               ...              
# 3    1.518135               ...
# dtype: float64


























































































