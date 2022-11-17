# 선형회귀 분석 모델
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

# 여러 매체의 광고비에 따른 판매량(액?) 데이터 사용
advdf = pd.read_csv("../testdata/Advertising.csv" , usecols=[1,2,3,4]) # 0 번째 칼럼은 쓰지말자
print(advdf.head(3), advdf.shape)
#       tv  radio  newspaper  sales
# 0  230.1   37.8       69.2   22.1
# 1   44.5   39.3       45.1   10.4
# 2   17.2   45.9       69.3    9.3   (200, 4) / 200행 4열 데이터
print(advdf.info()) # 결측치 없네

# 단순선형회귀 : tv, sales
print('상관계수(r) : ', advdf.loc[:, ['tv', 'sales']].corr()) # 0.782224 -> 양의 상관관계가 강하다.
# 'tv'가 'sales'에 영향을 준다 라고 가정

# 모델 생성
lm = smf.ols(formula='sales ~ tv', data = advdf).fit()
print(lm.summary()) 
# R-squared:                 0.612   61% 정도의 설명력
# Prob (F-statistic):        1.47e-42  < 0.05 유의한 모델

# 시각화
# plt.scatter(advdf.tv, advdf.sales)
# plt.xlabel('tv')
# plt.ylabel('sales')
# y_pred = lm.predict(advdf.tv)
# # print('y_pred : ', y_pred.values) # 예측값
# # print('real y : ', advdf.sales.values) # 실제값
# plt.plot(advdf.tv, y_pred, c='r')
# plt.show()

# 예측1 : 새로운 tv 값으로 sales를 추정
x_new = pd.DataFrame({'tv':[230.1, 44.5, 100]})
pred = lm.predict(x_new)
print('예측 결과 : ', pred.values)

print('----' * 20)
print(advdf.corr()) # tv > radio > newspaper
# lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data = advdf).fit()
lm_mul = smf.ols(formula='sales ~ tv + radio', data = advdf).fit()
print(lm_mul.summary()) # newspaper p값 : 0.860 / newspaper는 sales와 관련이 없을 수 있다고 생각을 해야함
# newspaper는 빼고 가자

# 예측2 : 새로운 tv, radio 값으로 sales를 추정
x_new2 = pd.DataFrame({'tv':[230.1, 44.5, 100], 'radio':[30.0, 40.0, 50.0]})
pred2 = lm_mul.predict(x_new2)
print('예측 결과2 : ', pred2.values)

print('***' * 30)
# 회귀분석모형의 적절성을 위한 조건 : 아래의 조건 위배 시에는 변수 제거나 조정을 신중히 고려해야 함.
# 1) 정규성 : 독립변수들의 잔차항이 정규분포를 따라야 한다.
# 2) 독립성 : 독립변수들 간의 값이 서로 관련성이 없어야 한다.
# 3) 선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지면 좋지 않다.
# 4) 등분산성 : 독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.
# 5) 다중공선성 : 독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.

# 잔차항 구하기
# print(advdf.iloc[:, 0:2])
fitted = lm_mul.predict(advdf.iloc[:, 0:2]) # 예측값
residual = advdf['sales'] - fitted # 실제값 - 예측값 -> 잔차 (표본 데이터의 예측값과 실제값의 차이)
print('residual : ', residual[:3]) # 3개만 보자
print(np.mean(residual))

print()
print('선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지면 좋지 않다.')
# 예측값과 잔차가 비슷하게 유지되어야함
sns.regplot(fitted, residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--')
plt.show() # 빨간 실선이 잔차의 추세이다. 점선은 예측값 / 이 둘이 비슷한 수준으로 흘러가야한다.
# 위 데이터는 선형성을 만족하지 못함 -> 이럴땐 다항회귀를 추천

print('정규성 : 독립변수들의 잔차항이 정규분포를 따라야 한다.')
# Q-Q plot으로 확인
import scipy.stats
sr = scipy.stats.zscore(residual) # 표본에 있는 각 값의 z값을 계산
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x,y)
plt.plot([-3,3] , [-3,3], '--', color='gray')
plt.show() # 분포가 점선을 따라 쭉 따라가야하는데 커브를 그리고 있다.
# 정규성을 만족하지 못함 : log를 취하는 등의 작업을 통해 정규분포를 따르도록 데이터 가공 작업 필요

# 정규성은 shapiro test로 확인 가능
print('shapiro test : ', scipy.stats.shapiro(residual)) # pvalue=4.190036317908152e-09 < 0.05 이므로 정규성 만족 못함

print('독립성 : 독립변수들 간의 값이 서로 관련성이 없어야 한다. 잔차가 독립적이어야 함(자기상관이 없어야함)')
# Durbin-Watsion : 잔차의 독립성 만족여부 확인 가능. 2에 근사하면 자기상관이 없다. 0 <- 양의 상관 - 2독립성 - 음의 상관 -> 4 (0에 가까울수록 양의, 4의 가까울수록 음의 상관관계)
# Durbin-Watson: 2.081
# summary()로 확인한 결과 2.081 이므로 잔차의 독립성 만족

print('등분산성 : 독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.')
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess=True , line_kws={'color':'red'})
plt.show() # 빨간색 실선이 완만하게 수평이 될수로 좋다. 
# 일정한 패턴의 곡선을 그리므로 등분산성을 만족하지 못함
# 이럴땐 아웃라이어 확인, 비선형인지 확인, 정규성을 확인
# 정규성은 만족하나 등분산성을 만족하지 못하는 경우에는 가중회귀분석을 추천

print('다중공선성 : 독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.')
# VIF(분산팽창계수)를 사용해서 확인
# VIF가 10이 넘으면 다중공선성 있다고 판단하며 5가 넘으면 주의할 필요가 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
# summay()의 결과에서 coef의 순서는 Intercept:0, tv:1, radio:2
print(variance_inflation_factor(advdf.values, 1)) # tv / 12.570312383503682
print(variance_inflation_factor(advdf.values, 2)) # radio / 3.1534983754953845
# 데이터프레임으로 보기
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(advdf.values, i) for i in range(1,3)]
print(vifdf)

print("참고 : Cook's distance - 극단값(이상치)을 나타내는 지표")
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm_mul).cooks_distance
print(cd.sort_values(ascending=False).head())

import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks' )
plt.show()

print(advdf.iloc[[130, 5, 35, 178, 126]]) # 이상치 데이터로 의심됨으로 제거를 권장

















































