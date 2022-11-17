# 회귀분석 문제 3) 
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv("../testdata/Carseats.csv")
print(df.head(2))
print(df.info())
df = df.drop([df.columns[6], df.columns[9], df.columns[10]], axis=1) # object type인 변수 날리기
print(df.head(2))
print(df.corr()) # 상관관계 보니까 comprice , population , education 이런건 너무 낮아보인다.
lm = smf.ols(formula = 'Sales ~ Income + Advertising + Price + Age', data=df).fit()
print('요약결과:\n', lm.summary())

df_lm = df.iloc[:, [0,2,3,5,6]] # Sales  Income  Advertising  Price  Age 만 뽑기
print(df_lm.head(2))

# 모델 저장하는법
import joblib
joblib.dump(lm, 'lyh.model')
del lm # 삭제도 가능
print('--------지금 부터는 저장된 모델을 읽어 사용함----------------')
#lm_jikwon = joblib.load('lyh.model') # 예를 들어 이용환이 만들고 저장한 모델을 다른 직원이 로드하는 과정
# print(lm_jikwon.summary())
lm = joblib.load('lyh.model') # 일단 이걸로 사용하자


print('--회귀분석보형의 적절성 확인 작업--')
# 잔차 구하기
fitted = lm.predict(df_lm)
residual = df_lm['Sales'] - fitted
print(residual[:3])
print('잔차의 평균 : ', np.mean(residual)) # -1.1102230246251565e-14

print('선형성 ---')
sns.regplot(fitted, residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='blue')
plt.show() # 그래프를 보니 잔차가 다소 일정하게 분포 되있다고 판단가능 -> 선형성 만족한다고 판단
# 잔차가 일정하게 분포되어 있으므로 선형성 만족

print('정규성 ---')
import scipy.stats as stats
sr = stats.zscore(residual) 
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x,y)
plt.plot([-3,3] , [-3,3], '--', color='blue')
plt.show() # 잔차항이 정규분포를 따름
# shapiro test로도 확인해보자
print('shapiro test : ', stats.shapiro(residual)) # pvalue=0.2127407342195511 > 0.05 이므로 정규성 만족!

print('독립성 ---')
# summary()에서
# Durbin-Watson: 1.931   ->  2에 근사하므로 독립성 만족

print('등분산성 ---')
sr = stats.zscore(residual)
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess=True , line_kws={'color':'red'})
plt.show() # 평균선을 기준으로 일정한 패턴을 보이지 않아 등분산성 만족

print('다중공선성 ---')
from statsmodels.stats.outliers_influence import variance_inflation_factor
df2 = df[['Income','Advertising','Price','Age']]
print(df2.head(2))
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(df2.values, i) for i in range(df2.shape[1])]
print(vifdf) # 모든 변수가 10을 넘기지 않음. 다중공선성 없음.
#    vif_value
# 0   5.971040
# 1   1.993726
# 2   9.979281
# 3   8.267760

# 완성된 모델로 새로운 독립변수의 값을 주고 Sales를 예측
new_df = pd.DataFrame({'Income':[33, 55, 66],'Advertising':[10, 13, 16],'Price':[100, 120, 140],'Age':[33, 35, 40],})
pred = lm.predict(new_df)
print('Sales에 대한 예측 결과 : \n', pred)








