# 일원분산분석으로 평균차이 검정 : 한 개의 요인에 따른 여러 개의 집단으로 데이터가 구성됨

# 강남구에 있는 GS 편의점 3개 지역 알바생의 급여에 대한 평균차이 검정을 실시
# 귀무 : 강남구에 있는 GS 편의점 알바생의 급여에 대한 평균은 차이가 없다.
# 대립 : 강남구에 있는 GS 편의점 알바생의 급여에 대한 평균은 차이가 있다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt"
data = pd.read_csv(url, header=None)
print(data.head(), type(data))
# print(data.describe())
data = data.values
print(data[:3], type(data))

print()
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')
print(data[:3], type(data))

# 세 지역 급여 평균 확인
gr1 = data[data[:,1]==1, 0]
gr2 = data[data[:,1]==2, 0]
gr3 = data[data[:,1]==3, 0]
print('gr1 : ', np.mean(gr1)) # 316.6
print('gr2 : ', np.mean(gr2)) # 256.4
print('gr3 : ', np.mean(gr3)) # 278.0 차이?

print()
# 정규성 확인
print(stats.shapiro(gr1).pvalue) # 0.33 > 0.05 만족
print(stats.shapiro(gr2).pvalue)
print(stats.shapiro(gr3).pvalue)
print()
# 등분산성 
print(stats.levene(gr1, gr2, gr3).pvalue) # 0.0458 < 0.05 만족 못하는데 엄청 조금의 차이니까 그냥 만족한다고 보고 진행 가능
# 웰치 anaova 사용가능
print(stats.bartlett(gr1, gr2, gr3).pvalue)

# 데이터의 퍼짐 정도 시각화
# plt.boxplot([gr1, gr2, gr3], showmeans=True)
# plt.show()

# 일원분산분석 방법1 : anova_lm
df = pd.DataFrame(data, columns=['pay', 'group'])
print(df.head(3))

lmodel = ols('pay ~ C(group)', data=df).fit() # ols : 최소자승법
print(anova_lm(lmodel, typ=1))
# 해석 : p-value 0.043589 < 0.05 이므로 귀무가설 기각. 대립가설 채택.
# 강남구에 있는 GS 편의점 알바생의 급여에 대한 평균은 차이가 있다.

print()
# 일원분산분석 방법2 : f_oneway()
f_sta, pvalue = stats.f_oneway(gr1, gr2, gr3)
print('f통계량 : ', f_sta) # 3.71133
print('유의확률 : ', pvalue) # 0.04358

# 각 지역의 평균 차이가 궁금. 사후 검정
# post hoc test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(turkeyResult) # (reject True) 1,2 그룹은 차이가 꽤크네요.
# 시각화
turkeyResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()  













