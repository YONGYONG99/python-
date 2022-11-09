# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance, F분포를 이용)을 이용하게 된다.
# * 서로 독립인 세 집단의 평균 차이 검정

# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. three_sample.csv'
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np

# 교육방법이 한 개의 요인. 요인이 세 가지 방법으로 구분 - 세 개의 집단이 됨 
# 귀무 : 세 가지 교육방법에 따른 시험점수에 차이가 없다.
# 대립 : 세 가지 교육방법에 따른 시험점수에 차이가 있다.

data =  pd.read_csv("../testdata/three_sample.csv")
print(data.head(3))
print(data.shape)
print(data.describe())

# plt.boxplot(data.score)
# plt.hist(data.score)
# plt.show() # 보니까 두녀석이 아웃라이어네

data = data.query('score <= 100') #  그 두녀석 제거
print(data.shape) # 78명된거 확인

result = data[['method', 'score']]
# print(result , type(result))
m1 = result[result['method'] == 1]
m2 = result[result['method'] == 2]
m3 = result[result['method'] == 3]
# print(m1)
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']
print(score1[:3])
print(score2[:3])
print(score3[:3])

print('평균 : ', np.mean(score1), ' ', np.mean(score2), ' ', np.mean(score3))
# 평균 :  67.384   68.357   68.875

# 정규성 확인
# 정규성을 만족하면 anova, 만족하지 않으면 kruskal-wallis test 사용
print(stats.shapiro(score1).pvalue) # 0.17467 > 0.05 만족
print(stats.shapiro(score2).pvalue)
print(stats.shapiro(score3).pvalue)
# 두개의 표본이 같은 분포를 따르는지 확인
print(stats.ks_2samp(score1, score2).pvalue) # 0.3096 > 0.05 만족
print(stats.ks_2samp(score1, score3).pvalue)
print(stats.ks_2samp(score2, score3).pvalue)

print()
# 등분산성 확인
# 등분산성을 만족하지 않으면 welchi_anova test 사용
print(stats.levene(score1,score2,score3).pvalue) # 0.1132 > 0.05 만족
print(stats.fligner(score1,score2,score3).pvalue)
print(stats.bartlett(score1,score2,score3).pvalue)
# 참고 : 등분산성을 만족하지 않는 경우 대안 (추천)
# 데이터를 정규화, 표준화, 자연 log를 붙이는 방법
print()
print('교육방법별 건수 : 교차표')
data2 = pd.crosstab(index=data['method'], columns='count')
data2.index = ['방법1','방법2','방법3']
print(data2)

print('교육방법별 만족여부 : 교차표')
data3 = pd.crosstab(data.method, data.survey)
data3.index = ['방법1','방법2','방법3']
data3.columns = ['만족','불만족']
print(data3)

# anova 진행
import statsmodels.api as sm

# 종속변수 ~ 독립변수
reg = ols('score ~ method', data=data).fit() # anova_lm : args로 linear model 필요해서 만드는중 
table = sm.stats.anova_lm(reg , type=2) 
print(table) # 분산 분석표 출력 / F검정을 통해 나온 p값을 볼수있네
print(27.980888/228.922822) 
# 해석 : p-value 0.727597 > 0.05 이므로 귀무 채택 / 세 가지 교육방법에 따른 시험점수에 차이가 없다.

print('-----------참고---------------------')
# 독립변수 : 2 , 종속변수 : 1
reg2 = ols('score ~ C(method + survey)', data=data).fit() 
table2 = sm.stats.anova_lm(reg2 , type=2) 
print(table) 
# -------------------------------------------------

print('사후검정 : 그룹 간의 평균에 차이여부를 알려주나 각 그룹 사이에 평균의 차이는 알려주지 않는다. 그래서 사후검정 수행')
# post hoc test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(turkeyResult)
# 시각화
turkeyResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show() # 그림 보면 그래프가 겹친것을 보면 평균의 차이가 크지 않다는것을 볼수있음 / 평균의 차이가 크면 그래프가 안겹침




























































