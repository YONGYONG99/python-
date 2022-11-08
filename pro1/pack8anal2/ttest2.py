# 두 집단의 가설검정 – 실습 시 분산을 알지 못하는 것으로 한정하겠다.
# * 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.


import pandas as pd
from scipy import stats
from numpy import average

# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.
# 대립 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 있다.

print(average(male), ' ', average(female)) # 83.8    72.24
print(83.8 - 72.24) # 11.560

# ttest independece 사용
two_sample = stats.ttest_ind(male, female) # 두 개의 표본에 대한 t-test 실시
print(two_sample)
# Ttest_indResult(statistic=1.233193127514512, pvalue=0.2525076844853278)
# 해석 : pvalue=0.2525 > 0.05 이므로 귀무가설 채택. 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다.

print('-------------------------------------------')

# 실습2) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'

data = pd.read_csv("../testdata/two_sample.csv")
print(data.head(3), len(data))

# 귀무 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# 대립 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.

ms = data[['method', 'score']]
print(ms)
# 교육방법별로 데이터 분리
m1 = ms[ms['method'] == 1]
m2 = ms[ms['method'] == 2]

score1 = m1['score']
score2 = m2['score']
# 결측값 확인
print(score1.isnull().sum()) # 0
print(score2.isnull().sum()) # 2  / NaN : 제거, 0 , 평균으로 대체  방법이 있는데
print()
sco1 = score1.fillna(score1.mean()) 
sco2 = score2.fillna(score1.mean()) # 평균으로 대체
print(sco2.isnull().sum()) # 결측값이 없지

# 정규성
import matplotlib.pyplot as plt
import seaborn as sns
# sns.histplot(sco1, kde=True, color='r')
# sns.histplot(sco2, kde=True, color='b')
# plt.show()

print(stats.shapiro(sco1).pvalue) # 0.3679903745651245 > 0.05 정규성 만족
print(stats.shapiro(sco2).pvalue) # 0.7177212238311768 > 0.05 정규성 만족
print()
# 등분산성 : X의 값에 관계없이 Y의 흩어진 정도가 같은 것을 의미한다. 
print(stats.levene(sco1,sco2).pvalue) # 0.43483931631275674 > 0.05 등분산성 만족
print(stats.fligner(sco1,sco2).pvalue) # 0.3751460060312316 > 0.05 등분산성 만족
print(stats.bartlett(sco1,sco2).pvalue) # 0.2676061039182235 > 0.05 등분산성 만족 / 비모수일때 주로 사용

result = stats.ttest_ind(sco1, sco2) # 정규성 만족, 등분산성 만족
print('t-value:%.5f, p-value:%.5f'%result) # t-value:-0.17337, p-value:0.86309
# 해석 : p-value:0.86309 > 0.05 이므로 귀무 채택

print('참고---------------')
result = stats.ttest_ind(sco1, sco2, equal_var=True) # 정규성 만족, 등분산성 만족 
print('t-value:%.5f, p-value:%.5f'%result)


result = stats.ttest_ind(sco1, sco2, equal_var=False) # 정규성 만족, 등분산성 불만족
print('t-value:%.5f, p-value:%.5f'%result)

print()
# result2 = stats.wilcoxon(sco1, sco2) # 정규성을 만족하지 않은 경우 / The samples x and y must have the same length.
result2 = stats.mannwhitneyu(sco1, sco2)
print('t-value:%.5f, p-value:%.5f'%result2)



















