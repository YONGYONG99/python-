# 이항검정 : 결과가 두 가지 값을 가지는 확률변수의 분포(이항분포) 를 판단하는데 효과적.
# 정규분포는 연속변량인데 반해 이항분포는 이산변량

# binom test

import pandas as pd
import scipy.stats as stats

# 귀무 : 직원을 대상으로 고객대응 교육 후 고객안내 서비스 만족율은 80%이다.
# 대립 : 직원을 대상으로 고객대응 교육 후 고객안내 서비스 만족율은 80%가 아니다. (양측 검정)


data = pd.read_csv("../testdata/one_sample.csv")
print(data)

print(data.survey.unique()) # [1 0]   # data['survey'].unique()

ctab = pd.crosstab(index = data['survey'], columns="count")
ctab.index = ['불만족', '만족']
print(ctab) # 만족 136 , 불만족 14

print('\n양픅 검정 : 방향성이 없다')
x = stats.binom_test([136, 14], p = 0.8 , alternative='two-sided')
print(x) # p-value : 0.000673 < 0.05 귀무 기각 
# 직원을 대상으로 고객대응 교육 후 고객안내 서비스 만족율은 80%가 아니다. (차이만 알려줄뿐 어디가 크고 작은지 방향성X)

print('\n단축 검정 : 방향성이 있다. 크다, 작다')
# 만족 값이 클거라 가정하고 greater
x = stats.binom_test([136, 14], p = 0.8 , alternative='greater')
print(x) # p-value : 0.0003179 < 0.05 귀무 기각 
# 직원을 대상으로 고객대응 교육 후 고객안내 서비스 만족율은 80%보다 크다.
print()
# 불만족 값이 작을거라 가정하고 less
x = stats.binom_test([136, 14], p = 0.8 , alternative='less')
print(x)

















