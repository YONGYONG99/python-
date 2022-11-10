# 추론통계 분석 중 비율검정
# - 비율검정 특징
# : 집단의 비율이 어떤 특정한 값과 같은지를 검증.
# : 비율 차이 검정 통계량을 바탕으로 귀무가설의 기각여부를 결정.

# one-sample
# A회사에는 100명 중에 45명이 흡연을 한다. 국가 통계를 보니 국민 흡연율은 35%라고 한다.
# 비율이 같냐? 비율의 동일여부를 검저하시오.
# 귀무 : A회사의 흡연율과 국민 흡연율의 비율은 같다.
# 대립 : A회사의 흡연율과 국민 흡연율의 비율은 다르다.

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = np.array([45])
nobs = np.array([100])
z , p = proportions_ztest(count=count, nobs=nobs, value=0.35)
print('z:{}, p:{}'.format(z, p))
# p:[0.04442318] < 0.05 이므로 귀무 기각. A회사의 흡연율과 국민 흡연율의 비율은 다르다.

print()
# two-sample
# A회사 사람들 300명 중 100명이 커피를 마시고, B회사 사람들 400명 중 170명이 커피를 마셨다.
# 비율이 같냐?

# 귀무 : 비율이 같다.
# 대립 : 비율이 다르다.

count = np.array([100, 170])
nobs = np.array([300, 400])

z , p = proportions_ztest(count=count, nobs=nobs, value=0)
print('z:{}, p:{}'.format(z, p))
# p:[0.04442318] < 0.05 이므로 귀무 기각. A회사의 흡연율과 국민 흡연율의 비율은 다르다.
# 해석 : p:0.0136 < 0.05 이므로 귀무기각. 비율이 다르다.











































































