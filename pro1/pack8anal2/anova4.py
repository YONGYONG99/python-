# Two-way ANOVA(이원분산분석) : 요인 복수 - 각 요인의 그룹도 복수
# https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt

import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
plt.rc('font', family='malgun gothic')

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(urllib.request.urlopen(url), delimiter=',')
print(data.head(3), data.shape)
print()

# data.boxplot(column='머리둘레', by='태아수')
# plt.show()
# data.boxplot(column='머리둘레', by='관측자수')
# plt.show()

# 귀무 : 태아수와 관측자수는 태아의 머리둘레와 관련이 없다.
# 대립 : 태아수와 관측자수는 태아의 머리둘레와 관련이 있다.

# 상호 작용을 빼고 한 경우
reg = ols("data['머리둘레'] ~ C(data['태아수']) + C(data['관측자수'])", data=data).fit()
result = anova_lm(reg, type=2)
print(result)

print()
# 상호 작용(교호작용, Interaction이란 한 요인의 효과가 다른 요인의 수준에 의존하는 경우를 말한다.)을 적용 한 경우 <- 더 일반적이라고 볼수 있다.
reg2 = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit()
result2 = anova_lm(reg2, type=2)
print(result2)
# 해석 : p-value 0.3295509 > 0.05 이므로 귀무가설 채택
# 태아수와 관측자수는 태아의 머리둘레와 관련이 없다.