# 집단 간 차이분석: 평균 또는 비율 차이를 분석
# : 모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.
# * T-test와 ANOVA의 차이
# - 두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설검정을 한다.
# - 세 집단 이상의 변수에 대한 평균차이를 검정할 경우에는 ANOVA를 이용하여 검정통계량 F값을 구해 가설검정을 한다.

# * 단일 모집단의 평균에 대한 가설검정(one samples t-test)
# 독립변수 : 범주형
# 종속변수 : 연속형

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 실습 예제 1) 단일표본 t검정 (one-sample t-test)
# 하나의 집단에 대한 표본평균이, 예측된 평균과 차이가 있는지 검증
# 어느 남성 집단의 평균키 검정
# 귀무 : 집단의 평균 키가 170이다. 
# 대립 : 집단의 평균 키가 170이 아니다.
one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean()) # 176.2199
print('정규성 확인 : ', stats.shapiro(one_sample)) # pvalue=0.5400 > 0.05 이므로 만족
# plt.plot(one_sample)
# plt.show()

result = stats.ttest_1samp(one_sample, popmean=170) # popmean : 귀무가설의 기대값
# result = stats.ttest_1samp(one_sample, popmean=160)
print('result : statistic(t-value) : {}, p-value:{}'.format(result[0], result[1])) 
# 해석 : p-value:0.15224 > 0.05 이므로 귀무가설 채택. 집단의 평균 키가 170이다.
print('----------------------------------------------')
# 실습 예제 2) 단일표본 t검정 (one-sample t-test)
# 귀무 : 어느 한 집단의 자료들 평균은 0이다.
# 대립 : 어느 한 집단의 자료들 평균은 0이 아니다.

np.random.seed(1)
mu = 0
n = 10
x = stats.norm(mu).rvs(n) # 랜덤한 데이터 10개 생성
print(x , np.array(x).mean()) # -0.0971408

# sns.displot(x, kde=True, rug = True)
# plt.show()
result2 = stats.ttest_1samp(x, popmean=0)
print('result2 : statistic(t-value) : {}, p-value:{}'.format(result2[0], result2[1]))
# p-value:0.8121 > 0.05 이므로 귀무가설 채택. 집단의 평균 키가 170이다.
print()
print('실습 예제 3')
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv
# 귀무 : A중학교 1학년 1반 학생들의 국어 점수 평균은 80이다.
# 대립 : A중학교 1학년 1반 학생들의 국어 점수 평균은 80이 아니다.

data = pd.read_csv("../testdata/student.csv")
print(data.head(3))

print(data.국어.mean()) # 72.9 vs 80.0 차이?
result3 = stats.ttest_1samp(data.국어, popmean=80)
print('result3 : statistic(t-value) : {}, p-value:{}'.format(result3[0], result3[1]))
# 해석 : p-value:0.1985 > 0.05 이므로 귀무가설 채택. A중학교 1학년 1반 학생들의 국어 점수 평균은 80이다.
print()
print("실습 예제 4")
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

babyData = pd.read_csv("../testdata/babyboom.csv")
print(babyData.head(3))
#print(babyData.describe())
print()
fdata = babyData[babyData.gender == 1] # 여아는 1, 남아는 2라고 가정
print(fdata, ' ', len(fdata))
# print(fdata.describe())
print(np.mean(fdata.weight)) # 3132.444 vs 2800 차이가 있는거야 없는거야??

# 정규분포 확인
# sns.distplot(fdata.iloc[:,2],kde = True)
# plt.show()
# stats.probplot(fdata.iloc[:,2], plot=plt) # Q-Q plot
# plt.show()

# print('정규성 :', stats.shapiro(fdata.iloc[:,2])) # pvalue=0.01798 < 0.05 이므로 정규성 만족 못함.
result4 = stats.ttest_1samp(fdata.weight, popmean=2800)
print('result4 : statistic(t-value) : {}, p-value:{}'.format(result4[0], result4[1]))
# 해석 : p-value:0.1985 > 0.05 이므로 귀무가설 채택. A중학교 1학년 1반 학생들의 국어 점수 평균은 80이다.

















