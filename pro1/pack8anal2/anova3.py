# 어느 음식점 매출 자료와 날씨 자료를 활용하여 온도(추움,보통,더움)에 따른 매출액 평균에 차이를 검정

# 귀무 : 음식점 매출액의 평균은 온도에 영향이 없다.
# 대립 : 음식점 매출액의 평균은 온도에 영향이 있다.

# ttest3에서 가져옴
# ----------------------------------------------------------------------
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# 데이터는 data.go.kr을 참조
# 매출 자료
sales_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tsales.csv", dtype={'YMD':'object'}) # int로된 YMD의 type을 object로 바꾼거
print(sales_data.head(3))
print(sales_data.info()) # 328 x 3

# 날씨 자료
wt_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tweather.csv")
print(wt_data.head(3))
print(wt_data.info())
# 두 data 타입을 일치시키기 위해 몇가지 작업을 하고 # dtype={'YMD':'object'}

# 두 데이터를 병합 : 날짜를 사용하자 근데 20190514 , 2018-06-01 이런식으로 표기가 다름 / 형태 변환하자
# wt_data의 tm을 2018-06-01 -> 20180601 형태로 바꾸자
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3)) # 바꿨다
print(wt_data.tail(3))

print('merge -----------')
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(5))
print(frame.tail(5))
print(len(frame)) # 328

print()
# 분석에 사용할 열만 추출
print(frame.columns) # ['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']
data = frame.iloc[:, [0, 1, 7, 8]] # 'YMD', 'AMT', 'maxTa', 'sumRn'
print(data.head(3))
# -----------------------------------------------------------------------

# 일별 최고온도(maxTa)를 구간설정을 해서 범주형 변수를 추가
print(data.maxTa.describe())

data['Ta_gubun'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=[0,1,2]) # 3구간으로 나눔
print(data.isnull().sum()) # 결측치 확인
print(data['Ta_gubun'].unique())

# 세 그룹의 매출액으로 정규성, 등분산성
x1 = np.array(data[data.Ta_gubun == 0])[:, 1]
x2 = np.array(data[data.Ta_gubun == 1].AMT)
x3 = np.array(data[data.Ta_gubun == 2].AMT)
print(x1[:5])
print(x2[:5])
print(x3[:5])

print()
# 정규성
print(stats.ks_2samp(x1, x2).pvalue) # 만족 못함
print(stats.ks_2samp(x1, x3).pvalue)
print(stats.ks_2samp(x2, x3).pvalue)

print()
# 등분산성
print(stats.levene(x1,x2,x3).pvalue) # 0.03900 < 0.05 만족 못함

print('온도별 매출액 평균')
spp = data.loc[:, ['AMT', 'Ta_gubun']]
print(spp.head(2))
print(spp.groupby('Ta_gubun').mean())

print(pd.pivot_table(spp, index=['Ta_gubun'], aggfunc='mean'))
# 1032362 vs 818106 vs 553710 차이?

# anova 진행
sp = np.array(spp)
print(sp[:3])
group1 = sp[sp[:, 1] == 0, 0]
group2 = sp[sp[:, 1] == 1, 0]
group3 = sp[sp[:, 1] == 2, 0]

# 데이터 분포 시각화
# plt.boxplot([group1,group2,group3] , showmeans = True)
# plt.show()

print()
print(stats.f_oneway(group1,group2,group3))
# F_onewayResult(statistic=99.1908012029983, pvalue=2.360737101089604e-34)
# 해석 : pvalue=2.360737101089604e-34 < 0.05 이므로 귀무 기각
# 음식점 매출액의 평균은 온도에 영향이 있다.

print()
# 정규성을 만족하지 않으므로
print(stats.kruskal(group1,group2,group3))
# (statistic=132.7022591443371, pvalue=1.5278142583114522e-29)
# 여전히 귀무 기각 

print()
# 등분산성을 만족하지 않으므로
# pip install pingouin   <-  프롬프트에서 설치
from pingouin import welch_anova
print(welch_anova(data=data, dv='AMT', between='Ta_gubun'))

print()
# 매출액 평균 차이가 궁금. 사후 검정
# post hoc test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=spp.AMT, groups=spp.Ta_gubun, alpha=0.05)
print(turkeyResult) 

# 시각화
turkeyResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()


