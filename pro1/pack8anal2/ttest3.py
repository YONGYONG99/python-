# 어느 음식점 매출 자료와 날씨 자료를 활용하여 강수 여부에 따른 매출액 평균에 차이를 검정

# 귀무 : 강수 여부에 따른 음식점 매출액의 평균에 차이가 없다.
# 대립 : 강수 여부에 따른 음식점 매출액의 평균에 차이가 있다.

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

print(data.isnull().sum()) # 결측치 : 0 wow~
print('두 집단 간의 매출액 평균 검정 : t-test')
# print(data['sumRn'] > 0)

data['rain_yn'] = (data['sumRn'] > 0).astype(int) # True면(비옴) 1, False면(비안옴) 0
# data['rain_yn'] = (data.loc[:,('sumRn')] > 0) * 1  # 이렇게도 가능 # True에 1곱하면 1 , False에 1곱하면 0
print(data.head(3))

# boxplot으로 강수 여부에 따른 매출액 시각화
sp = np.array(data.iloc[:, [1, 4]])
# print(sp)
tgroup1 = sp[sp[:, 1] == 0, 0] # 집단1 : 비안오는 그룹의 매출액
tgroup2 = sp[sp[:, 1] == 1, 0] # 집단2 : 비오는 그룹의 매출액
# plt.plot(tgroup1)
# plt.show()
# plt.plot(tgroup2)
# plt.show()
plt.boxplot([tgroup1, tgroup2], meanline = True, showmeans = True, notch=True)
plt.show()

print('평균은 ', np.mean(tgroup1), ' ', np.mean(tgroup2)) # 761040.2542372881  vs  757331.5217391305

# 정규성 확인
print(stats.shapiro(tgroup1).pvalue) # 0.05604
print(stats.shapiro(tgroup2).pvalue) # 0.88273 # 둘다 만족

# 등분산성 확인
print(stats.levene(tgroup1, tgroup2).pvalue) # 0.71234 만족

print(stats.ttest_ind(tgroup1, tgroup2, equal_var = True)) 
# statistic=0.10109828602924716, pvalue=0.919534587722196
# pvalue=0.9195 > 0.05 귀무가설 채택
# 강수 여부에 따른 음식점 매출액의 평균에 차이가 없다.    




