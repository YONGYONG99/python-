# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv("../testdata/student.csv")
print(df.head(3))
#     이름  국어  영어  수학
# 0  박치기  90  85  55
# 1  홍길동  70  65  80
# 2  김치국  92  95  76

# 단순선형회귀 : df.국어(독립변수, x), df.수학(종속변수, y) 

# 시각화
# plt.scatter(df.국어, df.수학)
# # 참고 numpy의 polyfit()을 이용하면 slope, intercept를 얻을 수 있음
# slope, intercept = np.polyfit(df.국어, df.수학, 1)
# print('slope:{}, intercept:{}'.format(slope,intercept)) # slope:0.5705497600474351, intercept:32.106922492542004
# plt.plot(df.국어, slope * df.국어 + intercept, color = 'b')
# plt.xlabel('국어')
# plt.ylabel('수학')
# plt.show()

result1 = smf.ols('수학 ~ 국어', data=df).fit()
print(result1.summary()) # p값 : 8.16e-05 -> 유의미한 변수로 판단됨
print()


# 특정국어 점수에 대한 수학 예상 점수 확인해보기
print('국어점수 90에 대한 예상 수학점수는 ', 0.5705  * 90 + 32.1069)
print('국어점수 70에 대한 예상 수학점수는 ', 0.5705 * 70 + 32.1069)
print('국어점수 92에 대한 예상 수학점수는 ', 0.5705 * 92 + 32.1069)


# 키보드로 값 받기 / predict 함수 사용
new_국어 = float(input('국어 점수 : '))
new_data1 = pd.DataFrame({'국어':[new_국어]})
new_pred1 = result1.predict(new_data1)
print('수학 점수 : ', new_pred1.values)

print('--------------------------------------------------------------------------')

# 다중 선형회귀 : df.국어 , df.영어 (feature, x) / df.수학(label, y)
result2 = smf.ols('수학 ~ 국어 + 영어', data=df).fit()
print(result2.summary()) # p값 :  0.000105 -> 유의한 모델로 판단가능하나 / 국어,영어 각각 p값이 0.663, 0.074로 의미있는 변수로 판단하기 어려움

print('국어 90, 영어 85에대한 예상 수학점수는 ', (0.1158 * 90) + (0.5942 * 85) + 22.6238)
print('국어 70, 영어 65에대한 예상 수학점수는 ', (0.1158 * 70) + (0.5942 * 65) + 22.6238)
print('국어 92, 영어 95에대한 예상 수학점수는 ', (0.1158 * 92) + (0.5942 * 95) + 22.6238)

# print('\n predict 함수 사용')
# new_data = pd.DataFrame({'hp':[110, 120, 150], 'wt':[5, 2, 7]})
# new_pred = result2.predict(new_data)
# print('예산 연비 : ', new_pred.values) # hp: 110 , wt: 5일때 / ... / ...

# 키보드로 값 받기
new_국어2 = float(input('국어점수 : '))
new_영어2 = float(input('영어점수 : '))
new_data2 = pd.DataFrame({'국어':[new_국어2], '영어':[new_영어2]})
new_pred2 = result2.predict(new_data2)
print('예상 수학점수 : ', new_pred2.values)











































