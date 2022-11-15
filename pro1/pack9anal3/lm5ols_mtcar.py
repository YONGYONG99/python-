# mtcars dataset으로 단순/다중회귀 모델 작성 : ols() 사용
# 귀납적 추론 : 개별적인 사실들로부터 일반적인 원리를 이끌어내는 추론방식이다. -> 머신러닝
# 연역적 추론 : 전제로부터 결론을 논리적으로 도출하는 추론방식이다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

# statsmodels이 제공하는 데이터 이용하자
mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3)) # 32 rows X 11 columns
#                 mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
# Mazda RX4      21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
# Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
# Datsun 710     22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
# print(mtcars.corr()) 너무 많으니까 두개씩만 보자
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1]) # -0.7761683718265864
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0,1]) # -0.8676593765172281

# 단순선형회귀 : mtcars.hp(독립변수, x), mtcars.mpg(종속변수, y) 
# 시각화
'''
plt.scatter(mtcars.hp, mtcars.mpg)
# 참고 numpy의 polyfit()을 이용하면 slope, intercept를 얻을 수 있음
slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)
print('slope:{}, intercept:{}'.format(slope,intercept)) # slope:-0.06822827807156362, intercept:30.098860539622482
plt.plot(mtcars.hp, slope * mtcars.hp + intercept, color = 'b')
plt.xlabel('마력수')
plt.ylabel('연비')
plt.show()
'''

result1 = smf.ols('mpg ~ hp', data=mtcars).fit()
print(result1.summary())
print(result1.conf_int(alpha=0.05))
print()
print(result1.summary().tables[0]) # 0 -> 위쪽만보기 , 1 -> 아래쪽보기

print('마력수 110에 대한 연비는 ', -0.088895 * 110 + 30.0989)
print('마력수 50에 대한 연비는 ', -0.088895 * 50 + 30.0989)
print('마력수 200에 대한 연비는 ', -0.088895 * 200 + 30.0989)

print('------------------------')
# 다중선형회귀 : mtcars.hp, mtcars.wt (feature, x), mtcars.mpg(label, y)
result2 = smf.ols('mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary()) # p값 확인해보니 의미있는 변수로 보임

print('마력수 110, 차체 무게 5톤에 대한 연비는 ', (-0.0318 * 110) + (-3.8778 * 5) + 37.2273)

print('\n predict 함수 사용')
new_data = pd.DataFrame({'hp':[110, 120, 150], 'wt':[5, 2, 7]})
new_pred = result2.predict(new_data)
print('예산 연비 : ', new_pred.values) # hp: 110 , wt: 5일때 / ... / ...

# 키보드로 값 받기
new_hp = float(input('새로운 마력수 : '))
new_wt = float(input('새로운 차체무게 : '))
new_data2 = pd.DataFrame({'hp':[new_hp], 'wt':[new_wt]})
new_pred2 = result2.predict(new_data2)
print('예산 연비 : ', new_pred2.values)















