# Logistic Regression
# 종속변수와 독립변수 간의 관계로 예측모델을 생성한다는 점에서 선형회귀분석과 유사하다. 하지만
# 독립변수(x)에 의해 종속변수(y)의 범주로 분류한다는 측면에서 분류분석 방법이다. 분류 문제에서 선형
# 예측에 시그모이드 함수를 적용하여 가능한 각 불연속 라벨 값에 대한 확률을 생성하는
# 모델로 이진분류 문제에 흔히 사용되지만 다중클래스 분류(다중 클래스 로지스틱 회귀 또는 다항회귀 )에도 사용될 수 있다.

# 독립변수 : 연속형, 종속변수 : 범주형
# 뉴럴네트워크(신경망)에서 사용됨
import numpy as np
import math

def sigFunc(x):
    return 1 / (1 + math.exp(-x)) # 시그모이드 함수 처리 결과 반환

print(sigFunc(3)) # 인자값은 로짓 전환된 값이라 가정
print(sigFunc(1))
print(sigFunc(37.6))
print(sigFunc(-3.4))
# 0.9525741268224334
# 0.7310585786300049
# 1.0
# 0.032295464698450516
# 모두 0에서 1사이의 값

print('\nmtcars dataset으로 분류 모델 작성')
import statsmodels.api as sm

# statsmodels에 있는 dataset 불러오고
carData = sm.datasets.get_rdataset('mtcars')
print(carData.keys()) # dict_keys(['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data'])

carDatas = sm.datasets.get_rdataset('mtcars').data
print(carDatas.head(3))
#                 mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
# Mazda RX4      21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
# Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
# Datsun 710     22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1

# mpg와 hp가 am(수동,자동)을 결정하는지 알아보기
mtcar = carDatas.loc[:, ['mpg','hp','am']]
print(mtcar.head(3))
#                 mpg   hp  am
# Mazda RX4      21.0  110   1
# Mazda RX4 Wag  21.0  110   1
# Datsun 710     22.8   93   1
print(mtcar['am'].unique()) # [1 0] -> 이항분류

# 연비와 마력수에 따른 변속기 분류(수동, 자동)
# 모델 작성 1 : logit() 함수 사용
import statsmodels.formula.api as smf

formula = 'am ~ hp + mpg'
result = smf.logit(formula=formula, data=mtcar).fit()
print(result)
print(result.summary())
#                  coef    std err          z      P>|z|   
# --------------------------------------------------------
# Intercept    -33.6052     15.077     -2.229      0.026    
# hp             0.0550      0.027      2.045      0.041 --> 0.05 보다 작은지     
# mpg            1.2596      0.567      2.220      0.026 --> 0.05 보다 작은지 확인

# print('예측값 : ', result.predict())

pred = result.predict(mtcar[:10])
print('예측값 : ', pred)

# 0.5 기준으로 분리하면 좋겠네..?
print('\n예측값 : ', pred.values)
print('\n예측값 : ', np.around(pred.values)) # np.around() --> 0.5를 기준으로 0, 1로 출력
print('\n실제값 : ', mtcar['am'][:10].values) # 성능이 우수해보이진 않다

print()
print(mtcar.shape) # (32, 3) 32개
conf_tab = result.pred_table()
print(conf_tab)
# [[16.  3.]
#  [ 3. 10.]]   32개중에서 26개 맞추고 6개 틀림
print('\n분류 정확도 : ',(16 + 10) / len(mtcar)) # 분류 정확도 :  0.8125
print('분류 정확도 : ',(conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))
from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))

print('------------------------------------------------------------------')
# 모델 작성 2 : glm() - 일반화된 선형모델 -> 연속적인 종속변수를 가공? / 선형회귀분석때는 ols 썼었지
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit() # Binomial - 이항분포
print(result2)
print(result2.summary()) # P값 : 0.041, 0.026 
glm_pred = result2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values)) # glm 예측값 :  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
print('glm 실제값 : ', mtcar['am'][:10].values) # glm 실제값 :  [1 1 1 0 0 0 0 0 0 0]

glm_pred2 = result2.predict(mtcar)
print('glm 분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2))) # glm 분류 정확도 :  0.8125

print('\n새로운 값으로 분류 예측')
newdf = mtcar.iloc[:2].copy()
print(newdf)
newdf['mpg'] = [50, 21]
newdf['hp'] = [100, 120]
print(newdf)
new_pred = result2.predict(newdf)
print('분류예측 결과 : ', np.around(new_pred.values)) # 분류예측 결과 :  [1. 0.]




































