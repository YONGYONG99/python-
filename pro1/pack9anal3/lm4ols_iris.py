# 선형회귀분석 : iris dataset으로 모델 생성
# 약한 상관관계 변수, 강한 상관관계 변수로 모델 작성

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# seaborn이 제공하는 iris dataset 사용해보자
iris = sns.load_dataset('iris') 

print(iris.head(3))
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa

print(type(iris)) # <class 'pandas.core.frame.DataFrame'>

print(iris.corr()) 
#               sepal_length  sepal_width  petal_length  petal_width
# sepal_length      1.000000    -0.117570      0.871754     0.817941
# sepal_width      -0.117570     1.000000     -0.428440    -0.366126
# petal_length      0.871754    -0.428440      1.000000     0.962865
# petal_width       0.817941    -0.366126      0.962865     1.000000

print()
# 약한 상관관계 변수
print('연습1 : 약한 상관관계 변수 - sepal_length, sepal_width')
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('요약결과1 : ' , result1.summary())
# R-squared:   0.014
# Prob (F-statistic):  0.152  
print('R-squared : ', result1.rsquared) # 0.0138 이므로 설명력이 너무 낮다.
print('p-value : ', result1.pvalues[1]) # 0.15189 > 0.05 이므로 독립변수로 유의하지 않다.
# 서로 영향을 주는 변수라고 판단하기 어렵다
print()
# 이 의미없는 모델로 예측 결과 확인해보기
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result1.predict()[:5])

# model1 시각화
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')
# plt.show()

print()
# 강한 상관관계 변수
print('연습2 : 강한 상관관계 변수 - sepal_length, petal_width')
result2 = smf.ols(formula='sepal_length ~ petal_width', data=iris).fit()
print('요약결과2 : ' , result2.summary())
# R-squared:   0.669
# Prob (F-statistic):  2.33e-37 
print('R-squared : ', result2.rsquared) # 0.6690 
print('p-value : ', result2.pvalues[1]) # 2.325498079793509e-37 

print()
# 의미있는 모델로 예측 결과 확인
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result2.predict()[:5])

# model2 시각화
plt.scatter(iris.petal_width, iris.sepal_length)
plt.plot(iris.petal_width, result2.predict(), color='b')
plt.show()

print('\n새로운 값(petal_width)으로 결과 예측(sepal_length)')
new_data = pd.DataFrame({'petal_width':[1.1, 3.3, 5.5, 7.7]})
y_pred = result2.predict(new_data)
print('예측결과 : ', y_pred.values)


