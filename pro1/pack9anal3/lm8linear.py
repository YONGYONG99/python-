import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # 정규화 지원
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error # 이거 보여주고싶어~
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# 공부시간에 따른 시험 점수 예측모델
df = pd.DataFrame({'study_time':[3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2], 'score':[76,74,74,89,95,84,82,70,60,88,80,50,85,50,60,79]})
print(df)

# 데이터세트 분리 : train / test split - 모델의 과적합(overfitting) 방지가 목적
# 학습이 끝난 모델을 검증하기
train, test = train_test_split(df, test_size=0.4, shuffle=True, random_state=12) # 6:4로 분리
x_train = train[['study_time']] # 모델 학습용 # matrix로 만들기
y_train = train['score']
x_test = test[['study_time']] # 모델 검증용
y_test = test['score']
print(x_train)
print(y_train)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (9, 1) (7, 1) (9,) (9,) / 독립변수는 2차원, 종속변수는 1차원

print()
model = LinearRegression()
model.fit(x_train, y_train) # train으로 학습하고

y_pred = model.predict(x_test) # test로 검증하기
print('예측값 : ',y_pred)
print('실제값 : ',y_test.values)

print('결정계수로 모델 성능을 확인----')
print('결정계수:',r2_score(y_test, y_pred)) # test data를 사용 / 결정계수: 0.21519697375066027

print('------------------------------------------------------------------------------------')
# 참고 : 결정계수는 표본 데이터가 많을수록 그 수치 또한 증가한다.

def linearFunc(df, test_size):
    train, test = train_test_split(df, test_size=test_size, shuffle=True, random_state=12) # 6:4로 분리
    x_train = train[['study_time']] # 모델 학습용 # matrix로 만들기
    y_train = train['score']
    x_test = test[['study_time']] # 모델 검증용
    y_test = test['score']
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    model = LinearRegression() # train
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test) # test
    print('결정계수:',r2_score(y_test, y_pred).round(2))
    print('test data 비율: 전체 데이터의 {}%'.format(i*100))
    # 시각화
    sns.scatterplot(x=df['study_time'], y=df['score'], color='green')
    sns.scatterplot(x=x_test['study_time'], y=y_test, color='red')
    sns.lineplot(x=x_test['study_time'], y=y_pred, color='blue')
    plt.show() # 결정계수는 데이터의 수에 따라 적절한 값을 찾아가게 됨
    
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
for i in test_sizes:
    linearFunc(df, i)
















