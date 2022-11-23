# [Randomforest 문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
# dataset은 winequality-red.csv 
# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv("winequality-red.csv")
print(df.head(3))
print(df.info)              # [3 rows x 12 columns]
print(df.isnull().sum())    # 결측치 X
print(df.shape)             # (1596, 12)

# feature와 label 분리
features = df.drop(['quality'], axis=1)
label = df['quality']

# 학습, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1117, 11) (479, 11) (1117,) (479,)

# RandomForest 분류 모델 
model = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=1)
model.fit(x_train, y_train)

# 예측값 실제값
y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:3])
print('실제값 : ', np.array(y_test)[:3])

# 모델 정확도
acc = accuracy_score(y_test, y_pred)
print('모델 정확도 : ', np.round(acc, 3))



