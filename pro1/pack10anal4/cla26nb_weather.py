# 날씨정보로 나이브베이즈 분류기 작성 - 비 예보
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv("../testdata/weather.csv")
print(df.head(3))
print(df.info())

features = df[['MinTemp', 'MaxTemp', 'Rainfall']]
# labels = df['RainTomorrow'].apply(lambda x:1 if x == 'Yes' else 0)
labels = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(features[:3])
#    MinTemp  MaxTemp  Rainfall
# 0      8.0     24.3       0.0
# 1     14.0     26.9       3.6
# 2     13.7     23.4       3.6
print(labels[:3])
# 0    1
# 1    1
# 2    1
print(set(labels)) # {0, 1}

# 7 : 3 split
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (274, 3) (92, 3) (274,) (92,)

# model
gmodel = GaussianNB()
gmodel.fit(train_x, train_y)

pred = gmodel.predict(test_x)
print('예측값 : ',pred[:10]) # 예측값 :  [0 0 0 0 0 0 0 1 0 0]
print('실제값 : ',test_y[:10].values) # 예측값 :  [0 0 0 0 0 0 0 1 0 0]

acc = sum(test_y == pred) / len(pred)
print('acc : ', acc)
print('acc : ', accuracy_score(test_y, pred))
# acc :  0.8695652173913043

# kfold
from sklearn import model_selection
cross_val = model_selection.cross_val_score(gmodel, features, labels, cv=5)
print('교차 검증 : ', cross_val) # 교차 검증 :  [0.51351351 0.78082192 0.82191781 0.79452055 0.80821918]
print('교차 검증 평균 : ', cross_val.mean()) # 교차 검증 평균 :  0.7437985931136617

print('새로운 자료로 분류 예측')
import numpy as np
new_weather = np.array([[8.0, 24.3, 0.0], [10.0, 25.3, 10.0], [10.0, 30.3, 5.0]])
print(gmodel.predict(new_weather))










































