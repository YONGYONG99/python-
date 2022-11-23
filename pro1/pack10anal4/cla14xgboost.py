# XGBoost로 분류 모델 작성
# breast_cancer dataset 사용
# pip install xgboost
# pip install lightgbm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier # xgboost 보다 성능이 우수함. 대용량 처리에 효과적. 데이터가 작으면 과적합 발생 우려 매우 높다.
import xgboost as xgb
from xgboost import plot_importance

dataset = load_breast_cancer()
print(dataset.keys())
x_feature = dataset.data
y_label = dataset.target
cancer_df = pd.DataFrame(data = x_feature, columns = dataset.feature_names)
print(cancer_df.head(2), cancer_df.shape) # (569, 30)
print(dataset.target_names)  # ['malignant' 'benign'] / 양성이냐 음성이냐
print(np.sum(y_label == 0)) # 양성이 212
print(np.sum(y_label == 1)) # 음성이 357

x_train,x_test,y_train,y_test = train_test_split(x_feature, y_label, test_size = 0.2, random_state = 12)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (455, 30) (114, 30) (455,) (114,)

# model
model  = xgb.XGBClassifier(booster='gbtree', max_depth=6, n_estimators = 500 ).fit(x_train, y_train) # 'gbtree' : 의사결정 기반
# model = LGBMClassifier().fit(x_train, y_train) 이것도 가능 나중에 위에 XGB 주석처리하고 해보기(아래 시각화는 지원X)
print(model)
pred = model.predict(x_test)
print('예측값 : ', pred[:10]) # 예측값 :  [0 1 1 1 1 1 1 1 1 0]
print('실제값 : ', y_test[:10]) # 실제값 :  [0 1 1 1 1 1 0 1 1 0]

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('acc : ', acc)

print()
cl_rep = metrics.classification_report(y_test, pred)
print('classification_report : \n', cl_rep)

# 중요 변수 시각화 
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(model, ax = ax)
plt.show() # feature명을 몰라서 f1,f2로 표시되는중 (칼럼의 순서라고 보면됨) , f0이 첫번째 feature











































































































