# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    Type
#                             ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../testdata/glass.csv')
print(df.head(3), df.shape) # (214, 10)
print(df['Type'].unique()) # [1 2 3 5 6 7] / 4가 없네..?

# label encoder 해줘야함 / 버전업이후로 xgboost 할떄 'ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5], got [1 2 3 5 6 7]' 에러 떄문에
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
print(df['Type'])
print(df['Type'].unique())

print(df.info()) # 결측치가 없네요


x_features = df.drop('Type', axis=1)  # Type 열은 독립 변수에서 제외
y_labels = df['Type']
print(x_features.head())
print(y_labels.head())
print(x_features.shape, y_labels.shape) # (214, 9) (214,)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_features, y_labels, test_size = 0.2, random_state = 12)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (171, 9) (43, 9) (171,) (43,)


# model
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

model = xgb.XGBClassifier(booster='gbtree', max_depth=7 , n_estimators = 500).fit(x_train, y_train)
print(model)
pred = model.predict(x_test)
print('예측값 : ', pred[:10]) # 예측값 :  [1 1 1 0 5 4 2 5 0 0]
print('실제값 : ', np.array(y_test[:10])) # 실제값 :  [1 1 4 0 5 4 2 5 0 0]





#
#
#
# # model
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score
#
# xgb_clf = XGBClassifier(n_estimators = 5, random_state=12)
# xgb_clf.fit(x_train, y_train, eval_metric='auc', early_stopping_rounds=2,
#             eval_set=[(x_train, y_train), (x_test, y_test)]) # early_stopping_rounds=2 : 조기중단 / 학습을 거치면서 auccarcy가 계속 증가하는 와중에 똑같은값이 2번나오면 종료해라 
#             # eval_set : 학습을 하는 도중에 tset도 학습?? (찾아보기...)         
#
# xgb_roc_curve = roc_auc_score(y_test, xgb_clf.predict_proba(x_test), multi_class="ovr")
# print('ROC AUC : {0:.4f}'.format(xgb_roc_curve)) # ROC AUC : 0.8399
# pred = xgb_clf.predict(x_test)
# print('예측값 : ', pred[:5]) # 예측값 :  [0 0 0 0 0]
# print('실제값 : ', y_test[:5].values) # 실제값 :  [0 0 0 0 0]
#
# from sklearn import metrics
# acc = metrics.accuracy_score(y_test, pred)
# print('acc : ', acc) # acc :  0.9611286503551697


############################################################
############ 선생님 풀이 ################################################
'''
# [XGBoost 문제]  이걸로 풀어야 함
# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    Type
#                           ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import pandas as pd
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.read_csv("../testdata/glass.csv")
print(data.columns)

x = data.drop('Type', axis=1)  # Type 열은 독립 변수에서 제외
y = data['Type']

print(set(y))  # {1, 2, 3, 5, 6, 7}

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y[:3], set(y)) # {0, 1, 2, 3, 4, 5}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

model = xgb.XGBClassifier(booster='gbtree', n_estimators=500, random_state=12)
model.fit(x_train,y_train)

print()  
y_pred = model.predict(x_test)
print('실제값 :', y_pred[:5])
print('예측값:', np.array(y_test[:5]))
print('정확도 :', metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_auc_score
xgb_roc_curve = roc_auc_score(y_test, model.predict_proba(x_test), multi_class="ovr")
# ValueError: multi_class must be in ('ovo', 'ovr') 예외 발생 에러가 나면 multi_class="ovr"를 주자.
print('ROC AUC : {0:.4f}'.format(xgb_roc_curve))

# 중요 변수 시각화
from xgboost import plot_importance
plot_importance(model)
plt.show()
'''








































