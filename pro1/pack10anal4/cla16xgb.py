# 산탄데르 은행 고객 만족 여부 분류 모델
# label name : TARGET - 0(만족), 1(불만족)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_san.csv', encoding='latin-1')
print(df.head(3), df.shape) # (76020, 371) / 열 371개나 있네...
print(df.info())
print()
print(df['TARGET'].value_counts()) 
# 0    73012 
# 1     3008
unsatified_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print('불만족 비율은 {0:.2f}'.format((unsatified_cnt / total_cnt))) # 불만족 비율은 0.04
# pd.set_option('display.max_columns',500)
print(df.describe()) # 데이터들의 분포 함 보자
# var3 min 보면 이상치 변수가 의심된다.
df['var3'].replace(-999999, 2, inplace=True)
# print(df.describe())
df.drop('ID', axis=1, inplace=True)

x_features = df.iloc[:, :-1] # target 뺀거
y_labels = df.iloc[:, -1] # target

print(x_features.shape, y_labels.shape) # (76020, 369) (76020,)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_features, y_labels, test_size = 0.2, random_state = 12)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (60816, 369) (15204, 369) (60816,) (15204,)

train_cnt = y_train.count()
test_cnt = y_test.count()
print('train 데이터 레이블 분포 비율 : ', y_train.value_counts() / train_cnt) # 0 : 0.960257 / 1 : 0.039743
print('test 데이터 레이블 분포 비율 : ', y_test.value_counts() / test_cnt) # 0 : 0.961129 / 1 : 0.038871
# 불만족비율 약 4프로정도 나오네

# model
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_clf = XGBClassifier(n_estimators = 5, random_state=12)
xgb_clf.fit(x_train, y_train, eval_metric='auc', early_stopping_rounds=2,
            eval_set=[(x_train, y_train), (x_test, y_test)]) # early_stopping_rounds=2 : 조기중단 / 학습을 거치면서 auccarcy가 계속 증가하는 와중에 똑같은값이 2번나오면 종료해라 
            # eval_set : 학습을 하는 도중에 tset도 학습?? (찾아보기...)         

xgb_roc_curve = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1])
print('ROC AUC : {0:.4f}'.format(xgb_roc_curve)) # ROC AUC : 0.8399
pred = xgb_clf.predict(x_test)
print('예측값 : ', pred[:5]) # 예측값 :  [0 0 0 0 0]
print('실제값 : ', y_test[:5].values) # 실제값 :  [0 0 0 0 0]

from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('acc : ', acc) # acc :  0.9611286503551697

# GridSearchCV로 best parameter 구한 후 모델 작성
# 중요변수를 알아내 feature를 줄이는 작업
# 성격이 유사한 변수들에 대해 차원축소를 하여 feature를 줄이는 작업
# 등등등...
































































