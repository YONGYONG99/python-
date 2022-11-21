# 일반화 모델을 위한 과적합 방지 방법
# iris dataset을 사용
# train/test split, KFold, GridSearchCV

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target
print(train_data[:3])
print(train_label[:3])

print()
# 분류 모델
dt_clf = DecisionTreeClassifier()
print(dt_clf)
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data) 

print('예측값 : ' ,pred)
print('실제값 : ' ,train_label)
print('예측 정확도 : ', accuracy_score(train_label, pred))
# 1.0 -> 분류정확도가 100프로다... 근데 뭔가 찝찝데스요 ~~~~~ 과적합 발생

print('\n과적합 방지 방법1 : train/test split')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = 121)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 4) (45, 4) (105,) (45,)
# print(x_test[:5])
# print(iris.data[:5])
dt_clf.fit(x_train, y_train) # train data로 학습
pred2 = dt_clf.predict(x_test) # test data로 검정

print('예측값 : ', pred2)
print('실제값 : ', y_test)
print('예측 정확도 : ', accuracy_score(y_test, pred2)) # 예측 정확도 :  0.9555555555555556 
# 100%가 되지 않도록 처리함 : 모델의 포용성
# 추론통계는 모든 환경에서 100%가 되어서는 안된다.

print('\n과적합 방지 방법2 : 교차검증(cross validation) - train/test split을 해도 과적합 발생 또는 데이터의 양이 적은 경우')
# 모델 학습 도중 train data를 쪼개 validation data를 만든 후 학습시 검증을 함께 함
# K-fold 교차검증 : train data를 k개 접어 validation data로 사용
from sklearn.model_selection import KFold
import numpy as np
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=123)
kfold = KFold(n_splits = 5)

cv_acc = []
print('iris shape : ', features.shape) # iris shape :  (150, 4)
# 전체 행수 : 150 , 학습데이터 : 4 / 5 -> 120개 , 검증데이터 : 1 / 5 -> 30개

n_iter = 0
for train_index, test_index in kfold.split(features):
    # print('n_iter: ', n_iter)
    # print('train_index: ', train_index)
    # print('test_index: ', test_index)
    # n_iter +=1
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측 (학습도중에 검증도)
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    # 반복할 때 마다 정확도 측정
    acc = np.round(accuracy_score(y_test, pred),3)
    train_size = x_train.shape[0] # (행, 열)
    test_size = x_test.shape[0]
    print('반복수:{0}, 교차검증 정확도 : {1}, 학습데이터 크기:{2}, 검정데이터 크기 : {3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{0}, 검증 데이터 인덱스 : {1}'.format(n_iter, test_index))
    cv_acc.append(acc)

print('평균 검정정확도 : ', np.mean(cv_acc)) # 평균 검정정확도 :  0.9199999999999999

print('\n과적합 방지 방법2-1 : 교차검증(cross validation) - 불균형한 분포(편향, 왜곡)를 가진 경우')
# 불균형이란 특정 레이블 값이 특이하게 많거나 적은 경우
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits = 5)

cv_acc = []
n_iter = 0
for train_index, test_index in skfold.split(features, label):  # label를 같이 준다 이게 차이점
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습 및 예측 (학습도중에 검증도)
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)
    n_iter += 1
    # 반복할 때 마다 정확도 측정
    acc = np.round(accuracy_score(y_test, pred),3)
    train_size = x_train.shape[0] # (행, 열)
    test_size = x_test.shape[0]
    print('반복수:{0}, 교차검증 정확도 : {1}, 학습데이터 크기:{2}, 검정데이터 크기 : {3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{0}, 검증 데이터 인덱스 : {1}'.format(n_iter, test_index))
    cv_acc.append(acc)

print('평균 검정정확도 : ', np.mean(cv_acc))

print('\n과적합 방지 방법3 : cross_val_score')
from sklearn.model_selection import cross_val_score
data = iris.data
label = iris.target

score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv = 5)
print('교차 검증별 정확도 : ', np.round(score, 3))
print('평균 검정 정확도 : ', np.round(np.mean(score), 3))

print('\n과적합 방지 방법4 : GridSearchCV - 모델 생성시 최적의 parameter 찾기')
from sklearn.model_selection import GridSearchCV
# max_depth = 트리의 최대 깊이, min_samples_split = 노드 분할을 위한 최소 표본수 제어 값
para = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}

grid_dtree = GridSearchCV(dt_clf, param_grid = para, cv=3, refit=True) # refit 재학습
grid_dtree.fit(x_train, y_train) # GridSearchCV를 사용하여 자동으로 복수의 내부 모형을 생성한 후 최적의 파라미터를 찾음

import pandas as pd
scores_df = pd.DataFrame(grid_dtree.cv_results_)
# scores_df[['params', 'mean_test_score', 'rank_test_score', 'split1_test_score', 'split2_test_score']]
pd.set_option('display.max_columns',500)
print(scores_df) # 1이 가장 우수한?
print('GridSearchCV 최적 파라미어 : ', grid_dtree.best_params_) # {'max_depth': 2, 'min_samples_split': 2}
print('GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_dtree.best_score_)) # 0.9333

# df_clf = DecisionTreeCalssifier(max_depth = 2, min_samples_split = 2)

best_df_clf = grid_dtree.best_estimator_
print(best_df_clf)
pred = best_df_clf.predict(x_test)
print(pred)
print('best_df_clf acc : ', accuracy_score(y_test, pred))






















