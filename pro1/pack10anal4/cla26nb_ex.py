# 독버섯(poisonous)인지 식용버섯(edible)인지 분류
# https://www.kaggle.com/datasets/uciml/mushroom-classification
# feature는 중요변수를 찾아 선택, label:class
#
# 데이터 변수 설명 : 총 23개 변수가 사용됨.
#
# 여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).
# 변수명 변수 설명
# class      edible = e, poisonous = p
# cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
# cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
# cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
# bruises        bruises = t, no = f
# odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
# gill-attachment attached = a, descending = d, free = f, notched = n
# gill-spacing close = c, crowded = w, distant = d
# gill-size       broad = b, narrow = n
# gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
# stalk-shape  enlarging = e, tapering = t
# stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
# stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
# stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
# veil-type      partial = p, universal = u
# veil-color     brown = n, orange = o, white = w, yellow = y
# ring-number none = n, one = o, two = t
# ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
# spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
# population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
# habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d

################# 최현호님 ################################
########################################################
'''
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb

data = pd.read_csv('mushrooms.csv')
print(data.head(3), data.shape)  # (8124, 23)
print(data.info())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
print(data.head(3))
print(data.isnull().sum()) # 널값없음

features = data.iloc[:, 1:23] # 독립
print(features.head(3))
labels = data['class'] # 종속
print(labels.head(3))

# XGBClassifier로 중요 변수 뽑아내기
model = xgb.XGBClassifier(booster = 'gbtree', max_depth = 6, n_estimators=500 ).fit(features, labels)
fig, ax = plt.subplots(figsize = (10, 12))
plot_importance(model, ax = ax) 
plt.show() # spore-print-color : 315, odor : 125, gill-size : 80, cap-color : 61  # 상위4개만 뽑음

i_features = data[['spore-print-color', 'odor', 'gill-size', 'cap-color']] # 중요변수 따로 뽑아서 담아줌
x_train, x_test, y_train, y_test = train_test_split(i_features, labels, test_size = 0.3, random_state=1)

# model
model = GaussianNB().fit(x_train, y_train)
pred = model.predict(x_test)
print('예상값 :', pred[:3])
print('실제값 :', y_test[:3].values)
print('총갯수 :%d, 오류수:%d'%(len(y_test), (y_test != pred).sum()))
print('분류정확도 :', metrics.accuracy_score(y_test, pred))
'''

######################################## 동현이 ######################################
#####################################################################################

# [GaussanNB 문제] 
# 독버섯(poisonous)인지 식용버섯(edible)인지 분류
# https://www.kaggle.com/datasets/uciml/mushroom-classification
# feature는 중요변수를 찾아 선택, label:class
#
# 데이터 변수 설명 : 총 23개 변수가 사용됨.
# 여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).


# 1. 라이브러리 Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


# 2. 데이터 준비
data = pd.read_csv('mushrooms.csv')
print(data.head(3))
#   class cap-shape cap-surface  ... spore-print-color population habitat
# 0     p         x           s  ...                 k          s       u
# 1     e         x           s  ...                 n          n       g
# 2     e         b           s  ...                 n          n       m


# 3. 데이터 정보 확인
print(data.info())
# RangeIndex: 8124 entries, 0 to 8123
#  #   Column                    Non-Null Count  Dtype 
# ---  ------                    --------------  ----- 
#  0   class                     8124 non-null   object
#  1   cap-shape                 8124 non-null   object
#  2   cap-surface               8124 non-null   object
#  3   cap-color                 8124 non-null   object
#  4   bruises                   8124 non-null   object
#  5   odor                      8124 non-null   object
#  6   gill-attachment           8124 non-null   object
#  7   gill-spacing              8124 non-null   object
#  8   gill-size                 8124 non-null   object
#  9   gill-color                8124 non-null   object
#  10  stalk-shape               8124 non-null   object
#  11  stalk-root                8124 non-null   object
#  12  stalk-surface-above-ring  8124 non-null   object
#  13  stalk-surface-below-ring  8124 non-null   object
#  14  stalk-color-above-ring    8124 non-null   object
#  15  stalk-color-below-ring    8124 non-null   object
#  16  veil-type                 8124 non-null   object
#  17  veil-color                8124 non-null   object
#  18  ring-number               8124 non-null   object
#  19  ring-type                 8124 non-null   object
#  20  spore-print-color         8124 non-null   object
#  21  population                8124 non-null   object
#  22  habitat                   8124 non-null   object


# 4. 데이터세트 Dummy Encoding 필요
encoder = LabelEncoder()
for colname in data.columns:
    data[colname] = encoder.fit_transform(data[colname])
print(data.info())
# RangeIndex: 8124 entries, 0 to 8123
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   class                     8124 non-null   int32
#  1   cap-shape                 8124 non-null   int32
#  2   cap-surface               8124 non-null   int32
#  3   cap-color                 8124 non-null   int32
#  4   bruises                   8124 non-null   int32
#  5   odor                      8124 non-null   int32
#  6   gill-attachment           8124 non-null   int32
#  7   gill-spacing              8124 non-null   int32
#  8   gill-size                 8124 non-null   int32
#  9   gill-color                8124 non-null   int32
#  10  stalk-shape               8124 non-null   int32
#  11  stalk-root                8124 non-null   int32
#  12  stalk-surface-above-ring  8124 non-null   int32
#  13  stalk-surface-below-ring  8124 non-null   int32
#  14  stalk-color-above-ring    8124 non-null   int32
#  15  stalk-color-below-ring    8124 non-null   int32
#  16  veil-type                 8124 non-null   int32
#  17  veil-color                8124 non-null   int32
#  18  ring-number               8124 non-null   int32
#  19  ring-type                 8124 non-null   int32
#  20  spore-print-color         8124 non-null   int32
#  21  population                8124 non-null   int32
#  22  habitat                   8124 non-null   int32


# 5. 상관관계 분석
print(data.corr()['class'])
# class                       1.000000
# cap-shape                   0.052951
# cap-surface                 0.178446
# cap-color                  -0.031384
# bruises                    -0.501530
# odor                       -0.093552
# gill-attachment             0.129200
# gill-spacing               -0.348387
# gill-size                   0.540024
# gill-color                 -0.530566
# stalk-shape                -0.102019
# stalk-root                 -0.379361
# stalk-surface-above-ring   -0.334593
# stalk-surface-below-ring   -0.298801
# stalk-color-above-ring     -0.154003
# stalk-color-below-ring     -0.146730
# veil-type                        NaN
# veil-color                  0.145142
# ring-number                -0.214366
# ring-type                  -0.411771
# spore-print-color           0.171961
# population                  0.298686
# habitat                     0.217179

# sns.heatmap(data.corr())
# plt.show()
# 음의 feature : bruises, gill-spacing, gill-color, ring-type
# 양의 feature : gill-size
# label : class


# 5. 특성 추출 - RandomForestClassifier or XGBClassifier
# 5-1. 학습, 테스트 데이터 분리
feature = data.iloc[:, 1:]
label = data['class']
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# 5-2. RandomForestClassifier 모델
rf = RandomForestClassifier(n_estimators=2000, criterion='entropy', random_state=1)
rf.fit(x_train, y_train)
y_rf_pred = rf.predict(x_test)
print('모델 정확도 : ', accuracy_score(y_test, y_rf_pred))
# 모델 정확도 :  1.0
rf_imp_df = pd.DataFrame(rf.feature_importances_.reshape(-1, 1), index=feature.columns)
index = np.array(rf_imp_df.index)
width = rf_imp_df.values.reshape(-1, )
plt.barh(y=index, width=width)
plt.show()
# -> 유의미 feature : odor, gill-color, spore-print-color, gill-size



# 5-3. XGBClassifier 모델
xg = XGBClassifier(n_estimators=2000, random_state=1)
xg.fit(x_train, y_train)
y_xg_pred = xg.predict(x_test)
print('XGBoost 모델 정확도 : ', accuracy_score(y_test, y_xg_pred))
# XGBoost 모델 정확도 :  1.0
fig, ax = plt.subplots(figsize=(10, 16))
plot_importance(xg, ax=ax)
plt.show()
# -> 유의미 feature : odor, cap-color, cap-shape, spore-print-color, gill-size

# [ 필요 특성 찾기 결과 ]
# * 상관관계
# bruises, gill-spacing, gill-color, ring-type, gill-size

# * RandomForest 특성 중요도
# odor, gill-color, spore-print-color, gill-size

# * XGBoost 특성 중요도
# odor, cap-color, cap-shape, spore-print-color, gill-size

# * 결과
# 주 feature : odor, gill-size, spore-print-color
# 예비 feature : cap-color, cap-shape, gill-color


# 6. 필요 칼럼 추출
feature = data[['odor', 'gill-size', 'spore-print-color']]
label = data['class']


# 7. 학습, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 3) (2438, 3) (5686,) (2438,)


# 8. 나이브베이즈 모델
model1 = GaussianNB()
model1.fit(x_train, y_train)


# 9. 예측값, 실제값 비교
y_pred1 = model1.predict(x_test)
print('예측값 : ', y_pred1[:10])
print('실제값 : ', np.array(y_test)[:10])
# 예측값 :  [0 1 1 1 1 1 1 0 0 0]
# 실제값 :  [0 1 1 1 0 1 1 0 1 1]


# 10. 분류 모델 성능 평가 - 정확도
acc = accuracy_score(y_test, y_pred1)
print('model1의 모델 정확도 : ', np.round(acc, 3))
# model1의 모델 정확도 :  0.703


# 11. 모델1의 ROC Curve 그리기
FPR, TPR, _ = roc_curve(y_test, y_pred1)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model1[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model1 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# 12-1. 예비 칼럼 추가 후 모델 구축 및 평가 - cap-color
# 예비 feature : cap-color, cap-shape, gill-color
feature1 = pd.concat([feature, data[['cap-color']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature1, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 4) (2438, 4) (5686,) (2438,)
model2 = GaussianNB()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
acc = accuracy_score(y_test, y_pred2)
print('model2의 모델 정확도 : ', np.round(acc, 3))
# model2의 모델 정확도 :  0.703
FPR, TPR, _ = roc_curve(y_test, y_pred2)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model2[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model2 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


# 12-2. 예비 칼럼 추가 후 모델 구축 및 평가 - gill-color
# 예비 feature : cap-color, cap-shape, gill-color
feature3 = pd.concat([feature, data[['gill-color']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature3, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 4) (2438, 4) (5686,) (2438,)
model3 = GaussianNB()
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
acc = accuracy_score(y_test, y_pred3)
print('model3의 모델 정확도 : ', np.round(acc, 3))
# model3의 모델 정확도 :  0.758
FPR, TPR, _ = roc_curve(y_test, y_pred3)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model3[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model3 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


# 12-3. 예비 칼럼 추가 후 모델 구축 및 평가 - cap-shape
# 예비 feature : cap-color, cap-shape, gill-color
feature4 = pd.concat([feature, data[['cap-shape']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature4, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 4) (2438, 4) (5686,) (2438,)
model4 = GaussianNB()
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
acc = accuracy_score(y_test, y_pred4)
print('model4의 모델 정확도 : ', np.round(acc, 3))
# model4의 모델 정확도 :  0.723
FPR, TPR, _ = roc_curve(y_test, y_pred4)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model4[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model4 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# -> gill-color 선택




# 13-1. gill-color 특성 추가 후, 재진행 - cap-color
# 예비 feature : cap-color, cap-shape
feature = pd.concat([feature, data[['gill-color']]], axis=1)
feature1 = pd.concat([feature, data[['cap-color']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature1, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 5) (2438, 5) (5686,) (2438,)
model5 = GaussianNB()
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
acc = accuracy_score(y_test, y_pred5)
print('model5의 모델 정확도 : ', np.round(acc, 3))
# model5의 모델 정확도 :  0.758
FPR, TPR, _ = roc_curve(y_test, y_pred5)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model5[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model5 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


# 13-1. gill-color 특성 추가 후, 재진행 - cap-shape
# 예비 feature : cap-color, cap-shape
feature2 = pd.concat([feature, data[['cap-shape']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature2, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 5) (2438, 5) (5686,) (2438,)
model6 = GaussianNB()
model6.fit(x_train, y_train)
y_pred6 = model6.predict(x_test)
acc = accuracy_score(y_test, y_pred6)
print('model6의 모델 정확도 : ', np.round(acc, 3))
# model6의 모델 정확도 :  0.767
FPR, TPR, _ = roc_curve(y_test, y_pred6)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model6[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model6 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# -> cap-shape 선택


# 14. cap-shape 특성 추가 후, 재진행 - cap-color
# 예비 feature : cap-color
feature = pd.concat([feature, data[['cap-shape']]], axis=1)
feature1 = pd.concat([feature, data[['cap-color']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature1, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (5686, 6) (2438, 6) (5686,) (2438,)
model7 = GaussianNB()
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
acc = accuracy_score(y_test, y_pred7)
print('model7의 모델 정확도 : ', np.round(acc, 3))
# model7의 모델 정확도 :  0.769
FPR, TPR, _ = roc_curve(y_test, y_pred7)
auc_value = np.round(auc(FPR, TPR), 1)
plt.plot(FPR, TPR, label=f'model7[GaussianNB]({auc_value}) area')
plt.plot([0, 1], [0, 1], '--', label='ROC Curve - AUC(0.5) area')
plt.title('model7 ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# 15. 모델7의 AUC 면적값 추출
print('model7의 AUC 값 : ', auc_value)
# model7의 AUC 값 :  0.8







