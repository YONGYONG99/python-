# [로지스틱 분류분석 문제3] 
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
#   참여 칼럼 : 
#   Daily Time Spent on Site : 사이트 이용 시간 (분)
#   Age : 나이,
#   Area Income : 지역 소독,
#   Daily Internet Usage:일별 인터넷 사용량(분),
#   Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# ROC 커브와 AUC 출력

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../testdata/advertisement.csv')
print(df.head(3))
#    Daily Time Spent on Site  Age  ...         Timestamp  Clicked on Ad
# 0                     68.95   35  ...   2016-03-27 0:53              0
# 1                     80.23   31  ...   2016-04-04 1:39              0
# 2                     69.47   26  ...  2016-03-13 20:35              0

df = df.loc[: , ['Daily Time Spent on Site' , 'Age', 'Area Income', 'Daily Internet Usage', 'Clicked on Ad']]
print(df.head(3))

# 상관관계 확인
sns.heatmap(data = df.corr())
plt.show()



# # 데이터 정리
# x = iris.data[:, [2, 3]] # petal.length, petal.width만 참여
# y = iris.target
# print(x[:3])
# print(y[:3], ' ', set(y)) # {0, 1, 2}
#
# # train / test split (7 : 3)
# x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

# print(np.corrcoef(df[:, 0], df[:, 1], df[:, 2], df[:, 3])) ....상관관계 확인하려고 했는데 실패

x = df.loc[: , ['Daily Time Spent on Site' , 'Age', 'Area Income', 'Daily Internet Usage']]
y = df.loc[: , ['Clicked on Ad']]
print(x)
print(y)



model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('y_hat : ', y_hat[:20])

print()
print(confusion_matrix(y, y_hat))
# [[464  36]
#  [ 67 433]]

accuracy = (464 + 433) / 1000  #   (TP + TN) / 전체수
recall = 464 / (464 + 36)      #   TP / (TP + FN)
precision =  464 / (464 + 67)  #   TP / (TP + FP)                     
specificity =  433 / (67 + 433)#   TN / (FP + TN)
fallout = 67 / (67 + 433)      #   FP / (FP + TN)

print('acc(정확도) : ', accuracy)
print('recall(재현율) : ', recall)         # TPR
print('precision(정밀도) : ', precision)
print('specificity(특이도) : ', specificity)
print('fallout(위양성률) : ', fallout)     # FPR
print('fallout(위양성률) : ', 1- specificity)

print()
from sklearn import metrics
acc_sco = metrics.accuracy_score(y, y_hat)
print('acc_sco : ', acc_sco)
print()
cl_rep = metrics.classification_report(y, y_hat)
print('cl_rep : ', cl_rep)

print()
fpr, tpr, threshold = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류결정 임계값(positive 예측값을 결정하는 확률 기준값) : ', threshold)

# ROC 커브
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0,1], [0,1], 'k--', label='random classifier line(AUC:0.5')
plt.plot([fallout], [recall], 'ro', ms=10) # 위양성율, 재현률 # rs 빨간색 동그라미 , ms 크기 10
plt.xlabel('fpr')
plt.ylabel('fpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# AUC : ROC 커브의 면적
print('AUC : ', metrics.auc(fpr, tpr)) # 1에 근사할 수록 좋은 분류 모델 # AUC : 0.9580599999999999
















































