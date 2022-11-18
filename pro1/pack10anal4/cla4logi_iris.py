# Logistic Regression : 다항분류 (얘는 활성화 함수 softmax - 결과값을 확률로 반환)
# LogisticRegression은 다중 클래스를 지원하도록 일반화 되어 있다. softmax 함수를 사용하기 때문이다.
# iris dataset을 사용 - 꽃의 종류를 3가지로 분류한다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn import datasets
from nltk.chunk.util import accuracy

iris = datasets.load_iris()
# print(iris.DESCR) 정보보기
print(iris) # target 0,1,2 세가지인 데이터 확인
print()
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
x = iris.data
print(x)
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])) # 0.96286543 강한 상관관계 확인

# 데이터 정리
x = iris.data[:, [2, 3]] # petal.length, petal.width만 참여
y = iris.target
print(x[:3])
print(y[:3], ' ', set(y)) # {0, 1, 2}

# train / test split (7 : 3)
x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)


# data scaling : 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버/언더 플로우 방지 ... 가능
"""
print(x_train[:3])
# [[3.5 1. ]
#  [5.5 1.8]
#  [5.7 2.5]]
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
# [[-0.05624622 -0.18650096]
#  [ 1.14902997  0.93250481]
#  [ 1.26955759  1.91163486]] 위의 x_train 데이터가 이렇게 표준화

# 스케일링 자료 원복
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
# [[3.5 1. ]
#  [5.5 1.8]
#  [5.7 2.5]]
# 일단 지금은 필요없으니까 주석처리
"""
print('------------------------------------------------------------------')
# 맛보기
model = LogisticRegression(C=1.0, random_state = 0, verbose=0) # C=1.0 : L2 규제(패널티 적용) / 모델을 튜닝할때 조정 -> 값이 작을수록 정규화규제가 강해짐 
print(model)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum())) # 총갯수:45, 오류수:1

# 분류 정확도(accuracy) 방법1
print('%.5f'%accuracy_score(y_test, y_pred)) # 0.97778

# 분류 정확도(accuracy) 방법2
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2])/ len(y_test)) # 0.9777777777777777

# 분류 정확도(accuracy) 방법3
print('test : ', model.score(x_test, y_test))
print('train : ', model.score(x_train, y_train))
# test :  0.9777777777777777
# train :  0.9714285714285714    둘의 차이가 크지 않다 -> 과적합 아니다

# 모델 저장
import pickle
pickle.dump(model, open('cla_model.sav', mode='wb'))
del model

mymodel = pickle.load(open('cla_model.sav', mode='rb'))

print('새로운 값으로 분류 예측 - petal.length, petal.width만 참여')
print(x_test[:1])
new_data = np.array([[5.1, 2.4], [0.3, 0.3], [3.4, 0.2]])
# 참고 : 만약 표준화로 학습했다면 new_data도 표준화 해 줘야 한다.
new_pred = mymodel.predict(new_data)
print('예측 결과 : ', new_pred) # 예측 결과 :  [2 0 1] -> softmax가 반환한 결과 가장 큰 인덱스를 취한 결과

print(mymodel.predict_proba(new_data)) #  내부적으로 softmax가 반환한 결과
# [[9.96150873e-05 8.40157413e-02 9.15884644e-01] -> 2번째가 젤크고
#  [9.98114817e-01 1.88518127e-03 1.94610093e-09] -> 0번째가 젤크고
#  [1.75251457e-01 8.23787901e-01 9.60641983e-04]] -> 1번쨰가 젤크다.

# 시각화
# https://cafe.daum.net/flowlife/SBU0/60 에 코드있음
from matplotlib.colors import ListedColormap

plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공')     










































