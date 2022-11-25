# Perceptron(퍼셉트론, 단층신경망)이 학습할 때 주어진 데이터를 학습하고 에러가 발생한 데이터에 기반하여 
# Weight(가중치)값을 기존에서 새로운 W값으로 업데이트 시켜주면서 학습. 
# input의 가중치합에 대해 임계값을 기준으로 두 가지 output 중 한 가지를 출력하는 구조.

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]
label = np.array([0,0,0,1]) # 0,0일때 0 / 0,1일때 0 / 1,1일때 1 --> and
# label = np.array([0,1,1,1]) # or
# label = np.array([0,1,1,0]) # xor -> 서로 다를때 1 반환
print(label)
# [0 0 0 1]
ml = Perceptron(max_iter=10, eta0=0.1, verbose=1).fit(feature, label)
# max_iter -> 학습량, 1줘보고 10줘보고 test해보자
# eta0 -> 학습률 / 움직이는 폭을 조절하는게 학습률의 역할이다. 즉 낮을수록 찔끔찔끔씩 움직임
# verbose=0 -> 학습내용 안보여줌 / 1 -> 보여줌 
print(ml)
pred = ml.predict(feature)
print('pred : ', pred) # 학습량 1일때 -> pred :  [0 0 0 0] 
print('acc : ', accuracy_score(label, pred)) # 학습량 1일때 -> acc :  0.75

# xor ( label = np.array([0,1,1,0]) ) 일때 아무리 학습률, 학습량 조정해도 못 맞추는거 test해보기 






















