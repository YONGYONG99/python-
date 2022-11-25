# MLP(다층 신경망)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]
# label = np.array([0,0,0,1]) # and
# label = np.array([0,1,1,1]) # or
label = np.array([0,1,1,0]) # xor도 잘 예측한다.
print(label)
# [0 0 0 1]

# model = MLPClassifier(hidden_layer_sizes=5, solver='adam', learning_rate_init=0.01).fit(feature, label)
# hidden_layer_sizes=1 -> 노드가 한개, 5개주고 해보자
# learning_rate_init -> 학습률 / 너무 낮으면 학습하는데 시간이 너무 오래 걸림 , 적당히 줘야함
# solver='adam' -> 디폴트값

# model = MLPClassifier(hidden_layer_sizes=30, solver='adam', learning_rate_init=0.01,
#                       max_iter=10, verbose=1).fit(feature, label)

model = MLPClassifier(hidden_layer_sizes=(10,10,10), solver='adam', learning_rate_init=0.01,
                      max_iter=10, verbose=1).fit(feature, label)
# hidden_layer_sizes=(10,10,10) 이렇게 해도 노드는 30개


pred = model.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))















































