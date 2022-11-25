# MLP : breast_cancer dataset

from sklearn.datasets import load_breast_cancer
from pandas.tests import scalar
cancer = load_breast_cancer()
print(cancer.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

x = cancer['data']
y = cancer['target']
print(cancer.target_names) # ['malignant' 'benign']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (426, 30) (143, 30) (426,) (143,)

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train, x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test) # 표준화한 결과

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), solver='adam',
                     max_iter=100, learning_rate_init=0.1, verbose = 1, random_state=1).fit(x_train, y_train)
pred = mlp.predict(x_test)
print('예측값 : ', pred[:5])
print('실제값 : ', y_test[:5])
print('train acc : ', mlp.score(x_train, y_train))
print('test acc : ', mlp.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))























































