# iris dataset으로 지도학습(KNN) / 비지도학습(K-Means) 

from sklearn.datasets import load_iris
import numpy as np


iris_dataset = load_iris()

print(iris_dataset['data'][:3])
print(iris_dataset['feature_names'])

# train / test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset['data'], iris_dataset['target'],
                                                    test_size=0.25, random_state=42)

# 지도학습(KNN)
from sklearn.neighbors import KNeighborsClassifier

knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(train_x, train_y) # feature, label

predict_label = knnModel.predict(test_x)
print(predict_label)
from sklearn import metrics
print('acc : ', metrics.accuracy_score(test_y, predict_label))

print()
# 비지도학습(K-Means)
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters = 3, init='k-means++', random_state=0)
kmeansModel.fit(train_x) # feature 만 줌

print(kmeansModel.labels_)

print('0 cluster : ', train_y[kmeansModel.labels_ == 0])
print('1 cluster : ', train_y[kmeansModel.labels_ == 1])
print('2 cluster : ', train_y[kmeansModel.labels_ == 2])

pred_cluster = kmeansModel.predict(test_x)
print('pred_cluster : ', pred_cluster)

# 성능보기
np_arr = np.array(pred_cluster)

pred_label = np_arr.tolist()
print(pred_label)
print('test acc : {:.2f}'.format(np.mean(pred_label == test_y)))


































