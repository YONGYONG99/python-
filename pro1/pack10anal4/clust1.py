# Clustering(군집화) : 사전정보(label)가 없는 자료에 대해 컴퓨터 스스로가 패턴을 찾아 여러 개의 군집을 형성함
# 비지도학습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

np.random.seed(123)
var = ['x', 'y']
labels = ['점0','점1','점2','점3','점4']
x = np.random.random_sample([5, 2]) * 10
# print(x)
# [[6.96469186 2.86139335]
#  [2.26851454 5.51314769]
#  [7.1946897  4.2310646 ]
#  [9.80764198 6.84829739]
#  [4.80931901 3.92117518]]
df = pd.DataFrame(x, columns=var, index=labels)
print(df)

plt.scatter(x[:, 0], x[:, 1], s=50, c='blue', marker='o')
plt.grid(True)
plt.show()

from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(df, metric='euclidean')  # 데이터(배열)에 각 요소간 거리를 계산한 후 1차원 배열로 반환
print('dist-vec : \n', dist_vec ) # 거리를 잰 값들 / 누가 어떤 거린지는 알기 어려움 -> squareform 사용하자
# dist-vec : 
#  [5.3931329  1.38884785 4.89671004 2.40182631 5.09027885 7.6564396
#  2.99834352 3.69830057 2.40541571 5.79234641]

row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist)
#           점0        점1        점2        점3        점4
# 점0  0.000000  5.393133  1.388848  4.896710  2.401826
# 점1  5.393133  0.000000  5.090279  7.656440  2.998344
# 점2  1.388848  5.090279  0.000000  3.698301  2.405416
# 점3  4.896710  7.656440  3.698301  0.000000  5.792346
# 점4  2.401826  2.998344  2.405416  5.792346  0.000000

# 계층적 군집분석 
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(dist_vec, method='complete')

df = pd.DataFrame(row_clusters)
print(df) # 뭐지?
df = pd.DataFrame(row_clusters , columns=['군집id1', '군집id2', '거리', '멤버수'])
print(df) # ㅇㅇ

# dendrogram으로 row_clusters를 시각화
from scipy.cluster.hierarchy import dendrogram
low_dend = dendrogram(row_clusters, labels=labels)
plt.ylabel('유클리드 거리')
plt.show()












































