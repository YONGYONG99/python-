# iris dataset 으로 군집화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head(3))
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# species가 없다 , 독립변수와 종속변수의 구분 없이 그냥 그룹만 나누는거
print()
# dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')
dist_vec = pdist(iris_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')
print('dist_vec : ', dist_vec)
print()
row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

row_clusters = linkage(dist_vec, method='complete')
print('row_clusters : ', row_clusters)
df = pd.DataFrame(row_clusters, columns=['군집id1','군집id2','거리','멤버수'])
print(df)

# dendrogram으로 row_clusters를 시각화
from scipy.cluster.hierarchy import dendrogram
low_dend = dendrogram(row_clusters)
plt.ylabel('유클리드 거리')
plt.show()
































