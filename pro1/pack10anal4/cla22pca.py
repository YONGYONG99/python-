# 특성공학 기법 중 차원축소(PCA - 주성분 분석)
# n개의 관측치와 p개의 변수로 구성된 데이터를 상관관계가 최소화된 k개의 변수로 축소된 데이터를 만든다.
# 데이터의 분산을 최대한 보존하는 새로운 축을 찾고 그 축에 데이터를 사영시키는 기법, 직교
# 목적 : 독립변수(x , feature)의 갯수를 줄임. 이미지 차원 축소로 용량을 최소화.

# https://datascienceschool.net/02%20mathematics/03.05%20PCA.html?highlight=pca

# iris dataset으로 PCA를 진행
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n = 10
x = iris.data[:n, :2]  # sepal length, sepal width만
print(x)
print(x.shape, type(x)) # (10, 2) <class 'numpy.ndarray'>
print(x.T)

# plt.plot(x.T, 'o:')
# plt.xticks(range(2))
# plt.grid()
# plt.legend(['표본{}'.format(i) for i in range(n)])
# plt.show() # 길이와 너비간에 어느정도 패턴 비율이 있네 눈으로 확인
# 꽃받침 길이가 크면 꽃받침 폭도 같이 커진다는 규칙을 알 수 있다.

'''
# 산점도
df = pd.DataFrame(x)
# print(df)
ax = sns.scatterplot(0, 1, data=pd.DataFrame(x), marker='s', s=100, color=['b'])
for i in range(n):
    ax.text(x[i,0] - 0.05, x[i, 1] - 0.07, '표본{}'.format(i+1))
plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침너비')
plt.axis('equal')
plt.show()
'''

# PCA
pca1 = PCA(n_components=1) # 변환할 차원수
x_low = pca1.fit_transform(x) # 비지도학습 (타겟을 주지않았음) , 차원축소
print('x_low : ', x_low, ' ', x_low.shape) # 차원 축소된 근사 데이터

x2 = pca1.inverse_transform(x_low) # 차원 축소된 근사 데이터를 원복
print('원복된 데이터 : ', x2, ' ' , x2.shape) # 근사치
print(x)
print()
print(x_low[0])
print(x2[0, :])
print(x[0])

# 시각화
'''
ax = sns.scatterplot(0, 1, data=pd.DataFrame(x), marker='s', s=100, color=['r'])
for i in range(n):
    d = 0.03 if x[i, 1] > x2[i, 1] else -0.04
    ax.text(x[i,0] - 0.05, x[i, 1] - d, '표본{}'.format(i+1))
    plt.plot([x[i, 0], x2[i, 0]], [x[i, 1], x2[i, 1]], 'k--')

plt.plot(x2[:, 0], x2[:, 1], 'o-', color='b', markersize=10)
plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침너비')
plt.axis('equal')
plt.show()
'''
print('-------------------------------')
# iris 4개의 열을 모두 참여
x = iris.data
print(x[:3])
pca2 = PCA(n_components = 2)
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2[:3], ' ', x_low2.shape)
print(pca2.explained_variance_ratio_) # 전체 변동성에서 개별 PCA 결과 별로 차지하는 변동성 비율을 제공
# [0.92461872 0.05306648] -> 합 0.976852 / 두개의 component가 원본데이터의 변동성을 약 97%을 설명하고 있다. 
print()
x4 = pca2.inverse_transform(x_low2)
print('최소 자료 : ', x[0])
print('차원 축소 : ', x_low2[0])
print('차원 복귀 : ', x4[0]) # PCA를 통해 근사행렬로 변환됨

print()
iris2 = pd.DataFrame(x_low2, columns=['f1', 'f2'])
print(iris2.head(3)) # 이걸 사용


