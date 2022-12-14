# 공분산 / 상관계수

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3))

# 표준 편차
print(np.std(df['친밀도']))
print(np.std(df['적절성']))
print(np.std(df['만족도']))
print()
print('--- 공분산 ---')
print(np.cov(df['친밀도'],df['적절성']))
print(np.cov(df['친밀도'],df['만족도']))
print('--- 모아보기 ---')
print(df.cov())
print()
print('--- 상관계수 ---')
print(np.corrcoef(df['친밀도'],df['적절성']))
print(np.corrcoef(df['친밀도'],df['만족도']))
print('--- 모아보기 ---')
print(df.corr())
print('--- 다른 방법 ---')
print(df.corr(method='pearson')) # 등간, 비율 척도. 정규성을 따름
print(df.corr(method='spearman')) # 서열 척도. 정규성을 따르지 않음
print(df.corr(method='kendall')) # spearman과 유사
print()
print('*')
# 만족도에 대한 다른 특성 사이의 상관관계 출력
co_re = df.corr()
print(co_re['만족도'].sort_values(ascending=False))

print()
df.plot(kind = 'scatter', x='만족도', y='적절성')
plt.show()

print()
from pandas.plotting import scatter_matrix
attr = ['친밀도','적절성','만족도']
scatter_matrix(df[attr], figsize=(10,6))
plt.show()

# 상관관계 시각화 : heatmap
import seaborn as sns
sns.heatmap(df.corr())
plt.show() # 색깔이 진할수록 약한 상관관계

print()
# https://github.com/pykwon/python/blob/master/hitmap%EC%B6%94%EA%B0%80.txt 
# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()








