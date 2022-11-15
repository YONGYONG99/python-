# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#     참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5

from io import StringIO
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# StringIO 클래스 사용해보자 / 문자열을 파일 객체처럼 다룰 수 있도록 하는 클래스
data = StringIO("""
구분,지상파,종편,운동
1,0.9,0.7,4.2
2,1.2,1.0,3.8
3,1.2,1.3,3.5
4,1.9,2.0,4.0
5,3.3,3.9,2.5
6,4.1,3.9,2.0
7,5.8,4.1,1.3
8,2.8,2.1,2.4
9,3.8,3.1,1.3
10,4.8,3.1,35.0
11,NaN,3.5,4.0
12,0.9,0.7,4.2
13,3.0,2.0,1.8
14,2.2,1.5,3.5
15,2.0,2.0,3.5
""")

df = pd.read_csv(data)
print(df.head(3))
#    구분  지상파   종편   운동
# 0   1  0.9     0.7  4.2
# 1   2  1.2     1.0  3.8
# 2   3  1.2     1.3  3.5

print(df.info())
 #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   구분      15 non-null     int64  
#  1   지상파     14 non-null     float64
#  2   종편      15 non-null     float64
#  3   운동      15 non-null     float64
# 지상파에 결측값 있나보다

# 결측값 평균으로 채우기
avg = df['지상파'].mean()
df = df.fillna(avg)
print(df)

# 이상치(아웃라이어) 제거
for d in df.운동:
    if d > 10:
        df = df[df.운동 != d]

for d in df.지상파:
    if d > 10:
        df = df[df.지상파 != d]

# 회귀분석모델
x = df.지상파
y = df.운동
# plt.scatter(x, y)
# plt.show()

model1 = stats.linregress(x, y)
print('slope : ', model1.slope) # slope :  -0.6684550167105406
print('intercept : ', model1.intercept) # intercept :  4.709676019780582

pred_data = np.polyval([model1.slope, model1.intercept], df.지상파)

plt.scatter(x, y)
plt.plot(df.지상파, pred_data, 'r')
plt.show()

















