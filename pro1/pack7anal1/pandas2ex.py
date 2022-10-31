# https://cafe.daum.net/flowlife/SBU0/10
import pandas as pd
import numpy as np
from boto3.docs import method
from bokeh.layouts import column
# pandas 문제 1)

# a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. 
print(np.random.randn(9,4))
np.random.seed(1)
df= pd.DataFrame(np.random.randn(9,4))
print(df)
# b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
df= pd.DataFrame(np.random.randn(9,4), columns=['No1','No2','No3','No4'])
print(df)
# c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
print(df.mean(axis=0))

print('-----------------------------------------------------------')

# pandas 문제 2)
# a) DataFrame으로 위와 같은 자료를 만드시오. colume(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
df1 = pd.DataFrame(np.arange(10,41,10).reshape(4,1), index=['a','b','c','d'] , columns = ['numbers'] )
print(df1)
# b) c row의 값을 가져오시오.
print(df1.values[2])
print(df1.loc['c'])
# c) a, d row들의 값을 가져오시오.
print(df1.values[0] , df1.values[3])
# d) numbers의 합을 구하시오.

# e) numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.

# f) floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5    

# g) names라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용.