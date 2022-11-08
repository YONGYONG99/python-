# 색인
import pandas as pd
import numpy as np
from boto3.docs import method

# Series의 재색인
data = pd.Series([1,3,2],index=(1,4,2)) # index는 list, tuple, set 가능, 단 set로 주면 index={1,4,2} , 1 2 4 이렇게 나옴
print(data)

# 순서를 재배치
data2 = data.reindex((1,2,4))
print(data2)

print()
data3 = data2.reindex([0,1,2,3,4,5])
print(data3) # 대응값이 없는 인덱스는 NaN(결측값)이 됨

print() # NaN(결측값) 채우기
data3 = data2.reindex([0,1,2,3,4,5], method='ffill') # 바로 이전값으로 채우기
print(data3) 
data3 = data2.reindex([0,1,2,3,4,5], method='pad') # 같은 내용
print(data3) 

data3 = data2.reindex([0,1,2,3,4,5], method='bfill') # 반대
print(data3) 

data3 = data2.reindex([0,1,2,3,4,5], method='backfill') 
print(data3) 

print('------------')
# 조건
df = pd.DataFrame(np.arange(12).reshape(4,3), index = ['1월','2월','3월','4월'], columns=['강남','강북','서초'])
print(df)
print('요기')
print(df['강남'])
print(df['강남'] > 3)
print(df[df['강남']>3]) # 조건이 참인 행을 출력

print('슬라이싱 관련 method : 복수 인덱싱 loc - 라벨지원, iloc - 숫자 지원') # 이것도 자주쓰이나 지금은 맛만 보자
print(df.loc['3월', :]) # 3월 행만 출력하기
print(df.loc[:'2월'])
print(df.loc[:'2월', ['서초']])

print()
print(df.iloc[2])
print(df.iloc[2, :])
print(df.iloc[:3])
print(df.iloc[:3,2])
print(df.iloc[1:3,1:3])

print('산술 연산')
s1 = pd.Series([1,2,3], index=['a','b','c'])
s2 = pd.Series([4,5,6,7], index=['a','b','d','c'])
print(s1)
print(s2)
print(s1+s2) # -,*,/ 모두가능
print(s1.add(s2)) # sub, mul, div

print()
df1 = pd.DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'),index=['서울','대전','부산'])
df2 = pd.DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'),index=['서울','대전','제주','목포'])
print(df1)
print(df2)
print()
print(df1+df2) # -,*,/ 
print(df1.add(df2))
print(df1.add(df2, fill_value = 0)) # NaN은 특정값으로 채움

print()
seri = df1.iloc[0]
print(seri)
print(df1 - seri) # Broadcastion 연산(전파되기)

print()
# 기술적 통계와 관련된 메소드(함수)
df = pd.DataFrame([[1.4, np.nan],[7, -4.5],[np.NaN, np.NAN],[0.5,-1]], columns=['one','two'])
print(df)
print()
print(df.drop(1)) # 행삭제
print(df.isnull()) # 결측치 값 탐지
print(df.notnull())
print()
print(df.dropna()) # na가 있는행 다 지우기
print(df.dropna(how='any')) # 하나라도 있으면
print(df.dropna(how='all')) # 모두 있어야
print(df.dropna(axis='rows'))
print()
print(df.fillna(0)) # 결측치를 0또는 평균 등의 값으로 대체
print(df.fillna(method='ffill'))
print(df.fillna(method='bfill'))
print(df.dropna(subset=['one'])) # 특정열에 NaN이 있는 행 삭제

print('---')
print(df)
print(df.sum())
print(df.sum(axis=0)) # 열의 합
print()
print(df.sum(axis=1)) # 행의 합
print()
print(df.mean(axis=1)) # 행의 평균
print(df.mean(axis=1, skipna=True))  # 행의 평균 : NaN은 연산에서 제외
print(df.mean(axis=1, skipna=False))
print(df.mean(axis=0, skipna=False)) # 열의 평균
print(df.mean(axis=0, skipna=True))
print()
print(df.describe()) # 요약 통계량 보기
print(df.info()) # 구조




