# Pandas
# 고수준의 자료구조(Series, DataFrame)와 빠르고 쉬운 데이터 분석용 자료구조 및 함수를 제공한다.
#- NumPy의 고성능 배열 계산 기능과 스프레드시트
#- SQL과 같은 RDMBS의 유연한 데이터 조작 기능을 갖고 있다.
#- 세련된 인덱싱 기능으로 쉽게 데이터를 재배치하여 집계 등의 처리를 편리하게 한다.

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# Series는 일련의 객체를 담을 수 있는 1차원 배열과 같은 자료구조로 색인을 갖는다.
obj = Series([3, 7, -5, 4])
# obj = Series((3, 7, -5, 4))
# obj = Series({3, 7, -5, 4})
# list나 tuple은 가능하나 set type은 불가(순서(index)가 없기 때문)
print(obj, type(obj))

obj2 = Series([3,7,-5,4], index=['a', 'b' , 'c' ,'d'])
print(obj2)
print(obj2.sum() , ' ', np.sum(obj2), ' ', sum(obj2))
print(obj2.mean(), obj2.std())
print()
print(obj2.values) # 배열로 반환한다.
print(obj2.index) # index를 반환해준다.

print()
print(obj2[0], obj2['a'], obj2[['a']]) # obj2[['a']] -> a행을 반환해줌

print(obj2[['a' , 'b']])
print(obj2['a':'b']) # 둘다 시리즈 타입
print(obj2[1:4])
print(obj2[[2,1]])
print(obj2>0)
print('a' in obj2)

print('dict로 Series 객체 생성 ---')
names = {'mouse':5000, 'keyboard':25000, 'monitor':'350000'}
obj3 = Series(names)
print(obj3, type(obj3))

obj3.index = ['마우스', '키보드' , '모니터']
print(obj3)

print()
obj3.name = '상품가격'
print(obj3)


# DataFrame :
# 표 모양(2차원 형태 자료)의 자료구조로 여러 개의 칼럼을 갖는다. (Series가 모인 형태)
# 각 칼럼은 서로 다른 종류의 값을 기억할 수 있다.

df = DataFrame(obj3) # Series로 DataFrame 객체 생성.
print(df)

print()
# 같은 길이의 리스트에 담긴 dict type의 데이터를 이용해 DataFrame 객체 생성.
data = {
    'irum':['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'juso':('역삼동', '신당동', '역삼동', '역삼동', '신사동'),
    'nai':[23, 25, 33, 30, 35],
} # 이것도 list와 tuple은 가능하지만 set는 안된다.
print(data)

frame = pd.DataFrame(data)
print(frame, type(frame)) # 타입 : 데이타프레임

print(frame['irum'])
print(frame.irum, type(frame.irum)) # 타입 : 시리즈 타입

print('열 순서 변경---')
print(DataFrame(data, columns=['juso','irum','nai']))

print() # 열추가 해보고 인덱스 바꿔보기
frame2 = DataFrame(data, columns=['irum','juso','nai','tel'], index=['a','b','c','d','e'] )
print(frame2)

# 수정하기
frame2['tel'] = '111-1111'
print(frame2)

val = Series(['222-2222','333-2222','444-2222'], index=['b','c','d'])
frame2['tel'] = val
print(frame2)

print()
print(frame2.T) # 행과 열을 바꾸기

print(frame.values) # 2차원 결과로 반환
print(type(frame.values))
print(frame.values[0,1]) # 0행 1열값 
print(frame.values[0:2])

# 삭제하기
print()
frame3 = frame2.drop('d') # axis=0 이 생략된것 , 행방향이라는 의미
print(frame3)

frame4 = frame2.drop('tel', axis=1) # 열방향 삭제
print(frame4)

print()
print(frame2)
# index명/열이름으로 정렬하기
print(frame2.sort_index(axis=0, ascending=False)) # 행방향 , descening
print()
print(frame2.sort_index(axis=1, ascending=False)) # 열방향 
print() # 순위 매겨주기
print(frame2.rank(axis=0)) # 사전순으로 순위 매기기

print()
print(frame['juso'].value_counts())

print()
data = {
    'juso':['강남구 역삼동','중구 신당동','강남구 대치동'],
    'inwon':[23,25,15]
}

fr = DataFrame(data)
print(fr , type(fr))
result1 = Series([x.split()[0] for x in fr.juso]) # 구 뽑기
result2 = Series([x.split()[1] for x in fr.juso]) # 동 뽑기
print(result1)
print(result2)
print(result1.value_counts())





