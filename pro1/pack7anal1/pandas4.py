# DataFrame merge(병합)
import numpy as np
import pandas as pd
df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','b','c','a','a','b',]})
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'], 'data2':range(3)})
print(df2)

print()
print(pd.merge(df1, df2, on='key')) # key를 기준으로 병합, inner join

print()
print(pd.merge(df1, df2, on='key', how='inner')) # key를 기준으로 병합, inner join

print()
print(pd.merge(df1, df2, on='key', how='outer')) # key를 기준으로 병합, full outer join

print()
print(pd.merge(df1, df2, on='key', how='left')) # key를 기준으로 병합, left outer join

print()
print(pd.merge(df1, df2, on='key', how='right')) # key를 기준으로 병합, right outer join

print()
# 공통 칼럼이 없는 경우
df3 = pd.DataFrame({'key2':['a','b','d'], 'data2':range(3)})
print(df3)
print(pd.merge(df1, df3, left_on='key', right_on='key2'))

print()
print(pd.concat([df1, df3])) # 이어 붙이기 axis = 0 은 생략
print(pd.concat([df1, df3], axis =1)) # 열단위

print('\n--------그룹화 연산 : pivot, groupby, pivot_table----------')
# 데이터 열 중에서 두 개의 키를 사용하여 데이터를 선택 후 연산. 구조 변경 후 집계표 작성.
data = {
    'city':['강남', '강북', '강남', '강북'],
    'year':[2000,2001,2002,2002],
    'pop':[3.3 , 2.5, 3.0, 2.0]
}
df = pd.DataFrame(data)
print(df)

print(df.pivot('city','year','pop'))
print()

print(df.pivot('year','city','pop'))
print()
print(df.set_index(['city','year']).unstack())

print()
print(df.groupby(['city']).sum())
print(df.groupby(['city']).agg('sum')) # 두개 같음
print(df.groupby(['city','year']).mean())

print()
print(df.pivot_table(index=['city']))
print(df.pivot_table(index=['city'], aggfunc=np.mean)) # aggfunc=np.mean 기본값
print(df.pivot_table(index=['city','year'], aggfunc=[len, np.sum]))
print(df.pivot_table(values=['pop'], index=['city']))
print(df.pivot_table(values=['pop'], index=['city'], aggfunc=np.mean))
print(df.pivot_table(values=['pop'], index=['city'], columns='year'))
print(df.pivot_table(values=['pop'], index=['year'], columns='city', margins=True)) # 행과 열의 합 출력
print(df.pivot_table(values=['pop'], index=['year'], columns='city', margins=True, fill_value=0))














