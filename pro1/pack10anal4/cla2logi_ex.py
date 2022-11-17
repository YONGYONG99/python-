# [로지스틱 분류분석 문제1]
# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

from io import StringIO
from scipy import stats
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

data = StringIO("""
요일,외식유무,소득수준
토,0,57
토,0,39
토,0,28
화,1,60
토,0,31
월,1,42
토,1,54
토,1,65
토,0,45
토,0,37
토,1,98
토,1,60
토,0,41
토,1,52
일,1,75
월,1,45
화,0,46
수,0,39
목,1,70
금,1,44
토,1,74
토,1,65
토,0,46
토,0,39
일,1,60
토,1,44
일,0,30
토,0,34
""")

df = pd.read_csv(data)
print(df.head(3))
#   요일  외식유무  소득수준
# 0  토     0    57
# 1  토     0    39
# 2  토     0    28
print(df['요일'].unique()) # ['토' '화' '월' '일' '수' '목' '금']
print(df['외식유무'].unique()) # [0 1]
date = df[df['요일'] == '월'].index | df[df['요일'] == '화'].index | df[df['요일'] == '수'].index | df[df['요일'] == '목'].index | df[df['요일'] == '금'].index
df2 = df.drop(date)
print(df2)
print(df2['요일'].unique()) # ['토' '일']

# train : test split == 7 : 3
train, test = train_test_split(df2, test_size=0.3, random_state=42)
print(train.shape, test.shape) # (14, 3) (7, 3)
formula = '외식유무 ~ 소득수준'
result = smf.logit(formula=formula , data = df2, family=sm.families.Binomial()).fit()
print(result)
print(result.summary()) # pvalue : 0.018 
print('예측값 : ', np.around(result.predict(test)[:7].values)) # 예측값 :  [1. 1. 0. 0. 1. 1. 1.]
print('실제값 : ', test['외식유무'][:7].values) # 실제값 :  [0 1 0 0 1 1 1]

# 정확도
# conf_mat = result.pred_table()
# print('conf_mat : \n', conf_mat)
# [[10.  1.]
# [ 1.  9.]] 21개중 19개 맞네
from sklearn.metrics import accuracy_score
pred = result.predict(test)
print('분류 정확도 : ', accuracy_score(test['외식유무'], np.around(pred))) # 분류 정확도 :  0.8571428571428571

# 키보드로 소득 수준 받기
new_소득수준 = float(input('소득수준 : '))
new_df = pd.DataFrame({'소득수준':[new_소득수준]})
new_pred = result.predict(new_df)
print('외식유무 : ', np.around(new_pred.values))


###################################################################################################
###################### 선생님꺼 ################################################################## 


# [로지스틱 분류분석 문제1]
# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
#
# 요일,외식유무,소득수준
# 토,0,57
# 토,0,39
# ...

# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import numpy as np
# from sklearn.metrics import accuracy_score
#
# fdata = pd.read_csv('로지문제1.txt')
# data = fdata.loc[(fdata['요일'] == '토') | (fdata['요일'] == '일')]
# print(data.head(3))
#
# model = smf.glm(formula='외식유무 ~ 소득수준', da    ta = data, family=sm.families.Binomial()).fit()
# print(model.summary())
# print()
# pred = model.predict(data)
# print('분류 정확도 : ', accuracy_score(data['외식유무'], np.around(pred))) 
#
# new_input_data = pd.DataFrame({'소득수준':[int(input('소득수준 : '))]})
# print('외식 유무 :', np.rint(model.predict(new_input_data)))
# print('외식을 함' if np.rint(model.predict(new_input_data))[0] == 1 else '외식안함')






