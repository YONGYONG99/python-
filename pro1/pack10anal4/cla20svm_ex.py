# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv
#
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
#
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)
#
# 데이터 예)
# "","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
# "1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
# "2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
# ...

import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('../testdata/Heart.csv')
print(df.head(3))
print(df.info())
df.drop('ChestPain',  axis=1 , inplace=True)
df.drop('Thal',  axis=1 , inplace=True)
print(df.info())
df = df.fillna(df.mean())

AHD = df['AHD']
print(AHD[:3])
AHD = AHD.map({'No':0, 'Yes':1})
print(AHD[:3])

x = np.array(df.iloc[:, 1:12])
y = np.array(AHD)
print(x)
print(y)

# train / test split
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (212, 11) (91, 11) (212,) (91,)

print()
# model
model = svm.SVC().fit(x_train, y_train)

pred = model.predict(x_test)
print('예측값 : ', pred[:10])  # 예측값 :  [0 0 0 1 0 0 0 1 1 1]
print('실제값 : ', y_test[:10]) # 실제값 :  [0 0 0 1 1 0 0 1 1 1]

acc = metrics.accuracy_score(y_test, pred) 
print('acc : ', acc)   # acc :  0.6483516483516484

new_data = pd.DataFrame({'Age':[63,67], 'Sex':[1,1], 'RestBP':[145,160],'Chol':[233,286], 'Fbs':[1,0], 'RestECG':[2,2], 'MaxHR':[150,108], 'ExAng':[0,1], 'Oldpeak':[2.3,1.5], 'Slope':[3,2], 'Ca':[0,3]})
print(new_data)
new_pred = model.predict(new_data)
print('새로운 예측값 : ', new_pred) # 새로운 예측값 :  [0 1] 즉 NO, YES

'''
동현이 이상치 제거 , 스케일링

# 6. 이상치 확인
plt.boxplot(data)
plt.show()
# -> 4번 인덱스 칼럼 : 360이하
# -> 3번 인덱스 칼럼 : 170이하
# -> 4번, 7번 인덱스 칼럼 feature scaling 필요



# 7. 이상치 제거
# data = data[data.iloc[:, 4] <= 360]
# data = data[data.iloc[:, 3] <= 170]



# 8. feature scaling
# scaler = StandardScaler()
# data['Chol'] = scaler.fit_transform(data[['Chol']])
# data['MaxHR'] = scaler.fit_transform(data[['MaxHR']])
plt.boxplot(data)
plt.show()
'''

######################################### 선생님 꺼

'''
# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv
#
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
#
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)

import pandas as pd 
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection._split import train_test_split

heartdata = pd.read_csv("../testdata/Heart.csv")
print(heartdata.info())

data = heartdata.drop(["ChestPain", "Thal"], axis = 1)  # object type은 제외
data.loc[data.AHD=="Yes", 'AHD'] = 1
data.loc[data.AHD=="No", 'AHD'] = 0
print(heartdata.isnull().sum())      # Ca 열에 결측치 4개

Heart = data.fillna(data.mean())   # CA에 결측치는 평균으로 대체
label = Heart["AHD"]
features = Heart.drop(["AHD"], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.3, random_state = 12)
print()
model = svm.SVC(C=0.1).fit(x_train, y_train)
pred = model.predict(x_test)
print('예측값 : ', pred)
print('실제값 : ', np.array(y_test))

# 분류 정확도 
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print('분류 정확도 : ', metrics.accuracy_score(y_test, pred))

# 새 값으로 예측
new_test = x_test[:2].copy()
print(new_test)
new_test['Age'] = 10
new_test['Sex'] = 0
print(new_test)

new_pred = model.predict(new_test)
print('예측결과 : ', new_pred)
'''


