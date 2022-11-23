# 체질량지수(BMI)는 자신의 몸무게(kg)를 키의 제곱(m)으로 나눈 값입니다.
#
print(71 / ((178 / 100) * (178 / 100))) # 22.408786769347305

'''
# 데이터 직접 만들기
import random
random.seed(12)

def calc_bmi(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 25.0: return 'normal'
    return 'fat'

# print(calc_bmi(178, 71))

fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n')


# 무작위로 데이터 생성
cnt = {'thin':0, 'normal':0, 'fat':0}

for i in range(50000):
    h = random.randint(150, 200)
    w = random.randint(35, 100)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h, w, label))
    
fp.close()
'''
# 데이터 만들고 주석처리

import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tbl = pd.read_csv('bmi.csv')
print(tbl.head(3), tbl.shape)
#    height  weight   label
# 0     180      69  normal
# 1     192      79  normal
# 2     159      83     fat (50000, 3)
print(tbl.describe())

label = tbl['label']
print(label[:3])
# 0    normal
# 1    normal
# 2       fat

w = tbl['weight'] / 100 # 정규화
h = tbl['height'] / 200 # 정규화
print(w[:3])
print(h[:3])

wh = pd.concat([w, h], axis=1)
print(wh[:3], wh.shape)

# label을 dummy
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label[:3])

# train / test split
x_train, x_test , y_train, y_test = train_test_split(wh, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (35000, 2) (15000, 2) (35000,) (15000,)

print()
# model
model = svm.SVC(C=0.01).fit(x_train, y_train)

pred = model.predict(x_test)
print('예측값 : ', pred[:10]) # 예측값 :  [2 0 1 1 0 0 2 1 0 0]
print('실제값 : ', y_test[:10].values) # 실제값 :  [2 0 1 1 0 0 2 1 0 0]

acc = metrics.accuracy_score(y_test, pred)
print('acc : ', acc) # acc :  0.9705333333333334

print()
# 교차 검증
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, wh, label, cv = 3)
print('각각의 검증 정확도 : ', cross_vali) # 각각의 검증 정확도 :  [0.96940061 0.96586068 0.96681867]
print('평균 검증 정확도 : ', cross_vali.mean()) # 평균 검증 정확도 :  0.9673599891736715

# 시각화 
tbl2 = pd.read_csv('bmi.csv', index_col = 2)

def scatter_func(lb1, color):
    b = tbl2.loc[lb1]
    plt.scatter(b['weight'], b['height'], c=color, label=lb1)
    
scatter_func('fat', 'red')
scatter_func('normal', 'yellow')
scatter_func('thin', 'blue')
plt.legend()
plt.show()

# 새 값으로 예측
new_data = pd.DataFrame({'weight':[66, 55], 'height':[170, 180]})
# 정규화
new_data['weight'] = new_data['weight'] / 100
new_data['weight'] = new_data['height'] / 200
new_pred = model.predict(new_data)
print('새로운 예측값 : ', new_pred)















