import pandas as pd

data = pd.read_csv('../testdata/titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
print(data.head(2), data.shape) # (891, 12)
data.loc[data["Sex"] == "male","Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
print(data["Sex"].head(2))
print(data.columns)

feature = data[["Pclass", "Sex", "Fare"]]
label = data["Survived"]

# 이하 소스 코드를 적으시오.
# 1) train_test_split (7:3), random_state=12
# 2) DecisionTreeClassifier 클래스를 사용해 분류 모델 작성
# 3) 분류 정확도 출력

import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split

# 1) train_test_split (7:3), random_state=12
x_train, x_test , y_train, y_test = train_test_split(feature,label, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (623, 3) (268, 3) (623,) (268,)

# 2) DecisionTreeClassifier 클래스를 사용해 분류 모델 작성
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth = 5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum())) # 총갯수:268, 오류수:67

# 3) 분류 정확도 출력
from sklearn.metrics import accuracy_score
print('분류정확도 : %.5f'%accuracy_score(y_test, y_pred)) # 분류정확도 : 0.75000













