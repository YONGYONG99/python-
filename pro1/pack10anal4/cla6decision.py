# 의사결정나무(Decision Tree)
# 의사결정나무는 데이터를 분석하여 이들 사이에 존재하는 패턴을 예측 가능한 규칙들의 조합으로 나타내며, 
# 그 모양이 ‘나무’와 같다고 해서 의사결정나무라 불립니다. 
# 질문을 던져서 대상을 좁혀나가는 ‘스무고개’ 놀이와 비슷한 개념입니다.
# 의사결정나무는 분류(classification)와 회귀(regression) 모두 가능합니다. 범주나 연속형 수치 모두 예측할 수 있다는 말입니다. 
# Entropy : 독립변수의 혼잡도. 0 ~ 1 사이의 값을 가지며 낮을수록 좋다.
# Information Gain : 지정된 속성이 얼마나 잘 학습 데이터들 구분하는가에 대한 수치.

import pydotplus
from sklearn import tree

 
# 키와 머리카락의 길이로 남녀 구분
x = [[180, 15],[177, 42],[156, 35],[174, 5],[166, 33],[170, 20],[167, 2],[186, 35],[170, 5],[176, 13]]
y = ['man', 'woman', 'woman', 'man', 'woman','woman','woman','man','woman','man']

label_names = ['height', 'hair length']

# 모델
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
print(model)
model.fit(x,y)
print('훈련 정확도 : ', model.score(x, y))

pred = model.predict(x)
print('예측값 : ', pred) # 예측값 :  ['man' 'woman' 'woman' 'man' 'woman']
print('실제값 : ', y) # 실제값 :  ['man', 'woman', 'woman', 'man', 'woman']

# 임의의 새로운 값에 대한 분류
mydata = [[171, 30]]
new_pred = model.predict(mydata)
print(new_pred)

# 시각화
dot_data = tree.export_graphviz(model, feature_names = label_names, 
                                out_file=None, filled = True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'orange')
import collections
edges = collections.defaultdict(list)

for e in graph.get_edge_list():
    edges[e.get_source()].append(int(e.get_destination()))

for e in edges:
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')

# 이미지 읽기
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
img = imread("tree.png")
plt.imshow(img)
plt.show() # entorpy 0 될때까지 더이상 나눌수 없을때까지 나누네




























































